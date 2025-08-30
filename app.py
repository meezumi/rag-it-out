import streamlit as st
import os
import chromadb

from langchain_unstructured import UnstructuredLoader as UnstructuredDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    pipeline,
    BitsAndBytesConfig,
)
import torch


# --- CONFIGURATION ---
CHROMA_HOST = os.environ.get("CHROMA_HOST", "localhost")
DOCUMENTS_DIR = "./documents"
COLLECTION_NAME = "local_scholar_collection"

# --- Model Selection ---
AVAILABLE_EMBEDDING_MODELS = {
    "MiniLM (Fast, Default)": "sentence-transformers/all-MiniLM-L6-v2",
    "BGE Small (Recommended)": "BAAI/bge-small-en-v1.5",
}

AVAILABLE_LLMS = {
    "Flan-T5 Base (Fast, Default)": "google/flan-t5-base",
    "Flan-T5 Large (Better Quality)": "google/flan-t5-large",
}


# --- CLIENT SETUP ---
chroma_client = chromadb.HttpClient(host=CHROMA_HOST, port=8000)


# --- HELPER FUNCTIONS ---
@st.cache_resource
def load_llm_pipeline(model_name):
    st.info(f"Loading LLM: {model_name}...")

    quantization_config = BitsAndBytesConfig(load_in_8bit=True)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_name,
        quantization_config=quantization_config,
        device_map="auto",  # Automatically use GPU if available
    )
    
    pipe = pipeline(
        "text2text-generation",
        model=model,
        tokenizer=tokenizer,
        max_length=512,
        temperature=0.7,
        top_p=0.95,
        repetition_penalty=1.15,
    )
    return HuggingFacePipeline(pipeline=pipe)


@st.cache_resource
def load_embedding_model(model_name):
    st.info(f"Loading embedding model: {model_name}...")
    return HuggingFaceEmbeddings(model_name=model_name, model_kwargs={"device": "cpu"})


def process_documents(embedding_model_name):
    if not os.path.exists(DOCUMENTS_DIR) or not os.listdir(DOCUMENTS_DIR):
        st.warning(
            f"No documents found in the '{DOCUMENTS_DIR}' directory. Please add PDF files."
        )
        return

    with st.spinner("Processing documents... This may take a moment."):
        status_placeholder = st.empty()

        status_placeholder.info("1/4 - Loading documents with Unstructured.io...")
        # --- Using UnstructuredDirectoryLoader for better PDF parsing ---
        loader = UnstructuredDirectoryLoader(DOCUMENTS_DIR, silent_errors=True)
        documents = loader.load()

        if not documents:
            st.warning("Could not load any documents.")
            return

        status_placeholder.info(
            f"2/4 - Splitting {len(documents)} document(s) into chunks..."
        )
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200
        )
        texts = text_splitter.split_documents(documents)

        status_placeholder.info(
            f"3/4 - Loading embedding model: {embedding_model_name}..."
        )
        embeddings = load_embedding_model(embedding_model_name)

        status_placeholder.info(
            "4/4 - Storing embeddings... This may take a while for new models."
        )
        Chroma.from_documents(
            documents=texts,
            embedding=embeddings,
            client=chroma_client,
            collection_name=COLLECTION_NAME,
        )
        status_placeholder.empty()
        st.success(f"{len(documents)} document(s) processed successfully!")


def setup_qa_chain(embedding_model_name, llm_model_name):
    embeddings = load_embedding_model(embedding_model_name)
    db = Chroma(
        client=chroma_client,
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
    )
    retriever = db.as_retriever(search_kwargs={"k": 3})
    llm = load_llm_pipeline(llm_model_name)
    template = "Use the following pieces of context to answer the user's question. If you don't know the answer, just say that you don't know, don't try to make up an answer. \nContext: {context} \nQuestion: {question} \nHelpful Answer:"
    QA_CHAIN_PROMPT = PromptTemplate.from_template(template)
    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": QA_CHAIN_PROMPT},
    )


# --- STREAMLIT UI ---
st.set_page_config(page_title="RAG Documentor", layout="wide")
st.title("RAG IT OUT")
st.write(f"An advanced RAG application to chat with your documents locally.")

if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    st.header("Configuration")

    st.selectbox(
        "Choose Embedding Model",
        options=list(AVAILABLE_EMBEDDING_MODELS.keys()),
        key="embedding_model_key",
    )
    st.selectbox("Choose LLM", options=list(AVAILABLE_LLMS.keys()), key="llm_key")

    embedding_model_name = AVAILABLE_EMBEDDING_MODELS[
        st.session_state.embedding_model_key
    ]
    llm_model_name = AVAILABLE_LLMS[st.session_state.llm_key]

    st.header("Controls")
    if st.button("Process Documents"):
        # Need to clear the database if the embedding model changes
        st.info("Clearing old database before processing with new embedding model...")
        try:
            chroma_client.delete_collection(name=COLLECTION_NAME)
        except Exception:
            st.warning("Collection not found to delete, proceeding.")
        process_documents(embedding_model_name)

    if st.button("Clear Database & Chat"):
        with st.spinner("Clearing..."):
            try:
                chroma_client.delete_collection(name=COLLECTION_NAME)
                st.session_state.messages = []
                st.success("Database and chat history cleared.")
            except Exception as e:
                st.error(f"Error: {e}")

st.header("Conversation")
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# To start the QA chain
try:
    qa_chain = setup_qa_chain(embedding_model_name, llm_model_name)

    if prompt := st.chat_input("Ask a question about your documents"):
        # Add user message to history
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Get assistant response
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            with st.spinner("Thinking..."):
                result = qa_chain.invoke({"query": prompt})
                answer = result["result"]
                with st.expander("Show Source Documents"):
                    for doc in result["source_documents"]:
                        st.write(f"**Source:** {os.path.basename(doc.metadata.get('source', 'Unknown'))}")
                        st.write(f"**Content:** {doc.page_content[:500]}...")
                message_placeholder.markdown(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})

except Exception as e:
    st.error(f"An error occurred. Did you process the documents with the selected embedding model? Error: {e}")

except Exception as e:
    st.error(
        f"Failed to initialize the QA chain. Have you processed the documents yet? Error: {e}"
    )
