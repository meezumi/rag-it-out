import streamlit as st
import os
import chromadb
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
import torch


# --- CONFIGURATION ---
CHROMA_HOST = os.environ.get("CHROMA_HOST", "localhost")
DOCUMENTS_DIR = "./documents"
COLLECTION_NAME = "local_scholar_collection"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL_NAME = "google/flan-t5-base"


# --- CLIENT SETUP ---
chroma_client = chromadb.HttpClient(host=CHROMA_HOST, port=8000)


# --- HELPER FUNCTIONS ---
@st.cache_resource
def load_llm_pipeline():
    """Loads the pre-trained language model and tokenizer."""
    tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(LLM_MODEL_NAME)
    device = 0 if torch.cuda.is_available() else -1
    pipe = pipeline(
        "text2text-generation",
        model=model,
        tokenizer=tokenizer,
        max_length=512,
        temperature=0.7,
        top_p=0.95,
        repetition_penalty=1.15,
        device=device,
    )
    return HuggingFacePipeline(pipeline=pipe)


@st.cache_resource
def load_embedding_model():
    """Loads the sentence transformer model for embeddings."""
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)


def process_documents():
    """Loads and processes documents for the RAG pipeline."""
    if not os.path.exists(DOCUMENTS_DIR) or not os.listdir(DOCUMENTS_DIR):
        st.warning(
            f"No documents found in the '{DOCUMENTS_DIR}' directory. Please add PDF files."
        )
        return

    with st.spinner("Processing documents... This may take a moment."):
        status_placeholder = st.empty()

        status_placeholder.info("1/4 - Loading PDF documents...")
        loader = PyPDFDirectoryLoader(DOCUMENTS_DIR)
        documents = loader.load()
        if not documents:
            st.warning("Could not load any documents from the PDF files.")
            return

        status_placeholder.info(
            f"2/4 - Splitting {len(documents)} document(s) into chunks..."
        )

        # Split documents into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200
        )
        texts = text_splitter.split_documents(documents)

        # Load embeddings model
        embeddings = load_embedding_model()

        # Create the vector store and add the documents
        # This will connect to the ChromaDB server and create/update the collection
        Chroma.from_documents(
            documents=texts,
            embedding=embeddings,
            client=chroma_client,
            collection_name=COLLECTION_NAME,
        )
        st.success(f"{len(documents)} document(s) processed successfully!")


def setup_qa_chain():
    """Sets up the RetrievalQA chain by connecting to the existing ChromaDB collection."""
    embeddings = load_embedding_model()

    # Load the existing vector store from the ChromaDB server
    db = Chroma(
        client=chroma_client,
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
    )

    retriever = db.as_retriever(search_kwargs={"k": 3})
    llm = load_llm_pipeline()

    template = """
    You are a helpful assistant. Use the following context to answer the question.
    If you don't know the answer, just say that you don't know, don't try to make up an answer.
    Context: {context}
    Question: {question}
    Answer:
    """
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
st.write(
    f"Upload research papers to the `{DOCUMENTS_DIR}` folder and ask questions about their content."
)

if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    st.header("Controls")
    if st.button("Process Documents"):
        process_documents()

    if st.button("Clear Database"):
        with st.spinner("Clearing vector database..."):
            try:
                chroma_client.delete_collection(name=COLLECTION_NAME)
                # --- ADDED: Clear conversation history when clearing DB ---
                st.session_state.messages = []
                st.success("Database and conversation history cleared successfully!")
            except Exception as e:
                st.error(f"An error occurred while clearing the database: {e}")

st.header("Conversation")
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# To start the QA chain
try:
    qa_chain = setup_qa_chain()

    if prompt := st.chat_input("Ask a question about your documents"):
        # Add user message to history
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Get assistant response
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            with st.spinner("Searching for answers..."):
                try:
                    result = qa_chain.invoke(
                        {"query": prompt}
                    )  # Use invoke for the latest LangChain
                    answer = result["result"]

                    # --- ADDED: Expander for sources ---
                    with st.expander("Show Source Documents"):
                        for doc in result["source_documents"]:
                            st.write(
                                f"**Source:** {os.path.basename(doc.metadata.get('source', 'Unknown'))}"
                            )
                            st.write(f"**Content:** {doc.page_content[:500]}...")

                    message_placeholder.markdown(answer)
                    # Add assistant response to history
                    st.session_state.messages.append(
                        {"role": "assistant", "content": answer}
                    )

                except Exception as e:
                    st.error(f"An error occurred during query execution: {e}")

except Exception as e:
    st.error(
        f"Failed to initialize the QA chain. Have you processed the documents yet? Error: {e}"
    )
