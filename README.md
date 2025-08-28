# RAG It Out

Created this project for learning Retrieval-Augmented Generation (RAG) and exploring related technologies.

## Tech Stack

- Python
- LangChain
- OpenAI API
- FAISS
- Streamlit

## Use Case

To build an application that retrieves relevant information from a custom dataset and generates context-aware responses using LLMs. It can be adapted for knowledge management, chatbots, or document search.

## Planned Architecture 

```
+----------------+      +-------------------+      +----------------+
|   PDF Docs     |----->|   Load & Chunk    |----->|    Embed Text  |
+----------------+      +-------------------+      +----------------+
      ^                                                   |
      |                                                   v
      | User adds files                     +----------------------+
      | to /documents                       | Chroma Vector DB     |
      |                                     +----------------------+
      |                                                   ^
      |                                                   | Semantic Search
      |  +---------------------------------+              |
      +--|          Streamlit UI           |<-------------+
         +---------------------------------+
         ^    ^               |          |
         |    | Ask Question  |          | Retrieve Context
         |    |               v          |
         |    +--------------------------+
         |    |      RAG Chain           |
         |    | (LangChain)              |
         |    +--------------------------+
         |                |
         |                | Augment Prompt
         | Answer         v
         |    +--------------------------+
         +----|       Flan-T5 LLM        |
              |   (Generate Answer)      |
              +--------------------------+

```
---