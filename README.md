# RAG(Retrieval-Augmented Generation) It Out

Created this project for learning Retrieval-Augmented Generation (RAG) and exploring related technologies. The application is a simple Question-Answering tool that runs entirely on the local machine. We can feed it research papers (or any PDFs), and then ask it specific questions about the content. The entire stack is built on free, open-source tools and is containerized with Docker, making setup is very simple.

## Features

*   **Chat with your PDFs:** Ask questions in natural language and get answers based on the content of the documents provided.
*   **Selectable AI Models:** Choose between different embedding models and Large Language Models (LLMs) to see how they impact performance.
*   **Built-in Chat History:** Keeps track of the current conversation.
*   **Easy Data Management:** Process new documents or clear the knowledge base with the click of a button.

## Tech Stack

- **Backend & Orchestration**: Python, LangChain
- **Web Interface**: Streamlit
- **Embedding Model**: Sentence-Transformers (all-MiniLM-L6-v2)
- **LLM (Generator)**: Google's Flan-T5 (Hugging Face Transformers)
- **Vector Database**: ChromaDB (for semantic search)
- **Containerization**: Docker & Docker Compose

## Use Case ##

To build an application that retrieves relevant information from a custom dataset and generates context-aware responses using LLMs. It can be adapted for knowledge management, chatbots, or document search. How is it different from it's alternatives is:

- If I were to ask about a past popular research paper, the LLM would have generated an acceptable answer, that won't be the case for a recent and relatively new paper. That's where RAG comes in. This will: <br>
      1. **Prevent hallucilations & refusal** of having any valid information on the paper. <br>
      2. We could have fine tuned the LLM with a lot of Q&A papers, but that would be slow and need significant computational power, RAG solves this **efficiently.** <br>
      3. We **can fact check the info provided** by the LLM, since we have the source on the basis of what it is responding to us. (_have implimente this in the project_) <br>


## Project Architecture 

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
## Setup and Installation

Follow these steps to get the project running on your machine.

### Prerequisites

You must have **Docker Desktop** installed and running on your system. This single application handles both `docker` and `docker-compose`.

*   [Download Docker Desktop](https://www.docker.com/products/docker-desktop/)

No other dependencies like Python or AI libraries are needed on your host machine—Docker takes care of everything.

### Step-by-Step Instructions

1.  **Clone the Repository**
    Open your terminal or command prompt and clone this project to your local machine.
    ```bash
    git clone https://github.com/meezumi/rag-it-out.git
    ```
    Navigate into the newly created project directory:
    ```bash
    cd rag-it-out
    ```

2.  **Add Your PDF Documents**
    Place any PDF files you want to analyze into the `documents/` folder. You can add one or more files.

3.  **Build and Run the Application**
    From the root of the project directory (`rag-it-out/`), run the following command in your terminal:
    ```bash
    docker-compose up --build
    ```
    **Note:** The very first time you run this command, it will take several minutes to complete. Docker needs to:
    *   Download the base Python image.
    *   Install all the required Python libraries.
    *   Download the AI models from Hugging Face (several gigabytes).
    
    Subsequent launches will be much faster.

4.  **Access the Application**
    Once the build is complete and the logs in your terminal indicate that the services are running, open your web browser and go to:
    > **http://localhost:8501**
    
--- 

