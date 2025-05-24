# Advanced Agentic RAG System for House Sales Data & More

## 1. Overview

This project documents the development of an advanced, multi-tool agentic AI system. It was built iteratively to explore and demonstrate sophisticated AI capabilities, starting from fundamental concepts and progressing to complex agentic behaviors. The system is designed to interact with the King County House Sales dataset, providing insights through a combination of Retrieval Augmented Generation (RAG) over textual data, direct SQL queries on structured data, real-time web searches for external information, and code execution for dynamic calculations.

The primary interface is a Streamlit-based web application, featuring an administrative panel for data management and an interactive chat interface for users to converse with the AI agent. This project serves as a comprehensive example of building a complex AI system, highlighting architectural choices, tool integration, prompt engineering, and iterative problem-solving.

---

## 2. Key Features

This system incorporates a range of advanced features to provide comprehensive and intelligent responses:

* **Multi-Tool Agent Orchestration:**
    * A central LangChain agent, built using `create_tool_calling_agent` and `AgentExecutor`, intelligently orchestrates multiple specialized tools. This architecture allows the agent to select the most appropriate capability based on the user's query, leading to more accurate and relevant answers.

* **Hybrid Data Retrieval and Querying:**
    * **RAG Tool (`HouseSalesInfoTool`):** Performs semantic search over textual descriptions of house sales data (stored in Qdrant). This tool is enhanced with a `CrossEncoder` model for re-ranking search results, significantly improving the relevance of context provided for LLM synthesis.
    * **SQL Query Tool (`SQLQueryTool`):** Interacts with a PostgreSQL database containing structured house sales data. It features an internal, specialized SQL sub-agent (created via `create_sql_agent`) capable of:
        * Generating SQL queries from natural language.
        * Inspecting table schemas and sample data.
        * Executing queries and synthesizing results.
        * Performing self-correction on SQL errors by re-analyzing and rewriting queries.
    * **Web Search Tool (`TavilySearchResults`):** Connects to the Tavily Search API to fetch and utilize real-time information from the internet, covering current events, general knowledge, or topics not present in the local datasets.
    * **Python REPL Tool (`PythonREPLTool`):** Provides the agent with a code interpreter to execute Python code. This enables dynamic calculations (e.g., price per square foot, percentage changes), data manipulation on retrieved information, and answering mathematical or logical questions.

* **Advanced Agentic Reasoning and Planning:**
    * The main agent employs a sophisticated system prompt that guides it through a multi-step thought process: analyzing the query, detecting key entities (like property IDs), formulating a plan (which may involve sequential tool use), strategically selecting tools, and synthesizing information from diverse sources into a coherent final answer.
    * It demonstrates the ability to break down complex, multi-faceted user questions into manageable sub-tasks, addressing each with the most suitable tool. For example, it can find a specific house using SQL, then use its details to perform a calculation with the Python REPL, and finally search the web for related contextual information.

* **Dynamic LLM Backend:**
    * The system is configured via `src/llm_services/llm_client.py` to dynamically select between Large Language Models, with a preference for Google's Gemini models and a fallback to OpenAI's GPT models, based on API key availability.
    * Includes `nest_asyncio` to resolve asyncio event loop conflicts, particularly for Gemini client initialization within the Streamlit environment.

* **Interactive User Interface (Streamlit):**
    * Provides a user-friendly web interface for interaction.
    * **Administrative Panel:** Facilitates data management tasks, including loading the raw CSV dataset, initializing or recreating the Qdrant vector collection, showing collection information, and triggering the preprocessing and ingestion of data into Qdrant. It also provides guidance for populating the SQL database.
    * **Chat Interface:** Allows users to converse naturally with the multi-tool agent, maintaining session-based chat history for contextual understanding.

* **Containerized and Orchestrated Environment (Docker):**
    * The entire application stack (Python application, Qdrant vector database, PostgreSQL relational database) is containerized using Docker.
    * `docker-compose.yml` orchestrates these services, ensuring consistent setup, networking, and data persistence through named volumes.

---

## 3. Tech Stack & Architecture

The project leverages a modern stack of AI and data technologies:

* **Core Framework:** LangChain (for agent creation, tool management, chains, prompts, LLM integration)
* **Large Language Models (LLMs):**
    * Google Gemini (via `langchain-google-genai`) - Preferred
    * OpenAI GPT models (via `langchain-openai`) - Fallback
* **Embedding Model:** `all-MiniLM-L6-v2` (from `sentence-transformers`) for generating vector embeddings of property descriptions.
* **Re-ranking Model:** `cross-encoder/ms-marco-MiniLM-L-6-v2` (from `sentence-transformers`) for improving RAG retrieval relevance.
* **Databases:**
    * **Vector Database:** Qdrant (version pinned for stability, e.g., `v1.9.2`) for efficient semantic search.
    * **Relational Database:** PostgreSQL (version pinned, e.g., `postgres:15`) for storing and querying structured house sales data.
* **Data Processing & Manipulation:** Pandas, NumPy.
* **Web Search Integration:** Tavily Search API (via `tavily-python`).
* **User Interface:** Streamlit.
* **Containerization & Orchestration:** Docker, Docker Compose.
* **Python Version:** 3.10
* **Environment Management:** Conda.
* **Supporting Libraries:** `psycopg2-binary` (PostgreSQL adapter), SQLAlchemy (used by LangChain SQL tools), `python-dotenv` (environment variable management), `nest-asyncio` (asyncio compatibility).

**Architectural Flow:**
A user interacts with the Streamlit UI. User input is passed to the main `AgentExecutor`. The agent, guided by its system prompt and the descriptions of available tools, decides whether to use a tool or respond directly. If a tool is chosen (e.g., `HouseSalesInfoTool`, `SQLQueryTool`, `TavilySearchResults`, `PythonREPLTool`), the agent invokes it. Tools like `HouseSalesInfoTool` and `SQLQueryTool` may have their own internal LLM calls for synthesis or sub-agent operations. The observation from the tool is returned to the main agent, which then either performs further steps (more tool calls) or synthesizes the final answer presented back to the user via the UI.

---

## 4. Setup and Installation Guide

To set up and run this project locally, please follow these steps:

**4.1. Prerequisites:**
    * Ensure [Anaconda](https://www.anaconda.com/products/distribution) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html) is installed.
    * Ensure [Docker Desktop](https://www.docker.com/products/docker-desktop/) is installed and **running**.
    * Ensure Git is installed.

**4.2. Clone the Repository:**
    (If applicable, otherwise, ensure you are in the project's root directory)
    ```bash
    # git clone <your-repository-url>
    # cd agentic_rag_project
    ```

**4.3. Create and Activate Conda Environment:**
    ```bash
    conda create -n agentic_rag python=3.10 -y
    conda activate agentic_rag
    ```

**4.4. Install Python Dependencies:**
    A `requirements.txt` file should be present in the project root, listing all necessary Python packages.
    ```bash
    pip install -r requirements.txt
    ```
    This file should include: `langchain`, `langchain-google-genai`, `langchain-openai`, `langchain-community`, `langchain-experimental`, `streamlit`, `pandas`, `numpy`, `qdrant-client`, `sentence-transformers`, `python-dotenv`, `nest-asyncio`, `tavily-python`, `sqlalchemy`, `psycopg2-binary`, and any other specific versions used.

**4.5. Configure Environment Variables:**
    1.  In the project root directory (e.g., `agentic_rag_project/`), create a file named `.env`.
    2.  Add your necessary API keys to this file:
        ```env
        GEMINI_API_KEY="your_actual_gemini_api_key"
        OPENAI_API_KEY="your_actual_openai_api_key_if_you_plan_to_use_it"
        TAVILY_API_KEY="your_actual_tavily_api_key"
        
        # The DATABASE_URL for the app container is set in docker-compose.yml.
        # This entry is primarily for the populate_sql_db.py script if run locally
        # or as a reference. For Docker, the app uses the env var from docker-compose.
        # Example for local connection if postgres is on host:
        # DATABASE_URL="postgresql://user:password@localhost:5432/housedata" 
        ```
    * Replace `"your_actual_..._key"` with your real API keys.

**4.6. Build and Run Docker Containers:**
    Navigate to the project root directory in your terminal (where `docker-compose.yml` is located).
    1.  To stop any existing project containers and remove them:
        ```bash
        docker compose down
        ```
    2.  To build (or rebuild) the application image and start all services (app, Qdrant, PostgreSQL) in detached mode:
        ```bash
        docker compose up --build -d
        ```
    * The first build process can take several minutes as it downloads base images and installs Python dependencies.
    * Allow a moment for the `postgres_db` and `qdrant_db` containers to initialize fully. You can check their status with `docker compose ps` and their logs with `docker logs postgres_db` or `docker logs qdrant_db`.

**4.7. Initial Data Population:**
    * **Populate PostgreSQL Database:**
        Once the `app` and `postgres_db` containers are running (check with `docker compose ps`), execute the SQL population script from your project root on your host machine:
        ```bash
        docker compose exec app python src/ingestion/populate_sql_db.py
        ```
        This script will connect to the PostgreSQL container, create the `house_sales` table, and ingest a sample of data (default: 1000 rows). Monitor the terminal output for success messages or errors.

    * **Populate Qdrant Vector Database:**
        1.  Open the Streamlit application in your web browser, typically at `http://localhost:8501`.
        2.  Use the **Admin Panel** located in the sidebar:
            * Click **"Load Raw Data from CSV"**. Wait for the success message.
            * Click **"Initialize/Recreate Qdrant Collection"**. This ensures the collection is set up correctly.
            * In the "Ingest Data into Qdrant" section, set the **"Number of rows to ingest into Qdrant"** (e.g., 100 to 1000 for initial testing, or 0 for all loaded data).
            * Click **"Preprocess & Populate Qdrant"**. This process involves generating descriptions, embedding them, and upserting to Qdrant, which may take some time depending on the number of rows.
        3.  Monitor the Streamlit UI and the Docker logs for the `rag_app` container (`docker logs rag_app -f`) for progress and any potential error messages.
        4.  Optionally, click "Show Qdrant Collection Info" to verify points have been added. You can also access the Qdrant dashboard at `http://localhost:6333/dashboard`.

---

## 5. How to Use the Application

1.  **Ensure all services are running:** After completing the setup, verify with `docker compose ps` that `rag_app`, `qdrant_db`, and `postgres_db` are up.
2.  **Access the UI:** Open your web browser and navigate to `http://localhost:8501`.
3.  **Data Check:** If you haven't populated the databases, use the Admin Panel as described in the setup instructions.
4.  **Interact with the Agent:**
    * Use the chat input field under the "Chat With Multi-Tool Agent" header.
    * Ask various types of questions to test the agent's capabilities:
        * **ID-specific SQL lookups:** "Tell me all details for house ID 5631500400." (Ensure this ID is in your SQL sample).
        * **Aggregations via SQL:** "What is the average price of houses in zipcode 98178?"
        * **Descriptive RAG queries:** "What are some common features of renovated houses that have a good view?"
        * **Web searches:** "What are the latest real estate news for King County?"
        * **Calculations via Python REPL:** "If a house price is $650,000 and the living area is 2000 sqft, what's the price per sqft?"
        * **Complex, multi-step queries:** "What are the features of the most expensive house sold in zipcode 98004, and what was its sale date? Also, what is the current average 30-year mortgage rate?"
5.  **Monitor Agent Behavior:** For development and understanding, keep an eye on the Docker logs of the `rag_app` container (`docker logs rag_app -f`). The `verbose=True` setting for the `AgentExecutor` will output the agent's thought process, tool selections, inputs to tools, and observations received.

---

## 6. Project Structure Overview

agentic_rag_project/
├── .env              
├── .gitignore            
├── docker-compose.yml    # Defines and orchestrates the multi-container Docker application.
├── Dockerfile            # Instructions to build the Docker image for the Python application.
├── requirements.txt      # Lists Python package dependencies for the project.
├── data/                 # Contains datasets used by the project.
│   └── raw/              # Raw, original data files (e.g., kc_house_data.csv).
│   └── processed/        # (Optional) For storing data after cleaning/transformation.
├── src/                  # Main source code for the application.
│   ├── init.py       # Makes 'src' a Python package.
│   ├── ingestion/        # Scripts for data loading and preprocessing.
│   │   ├── init.py
│   │   ├── load_data.py         # Loads raw data from CSV.
│   │   ├── populate_sql_db.py   # Populates the PostgreSQL database.
│   │   └── preprocess.py      # Cleans data and generates descriptions for RAG.
│   ├── knowledge_base/   # Modules for interacting with databases.
│   │   ├── init.py
│   │   └── vector_store.py    # Manages Qdrant vector database interactions.
│   ├── llm_services/     # Logic for LLM client initialization.
│   │   ├── init.py
│   │   └── llm_client.py      # Handles dynamic selection of LLMs (Gemini/OpenAI).
│   ├── tools/            # Definitions for custom LangChain tools.
│   │   ├── init.py
│   │   ├── house_sales_tool.py # RAG tool for house sales descriptions.
│   │   └── sql_database_tool.py  # Tool for querying the SQL house sales database.
│   ├── config.py         # Project-wide configurations and constants.
│   └── main_chatbot.py   # The main Streamlit application file (UI and agent orchestration).
└── README.md            

---

## 7. Key Learnings and Development Journey

This project has been an extensive learning experience, demonstrating the iterative nature of building advanced AI systems:

* **From Pipeline to Agent:** The initial RAG system evolved from a fixed pipeline to a flexible, tool-using LangChain agent, significantly enhancing its capabilities and decision-making autonomy.
* **The Power of Prompt Engineering:** The agent's behavior, tool selection accuracy, and the quality of its final synthesized answers are heavily influenced by the clarity, detail, and strategic guidance provided in its system prompt. Multiple iterations were required to achieve effective multi-step reasoning.
* **Tool Design and Integration:** Designing self-contained, well-described tools is crucial for an agentic system. The `HouseSalesInfoTool` (with RAG and re-ranking), `SQLQueryTool` (with its own internal SQL agent capable of self-correction), `TavilySearchResults`, and `PythonREPLTool` each provide distinct, valuable capabilities.
* **Importance of Data Handling:** Robust data ingestion, preprocessing (especially the generation of quality textual descriptions for RAG), and management across different database types (vector, relational) are foundational.
* **Debugging Complex AI Systems:** Effective debugging relies on:
    * `verbose=True` mode in LangChain agents to inspect the thought-action-observation loop.
    * Comprehensive logging within custom tools and service components.
    * Systematic testing with a variety of query types.
    * Iterative refinement based on observed failures or suboptimal responses.
* **Overcoming Technical Challenges:** The development journey involved resolving:
    * Python import path complexities within a packaged structure and Docker.
    * Asynchronous programming (asyncio) event loop conflicts between libraries (like Google's GenAI SDK) and the Streamlit environment, resolved using `nest_asyncio`.
    * Pydantic type errors in LangChain tool definitions.
    * Database stability issues (e.g., Qdrant storage corruption, version mismatches), addressed by clearing storage, pinning image versions, and aligning client/server versions.

---

## 8. Potential Future Work and Enhancements

While the current system is highly capable, several avenues exist for further advancement:

* **Formal Evaluation Framework (Phase 5):**
    * Implement a robust evaluation pipeline using frameworks like RAGAS or LangSmith.
    * Develop a comprehensive evaluation dataset (questions, ground truth contexts, ground truth answers) to quantitatively measure aspects like faithfulness, answer relevance, context precision/recall, and tool usage accuracy. This is critical for systematic improvement.

* **Advanced Agent Architectures (Phase 3):**
    * For tasks requiring even more complex, stateful, or cyclical reasoning (e.g., extended clarification dialogues, tasks with memory-dependent steps), explore **LangGraph** to explicitly define agentic workflows as graphs.

* **Memory and Personalization (Phase 3):**
    * Implement more sophisticated **conversational memory** (e.g., summarization memory, knowledge graph memory) to allow the agent to maintain context over longer interactions.
    * Introduce **user profiles** to store preferences or past interaction summaries, enabling personalized responses and behavior.

* **Proactive Agent Capabilities (Phase 3):**
    * Explore mechanisms for the agent to initiate actions or provide suggestions based on context, data changes (if monitored), or learned user needs, moving beyond purely reactive responses.

* **Knowledge Graph Integration (Phase 3):**
    * Extract entities and relationships from the house sales data (and potentially other sources) to build a knowledge graph (e.g., using Neo4j).
    * Develop a tool for the agent to query this knowledge graph, enabling it to answer questions about complex relationships and perform multi-hop reasoning not easily achieved by SQL or semantic search alone.

* **Enhanced Error Handling and Self-Correction by the Agent:**
    * Improve the agent's ability to recover from failed tool calls (e.g., by rephrasing input to a tool, trying an alternative tool, or explicitly asking the user for help).
    * Incorporate uncertainty quantification in its responses.

* **UI/UX Improvements (Phase 4):**
    * Integrate richer data visualizations (e.g., charts for price trends from SQL, maps for locations).
    * Implement more interactive elements or feedback mechanisms for users.

* **Productionization Aspects (Phase 4 & 5):**
    * Cloud deployment (e.g., AWS, GCP, Azure) with appropriate scaling.
    * CI/CD pipelines for automated testing and deployment.
    * Advanced logging, monitoring, and alerting for a production environment.
    * Thoroughly address any remaining stability warnings or minor errors (e.g., the Torch/Streamlit event loop error if it proves problematic).

---

This project has successfully demonstrated the creation of an advanced agentic RAG system, laying a strong foundation for these exciting future developments.
