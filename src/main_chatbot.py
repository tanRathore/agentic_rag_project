# src/main_chatbot.py
import streamlit as st
import pandas as pd
import os 

# Project-specific imports
from ingestion.load_data import load_raw_data
from ingestion.preprocess import preprocess_data_for_vectorization
from knowledge_base.vector_store import VectorDB
from llm_services.llm_client import get_llm
from tools.house_sales_tool import HouseSalesInfoTool
from tools.sql_database_tool import SQLQueryTool 

# LangChain imports
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.messages import AIMessage, HumanMessage
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.utilities import SQLDatabase
from langchain_experimental.tools import PythonREPLTool

import config 

st.set_page_config(page_title="Advanced Agentic System", layout="wide")
st.title("🧠 Advanced Agentic System - House Sales, Web, SQL & Code Interpreter")
@st.cache_resource 
def get_vector_db_instance():
    st.write("Initializing VectorDB and embedding model...")
    db = VectorDB()
    if db.is_ready():
        st.success("VectorDB and embedding model initialized successfully!")
    elif db.client and not db.embedding_model:
        st.warning("VectorDB connected, but embedding model failed to load. Embedding/Search features will not work.")
    else:
        st.error("Failed to initialize VectorDB or load embedding model. See Docker logs.")
    return db

vector_db_instance = get_vector_db_instance()

@st.cache_resource 
def get_llm_instance_cached():
    st.write("Initializing LLM for Agent and Tools...")
    llm_instance_val = get_llm(provider_preference="gemini")
    if llm_instance_val:
        st.success(f"LLM initialized successfully (using {llm_instance_val.__class__.__name__})!")
    else:
        st.error("Failed to initialize LLM. Check API keys and llm_client.py. See Docker logs.")
    return llm_instance_val

llm = get_llm_instance_cached()

@st.cache_resource 
def get_sql_database_connection_cached():
    db_url_env = os.getenv("DATABASE_URL")
    if not db_url_env:
        st.warning("DATABASE_URL not found in environment. SQL tool will be disabled.")
        return None
    try:
        display_db_url = db_url_env
        if "@" in db_url_env and ":" in db_url_env.split("@")[0]:
            creds, rest = db_url_env.split("@", 1)
            user, _ = creds.split(":", 1)
            display_db_url = f"{user}:********@{rest}"
        st.write(f"Initializing SQLDatabase connection to {display_db_url.split('@')[-1]}...")
        sqldb = SQLDatabase.from_uri(db_url_env, sample_rows_in_table_info=2, include_tables=['house_sales'])
        sqldb.get_usable_table_names() 
        st.success("SQLDatabase connection established successfully!")
        return sqldb
    except Exception as e:
        st.error(f"Failed to initialize SQLDatabase connection: {e}")
        import traceback
        print(traceback.format_exc())
        return None

sql_db_conn_instance = get_sql_database_connection_cached()


@st.cache_resource
def get_tools_list_cached(_vector_db, _llm_for_tools, _sql_db_conn):
    tools_list = []
    # Initialize House Sales Tool (RAG for Qdrant)
    if _vector_db and _vector_db.is_ready() and _llm_for_tools:
        st.write("Initializing HouseSalesInfoTool...")
        try:
            house_tool = HouseSalesInfoTool(vector_db=_vector_db, llm_for_synthesis=_llm_for_tools)
            tools_list.append(house_tool)
            st.success("HouseSalesInfoTool initialized!")
        except Exception as e:
            st.error(f"Failed to initialize HouseSalesInfoTool: {e}")
    else:
        st.warning("HouseSalesInfoTool could not be fully initialized (VectorDB or LLM not ready).")

    # Initialize Tavily Search Tool (Web Search)
    tavily_api_key = os.getenv("TAVILY_API_KEY")
    if tavily_api_key:
        st.write("Initializing TavilySearchResults Tool...")
        try:
            tavily_tool = TavilySearchResults(max_results=3)
            tavily_tool.name = "web_search"
            tavily_tool.description = "A search engine for finding up-to-date information on the internet, current events, general knowledge, or topics not covered by other specialized databases."
            tools_list.append(tavily_tool)
            st.success("TavilySearchResults Tool initialized!")
        except Exception as e:
            st.error(f"Failed to initialize TavilySearchResults Tool: {e}")
    else:
        st.warning("Tavily API key not found in environment. Web search tool will not be available.")

    # Initialize SQL Query Tool
    if _sql_db_conn and _llm_for_tools:
        st.write("Initializing SQLQueryTool...")
        try:
            # This now uses the globally initialized db_connection_instance from sql_database_tool.py
            # or the one created in this file if you prefer to pass it explicitly
            sql_tool = SQLQueryTool(llm_for_sql_agent=_llm_for_tools, database_connection=_sql_db_conn)
            if sql_tool.sql_agent_executor:
                tools_list.append(sql_tool)
                st.success("SQLQueryTool initialized successfully!")
            else:
                st.warning("SQLQueryTool initialized, but its internal SQL agent executor failed to create.")
        except Exception as e:
            st.error(f"Failed to initialize SQLQueryTool: {e}")
    elif not _sql_db_conn:
        st.warning("SQL Database connection not available. SQLQueryTool will be disabled.")
    
    # Initialize Python REPL Tool
    st.write("Initializing PythonREPLTool...")
    try:
        python_repl_tool = PythonREPLTool()
        tools_list.append(python_repl_tool)
        st.success("PythonREPLTool initialized!")
    except Exception as e:
        st.error(f"Failed to initialize PythonREPLTool: {e}")
    
    return tools_list

tools = []
if llm: 
    tools = get_tools_list_cached(vector_db_instance, llm, sql_db_conn_instance)


@st.cache_resource 
def get_agent_executor_cached(_llm_for_agent, _tools_list_for_agent):
    if not _llm_for_agent:
        st.warning("Cannot create agent: LLM not available.")
        return None
    if not _tools_list_for_agent:
        st.warning("Cannot create agent: No tools available. Check tool initializations and API keys.")
        return None
        
    st.write(f"Creating LangChain Agent with {len(_tools_list_for_agent)} tool(s) (Enhanced Multi-Step Reasoning)...")
    
    # Dynamically get tool names from the initialized tools list for the prompt
    tool_info_for_prompt = "\n".join([f"- Tool Name: '{tool.name}', Description: {tool.description}" for tool in _tools_list_for_agent])
    
    sql_tool_name = next((tool.name for tool in _tools_list_for_agent if isinstance(tool, SQLQueryTool)), "the_SQL_database_tool")
    rag_tool_name = next((tool.name for tool in _tools_list_for_agent if isinstance(tool, HouseSalesInfoTool)), "the_house_listings_information_tool")
    web_search_name = next((tool.name for tool in _tools_list_for_agent if tool.name == "web_search"), "the_web_search_tool")
    python_repl_name = next((tool.name for tool in _tools_list_for_agent if tool.name == "python_repl"), "the_Python_code_interpreter")

    # System Prompt for Multi-Step Task Decomposition and Better Data Utilization
    system_message_content = (
        "You are a meticulous and resourceful AI assistant specializing in King County house sales and general information retrieval. Your goal is to provide accurate, comprehensive, and well-synthesized answers.\n\n"
        "You have access to the following tools:\n"
        f"{tool_info_for_prompt}\n\n" 
        "**Your Critical Thought Process and Tool Usage Strategy:**\n"
        "1.  **Analyze the User's Full Request:** Deconstruct the user's request. Identify all distinct pieces of information needed. Is it a simple lookup, a comparison, an aggregation, a multi-part question?\n"
        "2.  **Formulate a Step-by-Step Plan:** If the query is complex or requires multiple pieces of information, explicitly outline your plan in your thought process. For example: 'To answer this, I first need to [Action 1 using Tool A to find X]. Then, I need to [Action 2 using Tool B to find Y based on X]. Finally, I will synthesize this information.'\n"
        "3.  **Strategic Tool Selection and Execution:**\n"
        f"    - **Property ID Lookup:** If a specific Property ID is clearly provided (e.g., 'details for ID 12345'), your **ABSOLUTE FIRST STEP** is to use the '{sql_tool_name}' with a query like `SELECT * FROM house_sales WHERE id = [ID];`. This tool should provide all structured details for that ID.\n"
        f"    - **SQL for Structured Data:** For questions about precise numbers, counts, averages, sums, min/max, or complex filtering/sorting on attributes in the 'house_sales' table (price, bedrooms, sqft, yr_built, zipcode, date): Use the '{sql_tool_name}'. Ensure your query requests all columns needed for the answer or subsequent steps.\n"
        f"    - **RAG for Descriptive Info:** For general characteristics, textual descriptions of listings, or semantic search on property features: Use the '{rag_tool_name}'. If an ID lookup with the SQL tool was insufficient for descriptive needs (e.g., you only got structured data but the user asked 'tell me *about* it'), you can use this tool with a query like 'description for property ID XXXXX'.\n"
        f"    - **Web Search:** For current events, real-time information (e.g., 'current mortgage rates'), or topics outside local databases: Use the '{web_search_name}'.\n"
        f"    - **Code Interpreter:** For calculations (percentages, ratios, custom aggregations on data from other tools), data manipulation, or math questions: Use the '{python_repl_name}'. Input must be Python code that `print()`s its result.\n"
        "4.  **Information Extraction from Tool Observations:** When a tool (especially the SQL tool) returns data, carefully examine the *entire observation*. If it provides multiple columns or a list of items, extract all relevant pieces of information needed for your plan or the final answer. Do not just rely on a brief summary if more detail is present in the observation.\n"
        "5.  **Iterative Refinement:** If a tool doesn't yield the exact information needed, assess if a modified query to the same tool or a different tool is necessary according to your plan. Update your plan if needed.\n"
        "6.  **Final Answer Synthesis:** Once all planned steps are complete and necessary information is gathered, synthesize a single, coherent, and comprehensive answer to the user's *original, complete* query. If some information could not be found, explicitly state that. Do not invent information.\n"
        "7.  **Direct Answer:** For simple greetings or conversation not requiring data, respond directly.\n\n"
        "**Example of a Complex Query Plan:** User asks: 'What are the features of the most expensive house sold in zipcode 98004, and what's its price per square foot?'\n"
        "Your internal plan (Thought Process):\n"
        "   1. I need to find the most expensive house in zipcode 98004. The SQL tool is best for this to get all its details.\n"
        f"   Action: Use '{sql_tool_name}' with the question: 'What are all details for the most expensive house in zipcode 98004?' (This implies a query like SELECT * FROM house_sales WHERE zipcode = 98004 ORDER BY price DESC LIMIT 1).\n"
        "   2. The SQL tool should return all columns for this house, including price and sqft_living.\n"
        "   3. Once I have the price and sqft_living, I need to calculate price per sqft. The Python REPL tool is best for this calculation.\n"
        f"   Action: Use '{python_repl_name}' with Python code like `price = [price_value]; sqft = [sqft_value]; print(price / sqft)`.\n"
        "   4. Synthesize the features from the SQL result and the calculated price per sqft into a final answer.\n"
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_message_content),
            MessagesPlaceholder(variable_name="chat_history", optional=True),
            ("user", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ]
    )
    
    agent = create_tool_calling_agent(_llm_for_agent, _tools_list_for_agent, prompt)
    agent_executor = AgentExecutor(
        agent=agent, 
        tools=_tools_list_for_agent, 
        verbose=True, 
        handle_parsing_errors=True, 
        max_iterations=10, 
    )
    st.success(f"LangChain AgentExecutor created with Enhanced Multi-Step Reasoning Prompt!")
    return agent_executor

agent_executor_instance = None
if llm and tools: 
    agent_executor_instance = get_agent_executor_cached(llm, tools)

# --- ADMIN PANEL (Sidebar) ---
st.sidebar.header("Admin Panel")
# Section 1: Load Data
st.sidebar.subheader("1. Load Data")
if st.sidebar.button("Load Raw Data from CSV"):
    with st.spinner("Loading raw data..."):
        df = load_raw_data()
        if df is not None:
            st.session_state.raw_data = df
            st.sidebar.success(f"Loaded {len(df)} rows into session state.")
        else:
            st.sidebar.error("Failed to load data. Check logs.")

if "raw_data" in st.session_state and isinstance(st.session_state.raw_data, pd.DataFrame):
    st.sidebar.caption(f"{len(st.session_state.raw_data)} rows currently loaded in session.")
    if st.sidebar.checkbox("Show raw data sample in main area", key="show_raw_data_main_agent_v6_final"): # Unique key
        st.subheader("Raw Data Sample (First 10 rows)")
        st.dataframe(st.session_state.raw_data.head(10))
else:
    st.sidebar.info("Raw data not loaded yet. Use the button above.")

# Section 2: Qdrant Collection Management
st.sidebar.subheader("2. Qdrant Collection")
if st.sidebar.button("Initialize/Recreate Qdrant Collection"):
    if not vector_db_instance.client: 
        st.sidebar.error("VectorDB client not available. Cannot manage collection.")
    else:
        with st.spinner(f"Attempting to create/recreate collection: {config.QDRANT_COLLECTION_NAME}..."):
            if vector_db_instance.create_collection(collection_name=config.QDRANT_COLLECTION_NAME, vector_size=config.VECTOR_SIZE): 
                st.sidebar.success(f"Collection '{config.QDRANT_COLLECTION_NAME}' is ready.")
            else:
                st.sidebar.error("Failed to create/recreate collection. Check logs.")

if st.sidebar.button("Show Qdrant Collection Info"):
    if not vector_db_instance.client: 
        st.sidebar.error("VectorDB client not available. Cannot get collection info.")
    else:
        collection_name = config.QDRANT_COLLECTION_NAME
        with st.spinner(f"Getting info for collection: {collection_name}..."):
            info = vector_db_instance.get_collection_info(collection_name)
            if info:
                vector_params_obj = None
                if hasattr(info.config.params, 'vectors') and info.config.params.vectors and not isinstance(info.config.params.vectors, dict):
                    vector_params_obj = info.config.params.vectors
                elif hasattr(info.config.params, 'vectors') and isinstance(info.config.params.vectors, dict) and "" in info.config.params.vectors:
                     vector_params_obj = info.config.params.vectors[""]
                
                display_info = {
                    "name": collection_name, "status": str(info.status) if info.status else "N/A",
                    "points_count": info.points_count if info.points_count is not None else 0,
                    "vectors_count": info.vectors_count if info.vectors_count is not None else 0, 
                    "vector_size": vector_params_obj.size if vector_params_obj and hasattr(vector_params_obj, 'size') else "N/A",
                    "distance_metric": str(vector_params_obj.distance) if vector_params_obj and hasattr(vector_params_obj, 'distance') else "N/A",
                }
                st.sidebar.json(display_info)
            else:
                st.sidebar.warning(f"Could not retrieve info for '{collection_name}'.")

# Section 3: Data Ingestion
st.sidebar.subheader("3. Data Ingestion")
st.sidebar.markdown("**SQL Database:** Population is managed via `docker compose exec app python src/ingestion/populate_sql_db.py`. Ensure this has been run if you intend to use the SQL tool.")

num_rows_to_ingest_qdrant = st.sidebar.number_input(
    "Number of rows to ingest into Qdrant (0 for all):",
    min_value=0, value=50, key="num_rows_ingest_qdrant_v6_final", 
    help="Applies to Qdrant vector ingestion."
)
if st.sidebar.button("Preprocess & Populate Qdrant"):
    if not vector_db_instance.is_ready():
        st.sidebar.error("VectorDB client or embedding model not available for Qdrant ingestion.")
    elif "raw_data" not in st.session_state or st.session_state.raw_data is None:
        st.sidebar.error("Please load raw data first for Qdrant ingestion.")
    else:
        raw_df_to_process = st.session_state.raw_data.copy() 
        if num_rows_to_ingest_qdrant > 0:
            st.sidebar.info(f"Processing first {num_rows_to_ingest_qdrant} rows for Qdrant...")
            raw_df_to_process = raw_df_to_process.head(num_rows_to_ingest_qdrant)
        else:
            st.sidebar.info(f"Processing all {len(raw_df_to_process)} rows for Qdrant...")
        
        with st.spinner("Preprocessing data for Qdrant... This may take a while."):
            processed_df = preprocess_data_for_vectorization(raw_df_to_process)
        
        if processed_df is not None and not processed_df.empty:
            st.sidebar.success(f"Preprocessing for Qdrant complete. {len(processed_df)} rows prepared.")
            if st.sidebar.checkbox("Show sample processed descriptions for Qdrant", key="show_proc_desc_agent_v6_final"): 
                 for i, desc in enumerate(processed_df['description'].head(3)):
                     st.sidebar.caption(f"{i+1}: {desc[:150]}...")
            df_to_upsert = processed_df
            collection_name = config.QDRANT_COLLECTION_NAME
            current_collection_info = vector_db_instance.get_collection_info(collection_name)
            if not current_collection_info:
                 st.sidebar.warning(f"Qdrant Collection '{collection_name}' not found. Attempting to create it.")
                 if not vector_db_instance.create_collection(collection_name, config.VECTOR_SIZE): 
                     st.sidebar.error(f"Failed to create Qdrant collection '{collection_name}'. Cannot ingest data.")
                     st.stop()
            st.sidebar.write(f"Populating Qdrant collection '{collection_name}'...")
            with st.spinner(f"Upserting {len(df_to_upsert)} points to Qdrant. Check Docker logs for batch progress."):
                points_added = vector_db_instance.upsert_points_batch(
                    collection_name=collection_name, points_df=df_to_upsert, batch_size=64
                )
            if points_added > 0: st.sidebar.success(f"Successfully populated Qdrant with {points_added} points!")
            elif points_added == 0 and len(df_to_upsert) > 0: st.sidebar.warning("Qdrant ingestion done, but no new points added. Check logs.")
            else: st.sidebar.error("No points added to Qdrant or error during upsert. Check logs.")
        else:
            st.sidebar.error("Preprocessing for Qdrant failed or returned no data. Check logs.")

# --- CHAT INTERFACE (Main Area) ---
st.header("Chat With Multi-Tool Agent (Enhanced Multi-Step Reasoning)") 

if "agent_chat_history_v6_final" not in st.session_state: 
    st.session_state.agent_chat_history_v6_final = [{"role": "assistant", "content": "Hello! I'm your advanced AI assistant for King County House Sales. I can break down complex questions and use multiple tools (RAG, SQL, Web Search, Code Interpreter). How can I help?"}] 

for message in st.session_state.agent_chat_history_v6_final:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if user_prompt := st.chat_input("Ask a complex question or a simple one..."):
    st.session_state.agent_chat_history_v6_final.append({"role": "user", "content": user_prompt})
    with st.chat_message("user"):
        st.markdown(user_prompt)

    with st.chat_message("assistant"):
        if not agent_executor_instance:
            error_msg = "Agent is not available. Please check initialization messages (VectorDB, LLM, Tools might have failed)."
            st.error(error_msg)
            st.session_state.agent_chat_history_v6_final.append({"role": "assistant", "content": error_msg})
        else:
            with st.spinner("Agent is thinking, planning, and may use its tools... (Check Docker logs for detailed thoughts)"):
                try:
                    lc_chat_history = []
                    for msg in st.session_state.agent_chat_history_v6_final[:-1]: 
                        if msg["role"] == "user": lc_chat_history.append(HumanMessage(content=msg["content"]))
                        elif msg["role"] == "assistant": lc_chat_history.append(AIMessage(content=msg["content"]))
                    
                    response = agent_executor_instance.invoke(
                        {"input": user_prompt, "chat_history": lc_chat_history}
                    )
                    agent_answer = response.get("output", "I tried to process your request but didn't arrive at a clear answer.")
                    
                    st.markdown(agent_answer)
                    st.session_state.agent_chat_history_v6_final.append({"role": "assistant", "content": agent_answer})

                except Exception as e:
                    error_msg = f"An error occurred during agent execution: {str(e)}"
                    st.error(error_msg)
                    st.session_state.agent_chat_history_v6_final.append({"role": "assistant", "content": error_msg})
                    import traceback 
                    print("--- AGENT EXECUTION ERROR TRACEBACK ---")
                    print(traceback.format_exc()) 
                    print("--- END OF TRACEBACK ---")