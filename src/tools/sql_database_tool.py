# src/tools/sql_database_tool.py
import os
import sys
from langchain_community.utilities import SQLDatabase
from langchain.agents import create_sql_agent
from langchain_core.tools import BaseTool
from typing import Type, Optional, Any
from pydantic.v1 import BaseModel, Field 

# DB_URL and db_connection_instance (global for this module)
DB_URL = os.getenv("DATABASE_URL")
db_connection_instance = None 

if DB_URL:
    try:
        db_connection_instance = SQLDatabase.from_uri(
            DB_URL, 
            sample_rows_in_table_info=2, # Show 2 sample rows in table info
            include_tables=['house_sales'] # Explicitly include the table
        )
        display_db_url = DB_URL 
        if "@" in DB_URL and ":" in DB_URL.split("@")[0]: # Mask password for printing
            creds, rest = DB_URL.split("@", 1)
            user, _ = creds.split(":", 1)
            display_db_url = f"{user}:********@{rest}"
        print(f"sql_database_tool.py: SQLDatabase object created for URI: {display_db_url.split('@')[-1]}")
    except Exception as e:
        print(f"sql_database_tool.py: Error creating SQLDatabase object globally: {e}")
        db_connection_instance = None
else:
    print("sql_database_tool.py: WARNING: DATABASE_URL not found in environment. SQLTool may not be fully functional.")


class SQLQueryInput(BaseModel): 
    question: str = Field(description="A natural language question for querying the 'house_sales' SQL database. Suitable for specific values, counts, averages, min/max, or lists based on structured attributes like price, bedrooms, yr_built, zipcode, or a specific property 'id'.")

class SQLQueryTool(BaseTool):
    name: str = "house_sales_sql_database_query"
    description: str = ( 
        "Use this tool to answer specific questions about house sales data by querying a SQL database named 'house_sales'. "
        "This is best for questions requiring precise numerical answers, aggregations (count, average, sum, min, max), "
        "complex filtering on structured attributes (price, bedrooms, sqft, yr_built, zipcode), or finding a specific property if its 'id' is known. "
        "Input should be a natural language question that can be translated to a SQL query against the 'house_sales' table."
    )
    args_schema: Type[BaseModel] = SQLQueryInput
    sql_agent_executor: Optional[Any] = None 

    def __init__(self, llm_for_sql_agent: Any, database_connection: SQLDatabase, **kwargs):
        super().__init__(**kwargs)
        if not llm_for_sql_agent:
            raise ValueError("An LLM instance (llm_for_sql_agent) must be provided for SQLQueryTool.")
        if not database_connection:
            print("Warning: Database connection not provided to SQLQueryTool constructor. It will not function.")
            self.sql_agent_executor = None
            return
        
        print("Initializing SQLQueryTool with an SQL Agent Executor...")
        try:
            # ENHANCED SQL Agent Prefix for more verbose output
            sql_agent_prefix = (
                "You are an agent designed to interact with a PostgreSQL database table named 'house_sales'. "
                "Given an input question, create a syntactically correct PostgreSQL query to run against the 'house_sales' table, then execute it. "
                "Unless the user specifies a specific number of examples, do not limit your query with 'LIMIT' unless it's for a 'TOP N' style question (e.g., 'top 5 most expensive'). "
                "You have tools for listing tables, getting schema, and running queries. "
                "Only use the information returned by these tools to construct your final answer. "
                "You MUST double check your query before executing it. If you get an error, analyze it and rewrite the query if possible. "
                "Do NOT make any DML statements (INSERT, UPDATE, DELETE, DROP, etc.) to the database.\n"
                "**CRITICAL OUTPUT INSTRUCTION: When your SQL query successfully retrieves one or more rows (especially from a SELECT * query or a query for specific details like an ID lookup), your final natural language answer provided as the tool's observation MUST clearly list all the columns retrieved by your query and their corresponding values for each row found. For example: 'Found 1 house with id 12345: price is 500000, bedrooms is 3, date is 2022-05-10, zipcode is 98001, sqft_living is 1500, yr_built is 1990.' or if multiple rows: 'Found 2 houses: Row 1 - id: 123, price: 500k; Row 2 - id: 456, price: 600k.' If no rows are found, state that clearly, for example: 'No house found with ID 99999'.**"
            )

            self.sql_agent_executor = create_sql_agent(
                llm=llm_for_sql_agent,
                db=database_connection, 
                agent_type="tool-calling", 
                verbose=True,
                handle_parsing_errors=True,
                prefix=sql_agent_prefix, # Apply the new detailed prefix
                # top_k=5 # You can adjust how many rows the SQL agent considers from query results for its *own* synthesis
            )
            print("SQL Agent Executor created successfully for SQLQueryTool with verbose output instructions.")
        except Exception as e:
            print(f"Error creating SQL Agent Executor for SQLQueryTool: {e}")
            self.sql_agent_executor = None

    def _run(self, question: str) -> str:
        if not self.sql_agent_executor:
            return "Error: SQL Agent is not initialized within the SQLQueryTool. Cannot execute query."
        
        print(f"\n<<< SQLQueryTool activated with question: {question} >>>")
        try:
            response = self.sql_agent_executor.invoke({"input": question})
            answer = response.get("output", "Could not get a specific answer from the SQL database using the SQL Agent.")
            print(f"SQLQueryTool raw synthesized answer to main agent: {answer[:500]}...") # Log more of the answer
            return answer
        except Exception as e:
            print(f"Error during SQLQueryTool execution: {e}")
            import traceback
            traceback.print_exc() 
            return f"Error querying SQL database: {str(e)}"

    async def _arun(self, question: str) -> str:
        if not self.sql_agent_executor:
            return "Error: SQL Agent is not initialized within the SQLQueryTool."
        print(f"\n<<< SQLQueryTool ASYNC activated with question: {question} >>>")
        try:
            response = await self.sql_agent_executor.ainvoke({"input": question})
            answer = response.get("output", "Could not get a specific answer from the SQL database (async).")
            print(f"SQLQueryTool ASYNC synthesized answer: {answer[:500]}...")
            return answer
        except Exception as e:
            print(f"Error during SQLQueryTool ASYNC execution: {e}")
            return f"Error querying SQL database (async): {str(e)}"

# Standalone test block (remains the same, good for testing this file in isolation)
if __name__ == '__main__':
    print("Attempting to test SQLQueryTool directly...")
    if not db_connection_instance:
        print("SQLDatabase connection (db_connection_instance) not available globally, cannot run SQLQueryTool test.")
    else:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_src_dir = os.path.abspath(os.path.join(current_dir, '..'))
        if project_src_dir not in sys.path:
            sys.path.insert(0, project_src_dir)
        from llm_services.llm_client import get_llm # Assuming get_llm can be called
        
        test_llm = get_llm()

        if test_llm:
            print("LLM and DB connection seem okay for testing SQLQueryTool.")
            # Pass the globally initialized db_connection_instance
            sql_tool = SQLQueryTool(llm_for_sql_agent=test_llm, database_connection=db_connection_instance)
            
            if sql_tool.sql_agent_executor:
                print("\n--- SQLQueryTool Standalone Test ---")
                # Ensure your 'house_sales' table exists and has data from populate_sql_db.py
                test_questions = [
                    "How many houses are in the house_sales table?",
                    "What are all details for house with id 7129300520 from the house_sales table?" # Test for verbose output
                ]
                for q_idx, question in enumerate(test_questions):
                    print(f"\nTesting with Q{q_idx+1}: {question}")
                    answer = sql_tool._run(question) # Use _run for testing
                    print(f"Answer Q{q_idx+1}: {answer}")
            else:
                print("SQLQueryTool could not initialize its SQL agent executor for the test.")
        else:
            print("Could not get LLM for SQLQueryTool test.")