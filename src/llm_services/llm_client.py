# src/llm_services/llm_client.py
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
import nest_asyncio 
nest_asyncio.apply()
dotenv_path = os.path.join(os.path.dirname(__file__), '..', '..', '.env') 
load_dotenv(dotenv_path=dotenv_path)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

def get_llm(provider_preference="gemini", model_name=None, temperature=0.1, **kwargs):
    """
    Dynamically initializes and returns an LLM client.
    Prefers Gemini if its API key is available and preferred.
    Falls back to OpenAI if Gemini is not available/preferred and OpenAI key is available.

    Args:
        provider_preference (str): "gemini" or "openai".
        model_name (str, optional): Specific model name.
                                    For Gemini: e.g., "gemini-1.5-flash-latest", "gemini-pro"
                                    For OpenAI: e.g., "gpt-4o", "gpt-3.5-turbo"
        temperature (float): The temperature for the LLM.
        **kwargs: Additional keyword arguments for the LLM constructor.

    Returns:
        An instance of ChatGoogleGenerativeAI or ChatOpenAI, or None if no key is found.
    """
    llm = None
    selected_provider = None

    if provider_preference.lower() == "gemini" and GEMINI_API_KEY:
        try:
            llm = ChatGoogleGenerativeAI(
                model=model_name if model_name else "gemini-1.5-flash-latest",
                google_api_key=GEMINI_API_KEY,
                temperature=temperature,
                convert_system_message_to_human=True,
                **kwargs
            )
            selected_provider = "Gemini"
        except Exception as e:
            print(f"Failed to initialize Gemini LLM: {e}. Trying OpenAI if available.")
    if not llm and OPENAI_API_KEY:
        try:
            llm = ChatOpenAI(
                model_name=model_name if model_name else "gpt-4o",
                openai_api_key=OPENAI_API_KEY,
                temperature=temperature,
                **kwargs
            )
            selected_provider = "OpenAI"
        except Exception as e:
            print(f"Failed to initialize OpenAI LLM: {e}")

    if llm and selected_provider:
        print(f"Using {selected_provider} LLM.")
    else:
        print("ERROR: No LLM API key found for preferred/fallback provider, or initialization failed. Please set GEMINI_API_KEY or OPENAI_API_KEY in your .env file.")
    return llm

if __name__ == "__main__":
    print("Attempting to load LLM (Gemini preferred by default):")
    llm_instance = get_llm()
    if llm_instance:
        print(f"Successfully initialized: {type(llm_instance)}")
    else:
        print("Could not initialize any LLM.")

    print("\nAttempting to load LLM (OpenAI preferred):")
    llm_instance_openai = get_llm(provider_preference="openai")
    if llm_instance_openai:
        print(f"Successfully initialized: {type(llm_instance_openai)}")
    else:
        print("Could not initialize any LLM.")
