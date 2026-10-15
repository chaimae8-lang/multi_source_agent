# C'est ici que réside le "cerveau" supérieur de notre assistant, celui qui va lui permettre d'orchestrer et de choisir ses outils.

from langchain_ollama import ChatOllama

# Define the universal function
def initialize_adaptable_agent(extern_tools_list):
    """
    This function takes as input any list of tools created by a client 
    application (HR, weather, finance, etc.) and links it to the LLM.
    """
    # Initialize the Ollama model
    llm = ChatOllama(model="llama3.1", temperature=0.0) 
    # Dynamically bind the list of tools received as an argument
    llm_with_tools = llm.bind_tools(extern_tools_list)

    return llm_with_tools