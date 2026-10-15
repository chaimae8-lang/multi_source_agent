from langchain_ollama import ChatOllama
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

def generate_rag_response(query, base_folder_path, instruction_system):
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    db = FAISS.load_local(base_folder_path, embeddings, allow_dangerous_deserialization=True) # Cela signifie : « Je fais confiance au fichier que je charge et j'accepte le risque potentiel. »
    found_docs = db.similarity_search(query, k=3)
    # Fusionner le texte des 3 meilleurs paragraphes trouvés
    context = "\n\n".join([doc.page_content for doc in found_docs])

    augmented_prompt = f"""Instructions : {instruction_system}
    Context : {context},
    Query: {query}"""
    
    llm = ChatOllama(model="llama3.1", temperature=0.0)     # Initialiser le modèle Gemini
    response = llm.invoke(augmented_prompt)       # Envoyer le prompt augmenté au modèle 
    
    return response.content

"""
# Voilà à quoi ressemble la variable 'found_docs' en mémoire :
found_docs = [ Document(page_content="Le télétravail est autorisé...", metadata={"source": "charte.pdf"}) ]
"""