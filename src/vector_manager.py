# Pour l'installations de FAISS : pip install langchain-community faiss-cpu sentence-transformers
# Pour l'installation de huggingface : pip install langchain-huggingface

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

def create_vectoriel_base(chunks_list, sauvegarde_path):
    embeddings = HuggingFaceEmbeddings(model_name='all-MiniLM-L6-v2')
    db = FAISS.from_documents(chunks_list, embeddings)
    db.save_local(sauvegarde_path)
    return db

