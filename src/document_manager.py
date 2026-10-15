from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

def decouper_text(texte, source): # source est pour les métadonnées
    document = Document(page_content=texte, metadata={"source": source})     # Créer un objet Document initial avec le texte et la source en métadonnée
    decoupeur = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks_list = decoupeur.split_documents([document])
    return chunks_list

