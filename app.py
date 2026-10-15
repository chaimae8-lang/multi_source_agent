import yaml
import streamlit as st
from pypdf import PdfReader

from src.document_manager import decouper_text
from src.vector_manager import create_vectoriel_base
from src.rag_engine import generate_rag_response
from src.tools_agent import initialize_adaptable_agent

# ---------------------------------------------------------------------
# STEP 1: CONFIGURATION & YAML CONTROL PANEL
# ---------------------------------------------------------------------
# Read the external configuration file
with open("config.yaml", "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

# Extract dynamic settings from YAML
app_title = config["app_settings"]["title"]
developer_instruction = config["app_settings"].get("system_instruction", "")

# Display the dynamic title on the main page
st.title(app_title)

# ---------------------------------------------------------------------
# STEP 2: GRAPHICAL USER INTERFACE (SIDEBAR ADAPTABILITY)
# ---------------------------------------------------------------------
with st.sidebar:
    st.header("Admin & Document Control")
    
    # 2a. Document Loader
    load_file = st.file_uploader("Upload a document (TXT, PDF)", type=["txt", "pdf"])
    
    st.write("---")
    st.header("Developer Settings")
    
    # 2b. Dynamic Text Area for the System Instruction
    new_instruction = st.text_area(
        "Define AI Behavior Instruction:", 
        value=developer_instruction,
        placeholder="Type the behavior prompt here (e.g., You are a medical assistant...)"
    )
    
    # 2c. Save Button to persist configuration into config.yaml via yaml.dump
    if st.button("Save Configuration"):
        config["app_settings"]["system_instruction"] = new_instruction
        with open("config.yaml", "w", encoding="utf-8") as f:
            yaml.dump(config, f, allow_unicode=True)
        st.success("Configuration successfully saved!")
        st.rerun()

# ---------------------------------------------------------------------
# STEP 3: SESSION STATE INITIALIZATION (MEMORY)
# ---------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

if "vector_db" not in st.session_state:
    st.session_state.vector_db = None

# ---------------------------------------------------------------------
# STEP 4: DATA INGESTION PIPELINE (RAG INITIALIZATION)
# ---------------------------------------------------------------------

if load_file is not None and st.session_state.vector_db is None:
    file_name = load_file.name
    
    # CONDITION : Si le fichier se termine par .pdf
    if file_name.endswith(".pdf"):
        # a. Utiliser PdfReader pour boucler sur les pages et extraire le texte
        reader = PdfReader(load_file)
        raw_text = ""
        for page in reader.pages:
            raw_text += page.extract_text()
            
    # CONDITION ALTERNATIVE : Si c'est un fichier texte classique
    else:
        raw_text = load_file.read().decode("utf-8")  
    
    with st.spinner("Analyzing document and creating embeddings... Please wait."):
        # Step 1: Chunking the text
        chunking = decouper_text(raw_text, file_name)
        # Step 2: Creating the local vector store
        sauvegarde_path = "vector_store/"
        st.session_state.vector_db = create_vectoriel_base(chunking, sauvegarde_path)
        
    st.success("Document successfully indexed! Your assistant is ready.")

# ---------------------------------------------------------------------
# STEP 5: DISPLAY CHAT HISTORY
# ---------------------------------------------------------------------
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# ---------------------------------------------------------------------
# STEP 6: USER INPUT & AI GENERATION LOOP
# ---------------------------------------------------------------------
user_question = st.chat_input("Ask a question about your document...")

if user_question:
    # 6a. Display and save user question
    with st.chat_message("user"):
        st.write(user_question)
    st.session_state.messages.append({"role": "user", "content": user_question})

    # 6b. Generate AI response using RAG Engine
    with st.spinner("The assistant is thinking..."):
        ai_response = generate_rag_response(user_question, "vector_store/", developer_instruction)
        
    # 6c. Display and save assistant response
    with st.chat_message("assistant"):
        st.write(ai_response)
    st.session_state.messages.append({"role": "assistant", "content": ai_response})