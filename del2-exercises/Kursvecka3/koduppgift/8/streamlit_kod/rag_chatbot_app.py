import streamlit as st
import numpy as np
from google import genai
from google.genai import types as genai_types
from pypdf import PdfReader
import time
import os

# Konfigurera sidan
st.set_page_config(
    page_title="RAG Chatbot - PDF Q&A",
    layout="wide"
)

# API-nyckel konfiguration
def get_api_key():
    """Hämtar API-nyckel från miljövariabel eller session state"""
    api_key = os.getenv("API_KEY")
    if api_key is None:
        # Om ingen miljövariabel finns, använd den från session state
        if "api_key" not in st.session_state:
            return None
        return st.session_state.api_key
    return api_key

# Initialisera session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "chunks" not in st.session_state:
    st.session_state.chunks = None
if "embeddings" not in st.session_state:
    st.session_state.embeddings = None
if "document_loaded" not in st.session_state:
    st.session_state.document_loaded = False
if "client" not in st.session_state:
    st.session_state.client = None

# Helper-funktioner
def cosine_similarity(vec1, vec2):
    """Beräknar cosinuslikhet mellan två vektorer"""
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

def create_embeddings(text, client, model="text-embedding-004", task_type="SEMANTIC_SIMILARITY"):
    """
    Skapar embeddings för text med Gemini API.
    Hanterar batchar om fler än 100 items.
    """
    if isinstance(text, list) and len(text) > 100:
        all_embeddings = []
        batch_size = 100
        
        progress_bar = st.progress(0)
        total_batches = (len(text) - 1) // batch_size + 1
        
        for i in range(0, len(text), batch_size):
            batch = text[i:i + batch_size]
            batch_num = i // batch_size + 1
            progress_bar.progress(batch_num / total_batches)
            
            batch_result = client.models.embed_content(
                model=model,
                contents=batch,
                config=genai_types.EmbedContentConfig(task_type=task_type)
            )
            all_embeddings.extend(batch_result.embeddings)
        
        progress_bar.empty()
        
        class EmbeddingsResult:
            def __init__(self, embeddings_list):
                self.embeddings = embeddings_list
        
        return EmbeddingsResult(all_embeddings)
    else:
        return client.models.embed_content(
            model=model,
            contents=text,
            config=genai_types.EmbedContentConfig(task_type=task_type)
        )

def semantic_search(query, chunks, embeddings, client, k=5):
    """Söker efter de k mest relevanta chunks baserat på semantisk likhet"""
    query_embedding = create_embeddings(query, client).embeddings[0].values
    
    similarity_scores = []
    for i, chunk_embedding in enumerate(embeddings.embeddings):
        similarity_score = cosine_similarity(query_embedding, chunk_embedding.values)
        similarity_scores.append((i, similarity_score))
    
    similarity_scores.sort(key=lambda x: x[1], reverse=True)
    top_indices = [index for index, _ in similarity_scores[:k]]
    
    return [chunks[index] for index in top_indices]

def generate_user_prompt(query, chunks, embeddings, client, k=5):
    """Skapar en user prompt med kontext från semantisk sökning"""
    relevant_chunks = semantic_search(query, chunks, embeddings, client, k=k)
    context = "\n\n".join(relevant_chunks)
    
    if not context or len(context.strip()) == 0:
        context = "[Ingen relevant kontext hittades]"
    
    user_prompt = f"Fråga: {query}\n\nRelevant kontext från dokumentet:\n{context}\n\nSvara på frågan baserat på kontexten ovan."
    return user_prompt

def generate_response(system_prompt, user_message, client, chunks, embeddings, model="gemini-2.0-flash", max_retries=3, delay=2):
    """Genererar svar med RAG med retry-logik"""
    for attempt in range(max_retries):
        try:
            if attempt > 0:
                wait_time = delay * (2 ** (attempt - 1))
                time.sleep(wait_time)
            
            user_prompt = generate_user_prompt(user_message, chunks, embeddings, client)
            
            response = client.models.generate_content(
                model=model,
                config=genai_types.GenerateContentConfig(
                    system_instruction=system_prompt
                ),
                contents=user_prompt
            )
            return response
            
        except Exception as e:
            error_message = str(e)
            if "429" in error_message or "RESOURCE_EXHAUSTED" in error_message:
                if attempt < max_retries - 1:
                    continue
                else:
                    raise Exception(
                        f"API-kvoten är uttömd efter {max_retries} försök. "
                        f"Vänta några minuter eller aktivera billing på Google Cloud."
                    )
            else:
                raise e
    
    raise Exception(f"Kunde inte generera svar efter {max_retries} försök.")

# System prompt
SYSTEM_PROMPT = """Du är en hjälpsam assistent som svarar på frågor baserat på den kontext som tillhandahålls.

Instruktioner:
- Svara endast baserat på informationen i kontexten som skickas med frågan
- Om kontexten innehåller relevant information för att svara på frågan, svara utförligt och tydligt
- Om kontexten INTE innehåller tillräcklig information för att svara på frågan, säg endast då "Det vet jag inte"
- Formulera dig enkelt och tydligt
- Dela upp långa svar i stycken

Viktigt: Använd kontexten aktivt när den finns tillgänglig!"""

# Huvudapplikation
def main():
    st.title("RAG Chatbot - PDF Q&A")
    st.markdown("Ladda upp en PDF-fil och ställ frågor om innehållet.")
    
    # Sidebar för konfiguration
    with st.sidebar:
        st.header("Konfiguration")
        
        # API-nyckel input
        api_key_input = st.text_input(
            "Gemini API-nyckel",
            type="password",
            value=st.session_state.get("api_key", ""),
            help="Ange din Gemini API-nyckel. Du kan hämta en på https://aistudio.google.com/"
        )
        
        if api_key_input:
            st.session_state.api_key = api_key_input
            try:
                st.session_state.client = genai.Client(api_key=api_key_input)
                st.success("API-nyckel konfigurerad")
            except Exception as e:
                st.error(f"Fel vid konfigurering av API-nyckel: {e}")
        
        st.divider()
        
        # PDF-uppladdning
        st.header("Ladda upp PDF")
        uploaded_file = st.file_uploader(
            "Välj en PDF-fil",
            type="pdf",
            help="Ladda upp en PDF-fil för att kunna ställa frågor om den"
        )
        
        if uploaded_file is not None:
            if st.button("Processa PDF", type="primary"):
                process_pdf(uploaded_file)
        
        # Visa status
        if st.session_state.document_loaded:
            st.success(f"Dokument laddat: {len(st.session_state.chunks)} chunks")
        
        st.divider()
        
        # Rensa chatt
        if st.button("Rensa chatt"):
            st.session_state.messages = []
            st.rerun()
    
    # Huvudområde
    if not st.session_state.client:
        st.warning("Vänligen ange din Gemini API-nyckel i sidopanelen för att börja.")
        return
    
    if not st.session_state.document_loaded:
        st.info("Ladda upp en PDF-fil i sidopanelen för att börja ställa frågor.")
        return
    
    # Chattgränssnitt
    st.header("Chatt")
    
    # Visa chatt-historik
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Input för ny fråga
    if prompt := st.chat_input("Ställ en fråga om dokumentet..."):
        # Lägg till användarens meddelande
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generera svar
        with st.chat_message("assistant"):
            with st.spinner("Söker i dokumentet och genererar svar..."):
                try:
                    response = generate_response(
                        SYSTEM_PROMPT,
                        prompt,
                        st.session_state.client,
                        st.session_state.chunks,
                        st.session_state.embeddings
                    )
                    answer = response.text
                    st.markdown(answer)
                    st.session_state.messages.append({"role": "assistant", "content": answer})
                except Exception as e:
                    error_msg = f"Fel: {str(e)}"
                    st.error(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})

def process_pdf(uploaded_file):
    """Processar en uppladdad PDF-fil"""
    try:
        with st.spinner("Läser PDF-fil..."):
            # Spara temporärt
            with open("temp_pdf.pdf", "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            # Läs PDF
            reader = PdfReader("temp_pdf.pdf")
            text = ""
            for page in reader.pages:
                text += page.extract_text()
            
            if not text or len(text.strip()) == 0:
                st.error("Kunde inte läsa text från PDF-filen. Kontrollera att filen innehåller text.")
                return
            
            st.success(f"PDF läst: {len(text)} tecken")
        
        with st.spinner("Dela upp text i chunks..."):
            # Chunking
            chunks = []
            n = 1000
            overlap = 200
            
            for i in range(0, len(text), n - overlap):
                chunks.append(text[i:i + n])
            
            st.session_state.chunks = chunks
            st.success(f"Text delad i {len(chunks)} chunks")
        
        with st.spinner("Skapar embeddings (detta kan ta en stund)..."):
            # Skapa embeddings
            embeddings = create_embeddings(
                chunks,
                st.session_state.client
            )
            st.session_state.embeddings = embeddings
            st.success(f"Embeddings skapade: {len(embeddings.embeddings)}")
        
        st.session_state.document_loaded = True
        st.session_state.messages = []  # Rensa chatt-historik
        st.success("PDF processad och redo för frågor!")
        st.rerun()
        
    except Exception as e:
        st.error(f"Fel vid processering av PDF: {str(e)}")
    finally:
        # Ta bort temporär fil
        if os.path.exists("temp_pdf.pdf"):
            try:
                os.remove("temp_pdf.pdf")
            except:
                pass

if __name__ == "__main__":
    main()
