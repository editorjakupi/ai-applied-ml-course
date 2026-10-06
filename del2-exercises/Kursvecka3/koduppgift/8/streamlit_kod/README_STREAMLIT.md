# RAG Chatbot - Streamlit App

En interaktiv Streamlit-applikation för att ställa frågor om PDF-dokument med RAG (Retrieval Augmented Generation).

## Installation

1. Installera nödvändiga bibliotek:

```bash
pip install -r requirements.txt
```

## Användning

1. Starta Streamlit-appen:

```bash
streamlit run rag_chatbot_app.py
python -m streamlit run rag_chatbot_app.py

```

2. I webbläsaren:
   - Ange din Gemini API-nyckel i sidopanelen
   - Ladda upp en PDF-fil
   - Klicka på "Processa PDF"
   - Ställ frågor om dokumentet i chattfältet

## Funktioner

- Ladda upp vilken PDF-fil som helst
- Automatisk textextraktion och chunking
- Semantisk sökning med embeddings
- Interaktiv chattgränssnitt
- RAG-baserade svar baserat på dokumentinnehåll

## API-nyckel

För att få en Gemini API-nyckel:

1. Gå till https://aistudio.google.com/
2. Klicka på "Create API key"
3. Kopiera nyckeln och ange den i appen

**Viktigt:** Dela aldrig din API-nyckel publikt!
