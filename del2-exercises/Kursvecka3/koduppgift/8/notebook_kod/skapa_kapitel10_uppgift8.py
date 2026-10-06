"""
Skript för att generera Kapitel10_Uppgift8_RAG.ipynb
En notebook för Koduppgift 8 - Kapitel 10 som implementerar RAG med Gemini API.
"""

import json

def create_notebook():
    """Skapar en Jupyter notebook för Koduppgift 8 - Kapitel 10 (RAG)."""
    
    notebook = {
        "cells": [],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "name": "python",
                "version": "3.8.0"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }
    
    # Cell 0: Titel
    notebook["cells"].append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "# Koduppgift 8 - Kapitel 10: RAG (Retrieval Augmented Generation)\n",
            "\n",
            "I denna uppgift ska du implementera en RAG-baserad chattbot som kan svara på frågor utifrån PDF-dokumentet \"The Murder of Reality.pdf\"."
        ]
    })
    
    # Cell 1: Installation
    notebook["cells"].append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "source": [
            "# Installera nödvändiga bibliotek\n",
            "# pip install google-genai\n",
            "# pip install pypdf"
        ]
    })
    
    # Cell 2: Imports
    notebook["cells"].append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "source": [
            "import numpy as np\n",
            "from google import genai\n",
            "from pypdf import PdfReader"
        ]
    })
    
    # Cell 3: Markdown om API-nyckel
    notebook["cells"].append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "# Konfigurera API-nyckel\n",
            "\n",
            "För att skapa en API-nyckel behöver du gå in på följande hemsida: https://aistudio.google.com/ och sen gå till \"Create API key\". \n",
            "\n",
            "**Viktigt:** Dela aldrig din privata API-nyckel publikt! Använd miljövariabler eller kommentera ut den när du delar koden."
        ]
    })
    
    # Cell 4: API-nyckel och client
    notebook["cells"].append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "source": [
            "# REKOMMENDERAT: Använd miljövariabel för säkerhet\n",
            "import os\n",
            "api_key = os.getenv(\"API_KEY\")\n",
            "\n",
            "# Om miljövariabel inte finns, använd direkt (ta bort när du delar koden!)\n",
            "if api_key is None:\n",
            "    api_key = 'AIzaSyD_9iXP8Mf6qhBQfRd-ef0AferGJHsriY8'\n",
            "\n",
            "client = genai.Client(api_key=api_key)"
        ]
    })
    
    # Cell 5: Markdown om RAG
    notebook["cells"].append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "# Retrieval Augmented Generation (RAG)\n",
            "\n",
            "I denna uppgift ska du bygga en RAG-baserad chattbot som kan svara på frågor utifrån PDF-dokumentet \"The Murder of Reality.pdf\"."
        ]
    })
    
    # Cell 6: Markdown om PDF-läsning
    notebook["cells"].append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## Steg 1: Läsa in PDF-fil"
        ]
    })
    
    # Cell 7: Läs PDF
    notebook["cells"].append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "source": [
            "# Läs in PDF-filen \"The Murder of Reality.pdf\"\n",
            "reader = PdfReader(\"The Murder of Reality.pdf\")\n",
            "\n",
            "text = \"\"\n",
            "for page in reader.pages:\n",
            "    text += page.extract_text()\n",
            "\n",
            "print(f\"Total längd på texten: {len(text)} tecken\")"
        ]
    })
    
    # Cell 8: Visa text-exempel
    notebook["cells"].append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "source": [
            "# Visa ett exempel på texten för att verifiera att den lästs korrekt\n",
            "print(text[:500])"
        ]
    })
    
    # Cell 9: Markdown om chunking
    notebook["cells"].append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## Steg 2: Chunking\n",
            "\n",
            "Dela upp texten i mindre delar (chunks) för att kunna göra semantisk sökning."
        ]
    })
    
    # Cell 10: Chunking-kod
    notebook["cells"].append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "source": [
            "# Fixed length-chunking\n",
            "# Dela upp texten i chunks med överlappning\n",
            "chunks = []\n",
            "n = 1000  # Storlek på varje chunk\n",
            "overlap = 200  # Överlappning mellan chunks\n",
            "\n",
            "for i in range(0, len(text), n - overlap):\n",
            "    chunks.append(text[i:i + n])\n",
            "\n",
            "print(f\"Antal chunks: {len(chunks)}\")\n",
            "print(f\"Första chunk (första 200 tecknen): {chunks[0][:200]}...\")"
        ]
    })
    
    # Cell 11: Markdown om embeddings
    notebook["cells"].append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## Steg 3: Skapa Embeddings\n",
            "\n",
            "Skapa embeddings för alla chunks så att vi kan göra semantisk sökning."
        ]
    })
    
    # Cell 12: Embeddings-funktion
    notebook["cells"].append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "source": [
            "from google.genai import types\n",
            "\n",
            "def create_embeddings(text, model=\"text-embedding-004\", task_type=\"SEMANTIC_SIMILARITY\"):\n",
            "    \"\"\"\n",
            "    Skapar embeddings för text med Gemini API.\n",
            "    \n",
            "    Args:\n",
            "        text: Texten att skapa embeddings för (kan vara str eller lista)\n",
            "        model: Modellnamn för embeddings\n",
            "        task_type: Typ av uppgift (SEMANTIC_SIMILARITY för semantisk sökning)\n",
            "    \n",
            "    Returns:\n",
            "        Embeddings-objekt med .embeddings attribut\n",
            "    \"\"\"\n",
            "    # Om text är en lista med fler än 100 items, dela upp i batchar\n",
            "    # Gemini API tillåter max 100 requests per batch\n",
            "    if isinstance(text, list) and len(text) > 100:\n",
            "        all_embeddings = []\n",
            "        batch_size = 100\n",
            "        \n",
            "        # Processa i batchar om 100\n",
            "        for i in range(0, len(text), batch_size):\n",
            "            batch = text[i:i + batch_size]\n",
            "            print(f\"Processar batch {i//batch_size + 1}/{(len(text)-1)//batch_size + 1} ({len(batch)} chunks)...\")\n",
            "            \n",
            "            batch_result = client.models.embed_content(\n",
            "                model=model,\n",
            "                contents=batch,\n",
            "                config=types.EmbedContentConfig(task_type=task_type)\n",
            "            )\n",
            "            \n",
            "            # Lägg till batch-resultatet till alla embeddings\n",
            "            all_embeddings.extend(batch_result.embeddings)\n",
            "        \n",
            "        # Skapa ett mock-objekt med samma struktur som API-svaret\n",
            "        class EmbeddingsResult:\n",
            "            def __init__(self, embeddings_list):\n",
            "                self.embeddings = embeddings_list\n",
            "        \n",
            "        return EmbeddingsResult(all_embeddings)\n",
            "    else:\n",
            "        # Om det är en sträng eller lista med <= 100 items, kör direkt\n",
            "        return client.models.embed_content(\n",
            "            model=model, \n",
            "            contents=text, \n",
            "            config=types.EmbedContentConfig(task_type=task_type)\n",
            "        )"
        ]
    })
    
    # Cell 13: Skapa embeddings för chunks
    notebook["cells"].append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "source": [
            "# Skapa embeddings för alla chunks\n",
            "# Detta kan ta några sekunder beroende på antal chunks\n",
            "print(\"Skapar embeddings för alla chunks...\")\n",
            "embeddings = create_embeddings(chunks)\n",
            "print(f\"Antal embeddings skapade: {len(embeddings.embeddings)}\")\n",
            "print(f\"Dimension på varje embedding: {len(embeddings.embeddings[0].values)}\")"
        ]
    })
    
    # Cell 14: Visa exempel på embedding-värden
    notebook["cells"].append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "source": [
            "# Visa första 10 värdena i första embedding-vektorn\n",
            "embeddings.embeddings[0].values[0:10]"
        ]
    })
    
    # Cell 15: Markdown om semantisk sökning
    notebook["cells"].append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## Steg 4: Semantisk sökning\n",
            "\n",
            "Implementera semantisk sökning med cosinuslikhet för att hitta de mest relevanta chunks."
        ]
    })
    
    # Cell 16: Cosine similarity
    notebook["cells"].append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "source": [
            "def cosine_similarity(vec1, vec2):\n",
            "    \"\"\"\n",
            "    Beräknar cosinuslikhet mellan två vektorer.\n",
            "    \n",
            "    Args:\n",
            "        vec1, vec2: Numpy arrays eller listor med samma längd\n",
            "    \n",
            "    Returns:\n",
            "        Cosinuslikhet (värde mellan -1 och 1, där 1 = identiska)\n",
            "    \"\"\"\n",
            "    return (np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2)))"
        ]
    })
    
    # Cell 17: Semantic search
    notebook["cells"].append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "source": [
            "def semantic_search(query, chunks, embeddings, k=5):\n",
            "    \"\"\"\n",
            "    Söker efter de k mest relevanta chunks baserat på semantisk likhet.\n",
            "    \n",
            "    Args:\n",
            "        query: Sökfrågan\n",
            "        chunks: Lista med text-chunks\n",
            "        embeddings: Embeddings-objekt från create_embeddings()\n",
            "        k: Antal toppresultat att returnera\n",
            "    \n",
            "    Returns:\n",
            "        Lista med de k mest relevanta chunks\n",
            "    \"\"\"\n",
            "    # Skapa embedding för frågan\n",
            "    query_embedding = create_embeddings(query).embeddings[0].values\n",
            "    \n",
            "    # Beräkna likhet mellan frågan och alla chunks\n",
            "    similarity_scores = []\n",
            "    \n",
            "    for i, chunk_embedding in enumerate(embeddings.embeddings):\n",
            "        similarity_score = cosine_similarity(query_embedding, chunk_embedding.values)\n",
            "        similarity_scores.append((i, similarity_score))\n",
            "\n",
            "    # Sortera efter likhet (högst först)\n",
            "    similarity_scores.sort(key=lambda x: x[1], reverse=True)\n",
            "    \n",
            "    # Hämta de k bästa indices\n",
            "    top_indices = [index for index, _ in similarity_scores[:k]]\n",
            "    \n",
            "    # Returnera motsvarande chunks\n",
            "    return [chunks[index] for index in top_indices]"
        ]
    })
    
    # Cell 18: Testa semantic search
    notebook["cells"].append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "source": [
            "# Testa semantisk sökning\n",
            "test_query = \"Vad handlar dokumentet om?\"\n",
            "relevant_chunks = semantic_search(test_query, chunks, embeddings, k=2)\n",
            "print(f\"Första relevanta chunk (första 300 tecknen):\\n{relevant_chunks[0][:300]}...\")"
        ]
    })
    
    # Cell 19: Markdown om RAG-generering
    notebook["cells"].append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## Steg 5: Generera svar med RAG\n",
            "\n",
            "Kombinera semantisk sökning med textgenerering för att skapa en RAG-baserad chattbot."
        ]
    })
    
    # Cell 20: System prompt
    notebook["cells"].append({
        "cell_type": "code",
        "execution_count": None,
            "metadata": {},
        "source": [
            "# System prompt som instruerar modellen att svara utifrån kontexten\n",
            "system_prompt = \"\"\"Du är en hjälpsam assistent som svarar på frågor baserat på den kontext som tillhandahålls.\n",
            "\n",
            "Instruktioner:\n",
            "- Svara endast baserat på informationen i kontexten som skickas med frågan\n",
            "- Om kontexten innehåller relevant information för att svara på frågan, svara utförligt och tydligt\n",
            "- Om kontexten INTE innehåller tillräcklig information för att svara på frågan, säg endast då \"Det vet jag inte\"\n",
            "- Formulera dig enkelt och tydligt\n",
            "- Dela upp långa svar i stycken\n",
            "\n",
            "Viktigt: Använd kontexten aktivt när den finns tillgänglig!\"\"\""
        ]
    })
    
    # Cell 21: Generate user prompt
    notebook["cells"].append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "source": [
            "def generate_user_prompt(query, k=5, debug=False):\n",
            "    \"\"\"\n",
            "    Skapar en user prompt med kontext från semantisk sökning.\n",
            "    \n",
            "    Args:\n",
            "        query: Användarens fråga\n",
            "        k: Antal relevanta chunks att hämta (default: 5)\n",
            "        debug: Om True, skriv ut debug-information\n",
            "    \n",
            "    Returns:\n",
            "        En formaterad prompt med frågan och relevant kontext\n",
            "    \"\"\"\n",
            "    # Hitta relevanta chunks\n",
            "    relevant_chunks = semantic_search(query, chunks, embeddings, k=k)\n",
            "    \n",
            "    # Kombinera chunks till kontext\n",
            "    context = \"\\n\\n\".join(relevant_chunks)\n",
            "    \n",
            "    if debug:\n",
            "        print(f\"\\n[DEBUG] Antal chunks hittade: {len(relevant_chunks)}\")\n",
            "        print(f\"[DEBUG] Kontext längd: {len(context)} tecken\")\n",
            "        print(f\"[DEBUG] Första 200 tecknen av kontext: {context[:200]}...\")\n",
            "    \n",
            "    # Om kontexten är tom, varna\n",
            "    if not context or len(context.strip()) == 0:\n",
            "        if debug:\n",
            "            print(\"[DEBUG] VARNING: Kontexten är tom!\")\n",
            "        context = \"[Ingen relevant kontext hittades]\"\n",
            "    \n",
            "    # Formatera prompten tydligare\n",
            "    user_prompt = f\"Fråga: {query}\\n\\nRelevant kontext från dokumentet:\\n{context}\\n\\nSvara på frågan baserat på kontexten ovan.\"\n",
            "    \n",
            "    return user_prompt"
        ]
    })
    
    # Cell 22: Generate response med retry och rate limiting
    notebook["cells"].append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "source": [
            "import time\n",
            "from google.genai import types as genai_types\n",
            "\n",
            "def generate_response(system_prompt, user_message, model=\"gemini-2.0-flash\", max_retries=3, delay=2):\n",
            "    \"\"\"\n",
            "    Genererar svar med RAG med retry-logik och rate limiting.\n",
            "    \n",
            "    Args:\n",
            "        system_prompt: Systeminstruktioner för modellen\n",
            "        user_message: Användarens fråga\n",
            "        model: Modellnamn (default: gemini-2.0-flash)\n",
            "        max_retries: Max antal försök vid fel (default: 3)\n",
            "        delay: Startfördröjning i sekunder mellan försök (default: 2)\n",
            "    \n",
            "    Returns:\n",
            "        Response-objekt med .text attribut\n",
            "    \"\"\"\n",
            "    for attempt in range(max_retries):\n",
            "        try:\n",
            "            # Rate limiting: vänta lite mellan requests\n",
            "            if attempt > 0:\n",
            "                wait_time = delay * (2 ** (attempt - 1))  # Exponential backoff\n",
            "                print(f\"Försök {attempt + 1}/{max_retries}. Väntar {wait_time:.1f} sekunder...\")\n",
            "                time.sleep(wait_time)\n",
            "            \n",
            "            response = client.models.generate_content(\n",
            "                model=model,\n",
            "                config=genai_types.GenerateContentConfig(\n",
            "                    system_instruction=system_prompt\n",
            "                ),\n",
            "                contents=generate_user_prompt(user_message)\n",
            "            )\n",
            "            return response\n",
            "            \n",
            "        except Exception as e:\n",
            "            error_message = str(e)\n",
            "            \n",
            "            # Om det är ett 429-fel (rate limit) och vi har fler försök\n",
            "            if \"429\" in error_message or \"RESOURCE_EXHAUSTED\" in error_message:\n",
            "                if attempt < max_retries - 1:\n",
            "                    continue  # Försök igen\n",
            "                else:\n",
            "                    raise Exception(\n",
            "                        f\"API-kvoten är uttömd efter {max_retries} försök. \"\n",
            "                        f\"Vänta några minuter eller aktivera billing på Google Cloud. \"\n",
            "                        f\"Fel: {error_message}\"\n",
            "                    )\n",
            "            else:\n",
            "                # För andra fel, kasta direkt\n",
            "                raise e\n",
            "    \n",
            "    # Om vi kommer hit utan att returnera, något gick fel\n",
            "    raise Exception(f\"Kunde inte generera svar efter {max_retries} försök.\")"
        ]
    })
    
    # Cell 23: Markdown om testning
    notebook["cells"].append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## Steg 6: Testa RAG-systemet\n",
            "\n",
            "Testa din RAG-baserade chattbot med några frågor om PDF-dokumentet."
        ]
    })
    
    # Cell 24: Testa RAG med fråga
    notebook["cells"].append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "source": [
            "# Testa med en fråga om PDF-dokumentet\n",
            "fråga = \"Vad handlar dokumentet om?\"\n",
            "svar = generate_response(system_prompt, fråga)\n",
            "print(f\"Fråga: {fråga}\")\n",
            "print(f\"\\nSvar:\\n{svar.text}\")"
        ]
    })
    
    # Cell 25: Ytterligare testfrågor
    notebook["cells"].append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "source": [
            "# Testa med fler frågor\n",
            "test_frågor = [\n",
            "    \"Vem är huvudpersonen i dokumentet?\",\n",
            "    \"Vad är huvudtemat i dokumentet?\",\n",
            "    \"Vad är meningen med livet?\"  # Denna bör ge \"Det vet jag inte\"\n",
            "]\n",
            "\n",
            "for i, fråga in enumerate(test_frågor):\n",
            "    print(f\"\\n{'='*60}\")\n",
            "    print(f\"Fråga: {fråga}\")\n",
            "    print(f\"\\nSvar:\")\n",
            "    \n",
            "    # Lägg till liten fördröjning mellan requests för att undvika rate limiting\n",
            "    if i > 0:\n",
            "        time.sleep(1)  # Vänta 1 sekund mellan frågor\n",
            "    \n",
            "    svar = generate_response(system_prompt, fråga)\n",
            "    print(svar.text)"
        ]
    })
    
    # Cell 26: Markdown om interaktiv chattbot
    notebook["cells"].append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## Steg 7: Skapa en interaktiv chattbot\n",
            "\n",
            "Skapa en enkel interaktiv chattbot där användaren kan ställa frågor om PDF-dokumentet."
        ]
    })
    
    # Cell 27: Debug - Testa vad som skickas till modellen
    notebook["cells"].append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "source": [
            "# Testa vad som faktiskt skickas till modellen\n",
            "# Kör denna cell för att se kontexten som hittas för en fråga\n",
            "test_fråga = \"Vad handlar dokumentet om?\"\n",
            "print(f\"Testfråga: {test_fråga}\\n\")\n",
            "print(\"=\"*60)\n",
            "user_prompt = generate_user_prompt(test_fråga, k=5, debug=True)\n",
            "print(\"\\n\" + \"=\"*60)\n",
            "print(\"\\nFullständig prompt som skickas till modellen:\")\n",
            "print(\"-\"*60)\n",
            "print(user_prompt[:500] + \"...\" if len(user_prompt) > 500 else user_prompt)"
        ]
    })
    
    # Cell 28: Interaktiv chattbot
    notebook["cells"].append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "source": [
            "print(\"*** RAG-baserad chattbot - The Murder of Reality ***\")\n",
            "print(\"Ställ frågor om PDF-dokumentet. Skriv 'q' för att avsluta.\\n\")\n",
            "\n",
            "while True:\n",
            "    prompt = input(\"Du: \")\n",
            "    if prompt.lower() == \"q\":\n",
            "        print(\"Hej då!\")\n",
            "        break\n",
            "    else:\n",
            "        try:\n",
            "            response = generate_response(system_prompt, prompt)\n",
            "            print(f\"\\nChattbot: {response.text}\\n\")\n",
            "        except Exception as e:\n",
            "            print(f\"Fel: {e}\\n\")"
        ]
    })
    
    # Cell 28: Markdown om evaluering (valfritt)
    notebook["cells"].append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## Steg 8: Evaluering (Valfritt)\n",
            "\n",
            "Om du vill kan du implementera en enkel evaluering av systemet med valideringsdata."
        ]
    })
    
    # Cell 29: Evaluering (valfritt)
    notebook["cells"].append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "source": [
            "# Exempel på valideringsdata (anpassa efter PDF-dokumentet)\n",
            "validation_data = [\n",
            "    {\n",
            "        \"question\": \"Vad handlar dokumentet om?\",\n",
            "        \"ideal_answer\": \"Dokumentet handlar om...\"  # Fyll i baserat på PDF-innehållet\n",
            "    }\n",
            "]\n",
            "\n",
            "# Utför evaluering\n",
            "for item in validation_data:\n",
            "    query = item[\"question\"]\n",
            "    response = generate_response(system_prompt, query)\n",
            "    print(f\"Fråga: {query}\")\n",
            "    print(f\"Svar: {response.text}\")\n",
            "    print(f\"Önskat svar: {item['ideal_answer']}\")\n",
            "    print(\"\\n\" + \"=\"*60 + \"\\n\")"
        ]
    })
    
    # Cell 30: Markdown om reflektion
    notebook["cells"].append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## Reflektion\n",
            "\n",
            "Tänk på följande frågor:\n",
            "1. Fungerar RAG-systemet bra för att svara på frågor om PDF-dokumentet?\n",
            "2. Vilka förbättringar skulle du kunna göra?\n",
            "3. Vad händer om du ställer en fråga som inte finns i dokumentet?\n",
            "4. Hur påverkar chunk-storleken och överlappningen resultatet?"
        ]
    })
    
    return notebook


def main():
    """Huvudfunktion som genererar notebooken."""
    print("Genererar notebook för Koduppgift 8 - Kapitel 10 (RAG)...")
    
    notebook = create_notebook()
    
    # Spara notebooken
    output_file = "Kapitel10_Uppgift8_RAG.ipynb"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(notebook, f, indent=1, ensure_ascii=False)
    
    print(f"Notebook genererad: {output_file}")
    print("\nViktigt:")
    print("1. Se till att PDF-filen 'The Murder of Reality.pdf' finns i samma mapp")
    print("2. Installera nödvändiga bibliotek: pip install google-genai pypdf")
    print("3. Skapa en API-nyckel på https://aistudio.google.com/")
    print("4. Uppdatera API-nyckeln i Cell 4")
    print("5. Kör cellerna i ordning")


if __name__ == "__main__":
    main()
