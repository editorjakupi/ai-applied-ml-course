"""
Skript för att generera kap_10_genomgång_lokal_llm.ipynb
En version av kap_10_genomgång.ipynb som använder lokala LLM-modeller istället för API-nyckel.
"""

import json

def create_notebook():
    """Skapar en Jupyter notebook med lokala LLM-modeller istället för API."""
    
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
            "# Chattbottar - Kapitel 10 (Lokal LLM)\n",
            "\n",
            "I detta kodexempel ska vi få en grundläggande förståelse för hur chattbottar fungerar med **lokala modeller** (ingen API-nyckel behövs!). Let's go!"
        ]
    })
    
    # Cell 1: Installation
    notebook["cells"].append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "source": [
            "# Installera nödvändiga bibliotek för lokala modeller\n",
            "# pip install transformers torch sentence-transformers\n",
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
            "from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline\n",
            "from sentence_transformers import SentenceTransformer\n",
            "from pypdf import PdfReader\n",
            "import torch\n",
            "import warnings\n",
            "warnings.filterwarnings('ignore')"
        ]
    })
    
    # Cell 3: Markdown om lokala modeller
    notebook["cells"].append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "# Använda en chattbot med lokal modell\n",
            "Vi kommer nu demonstrera hur vi kan ha en egen chattbot genom att använda en **lokal modell** som körs på din egen dator. Detta kräver **ingen API-nyckel** och är helt gratis!\n",
            "\n",
            "**Fördelar med lokala modeller:**\n",
            "- Ingen API-nyckel behövs\n",
            "- Helt gratis (ingen kostnad per anrop)\n",
            "- Bättre integritet (data stannar på din dator)\n",
            "- Fungerar offline\n",
            "- Snabbare (ingen nätverksfördröjning)\n",
            "\n",
            "**Nackdelar:**\n",
            "- Kräver mer minne (modeller kan vara stora, 2-20 GB+)\n",
            "- Kräver mer processorkraft (GPU rekommenderas för större modeller)\n",
            "- Sämre prestanda än större molnmodeller\n",
            "\n",
            "I detta exempel använder vi `gpt2` som är en lättviktig modell (~500 MB) som fungerar bättre med svenska text än DialoGPT (som är tränad på engelska konversationer)."
        ]
    })
    
    # Cell 4: Ladda lokal modell och testa
    notebook["cells"].append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "source": [
            "# Ladda ner och initiera lokal modell (görs bara en gång)\n",
            "# Detta kan ta några minuter första gången eftersom modellen laddas ner\n",
            "print(\"Laddar ner lokal modell... Detta kan ta några minuter första gången.\")\n",
            "\n",
            "# Använd GPT-2 istället för DialoGPT (fungerar bättre med svenska)\n",
            "# GPT-2 är mer generell och hanterar svenska text bättre än DialoGPT som är tränad på engelska konversationer\n",
            "model_name = \"gpt2\"  # GPT-2 fungerar bättre med svenska än DialoGPT\n",
            "tokenizer = AutoTokenizer.from_pretrained(model_name)\n",
            "model = AutoModelForCausalLM.from_pretrained(model_name)\n",
            "\n",
            "# Sätt pad_token (GPT-2 behöver detta)\n",
            "if tokenizer.pad_token is None:\n",
            "    tokenizer.pad_token = tokenizer.eos_token\n",
            "\n",
            "# Om du har GPU, använd den för snabbare inferens\n",
            "device = \"cuda\" if torch.cuda.is_available() else \"cpu\"\n",
            "model = model.to(device)\n",
            "print(f\"Modell laddad på: {device}\")\n",
            "\n",
            "# Funktion för att generera svar med lokal modell\n",
            "# GPT-2 fungerar bra både med och utan konversationshistorik\n",
            "def generate_content_local(prompt, chat_history_ids=None, max_length=100, temperature=0.7, max_input_length=512):\n",
            "    \"\"\"\n",
            "    Genererar svar med lokal GPT-2-modell.\n",
            "    \n",
            "    Args:\n",
            "        prompt: Texten att generera svar från\n",
            "        chat_history_ids: Tidigare konversationshistorik (torch tensor), None för första meddelandet\n",
            "        max_length: Maximal längd på genererat svar (i tokens)\n",
            "        temperature: Kreativitet (högre = mer kreativt, lägre = mer deterministiskt)\n",
            "        max_input_length: Maximal längd på input (för att undvika tokeniseringsfel)\n",
            "    \n",
            "    Returns:\n",
            "        response: Genererat svar (str)\n",
            "        new_history_ids: Uppdaterad konversationshistorik för nästa anrop (torch tensor)\n",
            "    \"\"\"\n",
            "    # Sätt pad_token om den inte finns (DialoGPT behöver detta)\n",
            "    if tokenizer.pad_token is None:\n",
            "        tokenizer.pad_token = tokenizer.eos_token\n",
            "    \n",
            "    # Trunkera prompt om den är för lång (för att undvika tokeniseringsfel)\n",
            "    # Tokenisera för att kontrollera längd\n",
            "    prompt_tokens = tokenizer.encode(prompt, add_special_tokens=False)\n",
            "    if len(prompt_tokens) > max_input_length:\n",
            "        # Trunkera från början (behåll slutet som är viktigast)\n",
            "        prompt_tokens = prompt_tokens[-max_input_length:]\n",
            "        prompt = tokenizer.decode(prompt_tokens, skip_special_tokens=True)\n",
            "    \n",
            "    # Tokenisera input med EOS token (använd truncation för säkerhet)\n",
            "    new_user_input_ids = tokenizer.encode(\n",
            "        prompt + tokenizer.eos_token, \n",
            "        return_tensors=\"pt\",\n",
            "        truncation=True,\n",
            "        max_length=max_input_length\n",
            "    ).to(device)\n",
            "    \n",
            "    # Kombinera med historik om den finns\n",
            "    if chat_history_ids is not None:\n",
            "        # Kontrollera total längd\n",
            "        total_length = chat_history_ids.shape[-1] + new_user_input_ids.shape[-1]\n",
            "        max_total_length = model.config.max_position_embeddings if hasattr(model.config, 'max_position_embeddings') else 1024\n",
            "        \n",
            "        if total_length > max_total_length - max_length:\n",
            "            # Trunkera historik om den är för lång\n",
            "            keep_length = max_total_length - max_length - new_user_input_ids.shape[-1] - 50  # Lämna lite marginal\n",
            "            if keep_length > 0:\n",
            "                chat_history_ids = chat_history_ids[:, -keep_length:]\n",
            "            else:\n",
            "                # Om input är för lång, använd bara input\n",
            "                chat_history_ids = None\n",
            "        \n",
            "        if chat_history_ids is not None:\n",
            "            bot_input_ids = torch.cat([chat_history_ids, new_user_input_ids], dim=-1)\n",
            "        else:\n",
            "            bot_input_ids = new_user_input_ids\n",
            "    else:\n",
            "        bot_input_ids = new_user_input_ids\n",
            "    \n",
            "    # Beräkna total max_length (input + nytt svar)\n",
            "    input_length = bot_input_ids.shape[-1]\n",
            "    max_total_length = model.config.max_position_embeddings if hasattr(model.config, 'max_position_embeddings') else 1024\n",
            "    total_max_length = min(input_length + max_length, max_total_length)\n",
            "    \n",
            "    # Generera svar\n",
            "    with torch.no_grad():\n",
            "        try:\n",
            "            chat_history_ids = model.generate(\n",
            "                bot_input_ids,\n",
            "                max_length=total_max_length,\n",
            "                temperature=temperature,\n",
            "                do_sample=True,\n",
            "                pad_token_id=tokenizer.eos_token_id,\n",
            "                no_repeat_ngram_size=3,  # Undvik repetering\n",
            "                num_return_sequences=1\n",
            "            )\n",
            "        except Exception as e:\n",
            "            print(f\"Fel vid generering: {e}\")\n",
            "            # Fallback: försök med kortare input\n",
            "            if bot_input_ids.shape[-1] > 100:\n",
            "                bot_input_ids = bot_input_ids[:, -100:]\n",
            "                input_length = bot_input_ids.shape[-1]\n",
            "                total_max_length = min(input_length + max_length, max_total_length)\n",
            "                chat_history_ids = model.generate(\n",
            "                    bot_input_ids,\n",
            "                    max_length=total_max_length,\n",
            "                    temperature=temperature,\n",
            "                    do_sample=True,\n",
            "                    pad_token_id=tokenizer.eos_token_id,\n",
            "                    num_return_sequences=1\n",
            "                )\n",
            "            else:\n",
            "                raise\n",
            "    \n",
            "    # Extrahera bara det nya svaret (ta bort input och historik)\n",
            "    response_ids = chat_history_ids[:, input_length:]\n",
            "    response = tokenizer.decode(response_ids[0], skip_special_tokens=True)\n",
            "    \n",
            "    return response, chat_history_ids\n",
            "\n",
            "# Testa modellen\n",
            "prompt = \"Hej! Kan du förklara vad AI är?\"\n",
            "response, _ = generate_content_local(prompt, max_length=150)\n",
            "print(response)"
        ]
    })
    
    # Cell 5: Markdown om enkel applikation
    notebook["cells"].append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "# Skapa en mycket enkel applikation\n",
            "Vi kan också skapa en enkel applikation. Med lite kreativitet så inser man att det finns många möjligheter att bygga/utveckla sådant man är intresserad av."
        ]
    })
    
    # Cell 6: Chattloop med lokal modell
    notebook["cells"].append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "source": [
            "print(\"*** Lokal chattbot ***\")\n",
            "print(\"Type <q> to exit chat.\")\n",
            "\n",
            "# Initiera konversationshistorik (DialoGPT behöver detta för att fungera bra)\n",
            "chat_history_ids = None\n",
            "\n",
            "while True:\n",
            "    prompt = input(\"User: \").strip()\n",
            "    \n",
            "    # Avsluta om användaren skriver 'q' eller tom sträng\n",
            "    if not prompt or prompt.lower() == \"q\":\n",
            "        break\n",
            "    \n",
            "    try:\n",
            "        # Använd lokal modell med konversationshistorik (ingen API-nyckel behövs!)\n",
            "        response, chat_history_ids = generate_content_local(\n",
            "            prompt, \n",
            "            chat_history_ids=chat_history_ids, \n",
            "            max_length=100  # Längd på genererat svar\n",
            "        )\n",
            "        print(\"Bot:\", response)\n",
            "    except Exception as e:\n",
            "        print(f\"Fel: {e}\")\n",
            "        print(\"Försöker igen...\")\n",
            "        # Återställ historik vid fel\n",
            "        chat_history_ids = None"
        ]
    })
    
    # Cell 7: Markdown om RAG
    notebook["cells"].append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "# Retrieval Augmented Generation (RAG)"
        ]
    })
    
    # Cell 8: Markdown om PDF
    notebook["cells"].append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## Läsa in PDF-fil"
        ]
    })
    
    # Cell 9: Markdown om PDF-läsning
    notebook["cells"].append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "Vi börjar med att läsa in en PDF-fil som chattbotten kommer ge svar utifrån."
        ]
    })
    
    # Cell 10: Läs PDF
    notebook["cells"].append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "source": [
            "reader = PdfReader(\"chattbot.pdf\")\n",
            "\n",
            "text = \"\"\n",
            "for page in reader.pages:\n",
            "    text += page.extract_text()"
        ]
    })
    
    # Cell 11: Visa längd
    notebook["cells"].append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "source": [
            "print(len(text))"
        ]
    })
    
    # Cell 12: Visa text-exempel
    notebook["cells"].append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "source": [
            "print(text[17:550])"
        ]
    })
    
    # Cell 13: Markdown om chunking
    notebook["cells"].append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## Chunking"
        ]
    })
    
    # Cell 14: Markdown om fixed length chunking
    notebook["cells"].append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "### Fixed length-chunking"
        ]
    })
    
    # Cell 15: Chunking-kod
    notebook["cells"].append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "source": [
            "chunks = []\n",
            "n = 1000\n",
            "overlap = 200\n",
            "for i in range(0, len(text), n - overlap):\n",
            "    chunks.append(text[i:i + n])\n",
            "\n",
            "print(f\"Antal chunks: {len(chunks)}.\")"
        ]
    })
    
    # Cell 16: Visa första chunk
    notebook["cells"].append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "source": [
            "print(chunks[0])"
        ]
    })
    
    # Cell 17: Visa text-exempel 2
    notebook["cells"].append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "source": [
            "print(text[26:584])"
        ]
    })
    
    # Cell 18: Markdown om embeddings
    notebook["cells"].append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## Embeddings\n",
            "Se t.ex. s.321 i kursboken \"Lär dig AI från grunden - Tillämpad maskininlärning med Python\" för vad Embeddings innebär. I korthet, ord representeras med vektorer/siffror.\n",
            "\n",
            "**Med lokal modell:** Vi använder `sentence-transformers` som är ett bibliotek för att skapa embeddings lokalt utan API-nyckel."
        ]
    })
    
    # Cell 19: Skapa embeddings med lokal modell
    notebook["cells"].append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "source": [
            "# Ladda lokal embedding-modell (görs bara en gång)\n",
            "# Detta kan ta några minuter första gången\n",
            "print(\"Laddar ner lokal embedding-modell...\")\n",
            "\n",
            "embedding_model_name = \"sentence-transformers/all-MiniLM-L6-v2\"  # Snabb och lättviktig (~80 MB)\n",
            "embedding_model = SentenceTransformer(embedding_model_name)\n",
            "print(\"Embedding-modell laddad!\")\n",
            "\n",
            "def create_embeddings_local(texts):\n",
            "    \"\"\"\n",
            "    Skapar embeddings för texter med lokal modell.\n",
            "    \n",
            "    Args:\n",
            "        texts: En text eller lista med texter\n",
            "    \n",
            "    Returns:\n",
            "        Embeddings som numpy-array\n",
            "    \"\"\"\n",
            "    # Om det är en enskild text, gör om till lista\n",
            "    if isinstance(texts, str):\n",
            "        texts = [texts]\n",
            "    \n",
            "    # Skapa embeddings\n",
            "    embeddings = embedding_model.encode(texts, convert_to_numpy=True)\n",
            "    \n",
            "    return embeddings"
        ]
    })
    
    # Cell 20: Skapa embeddings för chunks
    notebook["cells"].append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "source": [
            "# Skapa embeddings för alla chunks (detta kan ta några sekunder)\n",
            "print(\"Skapar embeddings för alla chunks...\")\n",
            "chunk_embeddings = create_embeddings_local(chunks)\n",
            "print(f\"Antal embeddings: {len(chunk_embeddings)}\")\n",
            "print(f\"Dimension på varje embedding: {chunk_embeddings[0].shape}\")"
        ]
    })
    
    # Cell 21: Visa exempel på embedding-värden
    notebook["cells"].append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "source": [
            "# Visa första 10 värdena i första embedding-vektorn\n",
            "chunk_embeddings[0][0:10]"
        ]
    })
    
    # Cell 22: Markdown om semantisk sökning
    notebook["cells"].append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## Semantisk sökning"
        ]
    })
    
    # Cell 23: Cosine similarity
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
    
    # Cell 24: Semantic search med lokal embedding
    notebook["cells"].append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "source": [
            "def semantic_search(query, chunks, chunk_embeddings, k=5):\n",
            "    \"\"\"\n",
            "    Söker efter de k mest relevanta chunks baserat på semantisk likhet.\n",
            "    \n",
            "    Args:\n",
            "        query: Sökfrågan\n",
            "        chunks: Lista med text-chunks\n",
            "        chunk_embeddings: Embeddings för chunks (numpy array)\n",
            "        k: Antal toppresultat att returnera\n",
            "    \n",
            "    Returns:\n",
            "        Lista med de k mest relevanta chunks\n",
            "    \"\"\"\n",
            "    # Skapa embedding för frågan med lokal modell\n",
            "    query_embedding = create_embeddings_local(query)[0]  # [0] eftersom vi får en lista tillbaka\n",
            "    \n",
            "    # Beräkna likhet mellan frågan och alla chunks\n",
            "    similarity_scores = []\n",
            "    \n",
            "    for i, chunk_embedding in enumerate(chunk_embeddings):\n",
            "        similarity_score = cosine_similarity(query_embedding, chunk_embedding)\n",
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
    
    # Cell 25: Testa semantic search
    notebook["cells"].append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "source": [
            "fråga = \"Vad kan RAG användas till?\"\n",
            "svar = semantic_search(fråga, chunks=chunks, chunk_embeddings=chunk_embeddings, k=1)\n",
            "print(svar[0])"
        ]
    })
    
    # Cell 26: Markdown om RAG-generering
    notebook["cells"].append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## Generera bra svar med RAG"
        ]
    })
    
    # Cell 27: System prompt
    notebook["cells"].append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "source": [
            "system_prompt = \"\"\"Jag kommer ställa dig en fråga, och jag vill att du svarar\n",
            "baserat bara på kontexten jag skickar med, och ingen annan information.\n",
            "Om det inte finns nog med information i kontexten för att svara på frågan,\n",
            "säg \"Det vet jag inte\". Försök inte att gissa.\n",
            "Formulera dig enkelt och dela upp svaret i fina stycken. \"\"\""
        ]
    })
    
    # Cell 28: Generate user prompt
    notebook["cells"].append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "source": [
            "def generate_user_prompt(query):\n",
            "    \"\"\"\n",
            "    Skapar en user prompt med kontext från semantisk sökning.\n",
            "    \n",
            "    Args:\n",
            "        query: Användarens fråga\n",
            "    \n",
            "    Returns:\n",
            "        En formaterad prompt med frågan och relevant kontext\n",
            "    \"\"\"\n",
            "    # Hitta relevanta chunks\n",
            "    context = \"\\n\".join(semantic_search(query, chunks, chunk_embeddings))\n",
            "    \n",
            "    # Kombinera frågan med kontexten\n",
            "    user_prompt = f\"Frågan är {query}. Här är kontexten: {context}.\"\n",
            "    \n",
            "    return user_prompt"
        ]
    })
    
    # Cell 29: Generate response med lokal modell
    notebook["cells"].append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "source": [
            "def generate_response(system_prompt, user_message, max_length=300):\n",
            "    \"\"\"\n",
            "    Genererar svar med RAG med lokal modell.\n",
            "    \n",
            "    Args:\n",
            "        system_prompt: Systeminstruktioner för modellen\n",
            "        user_message: Användarens fråga\n",
            "        max_length: Maximal längd på genererat svar\n",
            "    \n",
            "    Returns:\n",
            "        Ett Response-objekt med .text attribut\n",
            "    \"\"\"\n",
            "    # Kombinera system prompt och user prompt\n",
            "    full_prompt = f\"{system_prompt}\\n\\n{generate_user_prompt(user_message)}\"\n",
            "    \n",
            "    # Generera svar med lokal modell (ingen historik för RAG, varje fråga är oberoende)\n",
            "    # Använd kortare max_input_length för RAG eftersom prompts kan vara långa\n",
            "    response_text, _ = generate_content_local(\n",
            "        full_prompt, \n",
            "        chat_history_ids=None, \n",
            "        max_length=max_length, \n",
            "        temperature=0.7,\n",
            "        max_input_length=400  # Begränsa input-längd för RAG\n",
            "    )\n",
            "    \n",
            "    # Skapa ett enkelt Response-objekt för kompatibilitet\n",
            "    class Response:\n",
            "        def __init__(self, text):\n",
            "            self.text = text\n",
            "    \n",
            "    return Response(response_text)"
        ]
    })
    
    # Cell 30: Tom cell
    notebook["cells"].append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "source": []
    })
    
    # Cell 31: Testa RAG
    notebook["cells"].append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "source": [
            "print(generate_response(system_prompt, \"Vad är RAG?\").text)"
        ]
    })
    
    # Cell 32: Testa RAG med fråga utan kontext
    notebook["cells"].append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "source": [
            "fråga = \"Vad är meningen med livet?\"\n",
            "svar = generate_response(system_prompt, fråga).text\n",
            "print(svar)"
        ]
    })
    
    # Cell 33: Testa RAG med annan fråga
    notebook["cells"].append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "source": [
            "fråga = \"Vad är prompt engineering?\"\n",
            "svar = generate_response(system_prompt, fråga).text\n",
            "print(svar)"
        ]
    })
    
    # Cell 34: Tom cell
    notebook["cells"].append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "source": []
    })
    
    # Cell 35: Markdown om evaluering
    notebook["cells"].append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "# Evaluering"
        ]
    })
    
    # Cell 36: Validation data
    notebook["cells"].append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "source": [
            "validation_data = [\n",
            "    {\n",
            "        \"question\": \"Vilka delar utgör en RAG-modell?\",\n",
            "        \"ideal_answer\": \"\"\"En RAG-modell innehåller två delar: \n",
            "en retriever som söker efter relevanta stycken i en text, \n",
            "och en generator som genererar svar utifrån den givna kontexten.\"\"\"\n",
            "    }\n",
            "]\n",
            "\n",
            "print(validation_data[0][\"question\"])\n",
            "print()\n",
            "print(validation_data[0][\"ideal_answer\"])"
        ]
    })
    
    # Cell 37: Tom cell
    notebook["cells"].append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "source": []
    })
    
    # Cell 38: Evaluation system prompt
    notebook["cells"].append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "source": [
            "evaluation_system_prompt = \"\"\"Du är ett intelligent utvärderingssystem vars uppgift är att utvärdera en AI-assistents svar. \n",
            "Om svaret är väldigt nära det önskade svaret, sätt poängen 1. Om svaret är felaktigt eller inte bra nog, sätt poängen 0.\n",
            "Om svaret är delvis i linje med det önskade svaret, sätt poängen 0.5. Motivera kort varför du sätter den poäng du gör.\n",
            "\"\"\""
        ]
    })
    
    # Cell 39: Utför evaluering
    notebook["cells"].append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "source": [
            "query = validation_data[0][\"question\"]\n",
            "\n",
            "response = generate_response(system_prompt, query)\n",
            "\n",
            "evaluation_prompt = f\"\"\"Fråga: {query}\n",
            "AI-assistentens svar: {response.text}\n",
            "Önskat svar: {validation_data[0]['ideal_answer']}\"\"\"\n",
            "\n",
            "# Använd samma generate_response-funktion för evaluering\n",
            "evaluation_response = generate_response(evaluation_system_prompt, evaluation_prompt)\n",
            "print(evaluation_response.text)"
        ]
    })
    
    # Cell 40: Tom cell
    notebook["cells"].append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "source": []
    })
    
    # Cell 41: Ytterligare validation data
    notebook["cells"].append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "source": [
            "# Try change to \"ideal_answer\": \"Java\" and notice that if the user defines the wrong \"ideal answer\", then it gets weird. \n",
            "# In a wider context, how do you define ideal answers to questions that have no exact answer and who does this? \n",
            "\n",
            "validation_data_2 = [\n",
            "    {\n",
            "        \"question\": \"Vilket programmeringsspråk är kapitlet skrivet i?\",\n",
            "        \"ideal_answer\": \"Python\"\n",
            "    }\n",
            "]\n",
            "print(validation_data_2[0][\"question\"])\n",
            "print(validation_data_2[0][\"ideal_answer\"])"
        ]
    })
    
    # Cell 42: Testa validation data 2
    notebook["cells"].append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "source": [
            "query = validation_data_2[0][\"question\"]\n",
            "\n",
            "response = generate_response(system_prompt, query)\n",
            "print(response.text)"
        ]
    })
    
    # Cell 43: Evaluera validation data 2
    notebook["cells"].append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "source": [
            "evaluation_prompt = f\"\"\"Fråga: {query}\n",
            "AI-assistentens svar: {response.text}\n",
            "Önskat svar: {validation_data_2[0]['ideal_answer']}\"\"\"\n",
            "\n",
            "evaluation_response = generate_response(evaluation_system_prompt, evaluation_prompt)\n",
            "print(evaluation_response.text)"
        ]
    })
    
    # Cell 44: Tom cell
    notebook["cells"].append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "source": []
    })
    
    # Cell 45: Fördjupning
    notebook["cells"].append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "# Fördjupning\n",
            "Den som är intresserad av att arbeta med chattbottar för t.ex. del 2 av kunskapskontrollen kan fördjupa sig inom LangChain, se t.ex. här: \n",
            "https://academy.langchain.com/collections/foundation\n",
            "\n",
            "Vill man få en snabb överblick, se t.ex. \"Quickstart LangChain Essentials - Python\" här: https://academy.langchain.com/collections/quickstart\n",
            "\n",
            "Vill man få en överblick över LangChain, LangGraph och LangSmith, se här: https://www.youtube.com/watch?v=vJOGC8QJZJQ\n",
            "\n",
            "**För lokala modeller:** Du kan också utforska andra modeller på Hugging Face: https://huggingface.co/models"
        ]
    })
    
    # Cell 46: Tom cell
    notebook["cells"].append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "source": []
    })
    
    return notebook


def main():
    """Huvudfunktion som genererar notebooken."""
    print("Genererar notebook med lokala LLM-modeller...")
    
    notebook = create_notebook()
    
    # Spara notebooken
    output_file = "kap_10_genomgång_lokal_llm.ipynb"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(notebook, f, indent=1, ensure_ascii=False)
    
    print(f"Notebook genererad: {output_file}")
    print("\nViktigt:")
    print("1. Installera nödvändiga bibliotek: pip install transformers torch sentence-transformers pypdf")
    print("2. Första gången du kör notebooken kommer modellerna laddas ner (kan ta några minuter)")
    print("3. Modellerna körs lokalt - ingen API-nyckel behövs!")
    print("4. För bättre prestanda, använd GPU om tillgängligt")


if __name__ == "__main__":
    main()
