"""
Python-skript som genererar en Jupyter notebook för Koduppgift 9, Kapitel 8.
Uppgiften handlar om att använda förtränade modeller för att prediktera egna bilder
och bygga en Streamlit-applikation.
"""

import json

def create_notebook():
    """Skapar en komplett notebook för Koduppgift 9, Kapitel 8."""
    
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
    
    # Cell 1: Titel och introduktion
    notebook["cells"].append({
        "cell_type": "markdown",
        "id": "title",
        "metadata": {},
        "source": [
            "# Koduppgift 9 - Kapitel 8: Förtränade modeller och Streamlit-applikation\n",
            "\n",
            "I denna uppgift utgår vi ifrån kodexempel 2 i detta kapitel.\n",
            "\n",
            "**Uppgiften består av två delar:**\n",
            "\n",
            "a) Ta egna bilder som du predikterar med en förtränad modell.\n",
            "\n",
            "b) Bygg en applikation (med exempelvis Streamlit) som använder en förtränad modell för att prediktera bilder. Hur du designar applikationen och vilken funktionalitet du inkluderar väljer du själv.\n",
            "\n",
            "**Förtränade modeller:**\n",
            "- ResNet50: CNN-arkitektur med cirka 26 miljoner parametrar, tränad på ImageNet (1000 klasser)\n",
            "- Dokumentation: https://keras.io/api/applications/\n",
            "- ImageNet klasser: https://deeplearning.cms.waikato.ac.nz/user-guide/class-maps/IMAGENET/"
        ]
    })
    
    # Cell 2: Importera bibliotek
    notebook["cells"].append({
        "cell_type": "markdown",
        "id": "imports_header",
        "metadata": {},
        "source": [
            "## Importera bibliotek"
        ]
    })
    
    notebook["cells"].append({
        "cell_type": "code",
        "execution_count": None,
        "id": "imports",
        "metadata": {},
        "outputs": [],
        "source": [
            "import numpy as np\n",
            "import matplotlib.pyplot as plt\n",
            "from PIL import Image\n",
            "import tensorflow as tf\n",
            "from tensorflow import keras\n",
            "from tensorflow.keras.applications.resnet50 import ResNet50\n",
            "from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions\n",
            "from tensorflow.keras.preprocessing import image\n",
            "import os\n",
            "import streamlit as st\n",
            "\n",
            "print(f\"TensorFlow version: {tf.__version__}\")\n",
            "print(f\"Keras version: {keras.__version__}\")"
        ]
    })
    
    # Cell 3: Del A - Ladda förtränad modell
    notebook["cells"].append({
        "cell_type": "markdown",
        "id": "part_a_header",
        "metadata": {},
        "source": [
            "# Del A: Prediktera egna bilder med förtränad modell"
        ]
    })
    
    notebook["cells"].append({
        "cell_type": "markdown",
        "id": "load_model_a",
        "metadata": {},
        "source": [
            "## Ladda förtränad ResNet50 modell"
        ]
    })
    
    notebook["cells"].append({
        "cell_type": "code",
        "execution_count": None,
        "id": "load_pretrained_model",
        "metadata": {},
        "outputs": [],
        "source": [
            "# Ladda förtränad ResNet50 modell\n",
            "# Modellen är tränad på ImageNet med 1000 olika klasser\n",
            "model = ResNet50(weights='imagenet')\n",
            "\n",
            "print(\"ResNet50 modell laddad!\")\n",
            "print(f\"Modellen kan klassificera 1000 olika klasser från ImageNet.\")\n",
            "print(f\"Input-storlek: 224x224 pixlar med 3 kanaler (RGB)\")"
        ]
    })
    
    # Cell 4: Funktion för att prediktera bilder
    notebook["cells"].append({
        "cell_type": "markdown",
        "id": "predict_function",
        "metadata": {},
        "source": [
            "## Funktion för att prediktera bilder"
        ]
    })
    
    notebook["cells"].append({
        "cell_type": "code",
        "execution_count": None,
        "id": "define_predict_function",
        "metadata": {},
        "outputs": [],
        "source": [
            "def predict_image(image_path, model, top_n=5):\n",
            "    \"\"\"\n",
            "    Prediktera en bild med förtränad ResNet50 modell.\n",
            "    \n",
            "    Args:\n",
            "        image_path: Sökväg till bildfilen\n",
            "        model: Förtränad modell (ResNet50)\n",
            "        top_n: Antal topprediktioner att visa (standard: 5)\n",
            "    \n",
            "    Returns:\n",
            "        predictions: Lista med topprediktioner\n",
            "        preprocessed_img: Förbearbetad bild för visualisering\n",
            "    \"\"\"\n",
            "    # Ladda och förbehandla bilden\n",
            "    img = image.load_img(image_path, target_size=(224, 224))\n",
            "    \n",
            "    # Konvertera till array\n",
            "    img_array = image.img_to_array(img)\n",
            "    \n",
            "    # Expandera dimensioner för batch (1, 224, 224, 3)\n",
            "    img_batch = np.expand_dims(img_array, axis=0)\n",
            "    \n",
            "    # Förbehandla bilden för ResNet50\n",
            "    img_preprocessed = preprocess_input(img_batch)\n",
            "    \n",
            "    # Prediktera\n",
            "    predictions = model.predict(img_preprocessed, verbose=0)\n",
            "    \n",
            "    # Dekodera prediktioner till läsbara klasser\n",
            "    decoded_predictions = decode_predictions(predictions, top=top_n)[0]\n",
            "    \n",
            "    return decoded_predictions, img\n",
            "\n",
            "print(\"Predict-funktion definierad!\")"
        ]
    })
    
    # Cell 5: Prediktera egna bilder
    notebook["cells"].append({
        "cell_type": "markdown",
        "id": "predict_own_images",
        "metadata": {},
        "source": [
            "## Prediktera egna bilder\n",
            "\n",
            "Lägg dina egna bilder i samma mapp som denna notebook, eller ange sökvägen till bilderna.\n",
            "\n",
            "**Tips:**\n",
            "- Ta bilder med din mobil eller kamera\n",
            "- Bilderna kan vara i format JPG, PNG, JPEG\n",
            "- För bästa resultat: välj bilder på objekt som finns i ImageNet (1000 klasser)\n",
            "- Exempel: djur, fordon, mat, elektronik, möbler, etc."
        ]
    })
    
    notebook["cells"].append({
        "cell_type": "code",
        "execution_count": None,
        "id": "predict_single_image",
        "metadata": {},
        "outputs": [],
        "source": [
            "# Exempel: Prediktera en bild\n",
            "# Ändra sökvägen till din egen bild\n",
            "image_path = 'min_bild.jpg'  # Ändra till din bilds sökväg\n",
            "\n",
            "# Kontrollera om filen finns\n",
            "if os.path.exists(image_path):\n",
            "    # Prediktera\n",
            "    predictions, img = predict_image(image_path, model, top_n=5)\n",
            "    \n",
            "    # Visa bilden\n",
            "    plt.figure(figsize=(10, 5))\n",
            "    \n",
            "    plt.subplot(1, 2, 1)\n",
            "    plt.imshow(img)\n",
            "    plt.axis('off')\n",
            "    plt.title('Ursprungsbild')\n",
            "    \n",
            "    # Visa topprediktioner\n",
            "    plt.subplot(1, 2, 2)\n",
            "    plt.axis('off')\n",
            "    prediction_text = '\\n'.join([\n",
            "        f\"{i+1}. {pred[1]}: {pred[2]:.4f}\" \n",
            "        for i, pred in enumerate(predictions)\n",
            "    ])\n",
            "    plt.text(0.1, 0.5, f\"Topp 5 prediktioner:\\n\\n{prediction_text}\", \n",
            "             fontsize=12, verticalalignment='center',\n",
            "             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))\n",
            "    \n",
            "    plt.tight_layout()\n",
            "    plt.show()\n",
            "    \n",
            "    # Skriv ut prediktioner\n",
            "    print(f\"\\nPrediktioner för {image_path}:\")\n",
            "    print(\"-\" * 60)\n",
            "    for i, (imagenet_id, label, score) in enumerate(predictions, 1):\n",
            "        print(f\"{i}. {label}: {score:.4f} ({score*100:.2f}%)\")\n",
            "else:\n",
            "    print(f\"Bildfilen '{image_path}' hittades inte.\")\n",
            "    print(\"Lägg din bild i samma mapp som denna notebook eller ange rätt sökväg.\")"
        ]
    })
    
    # Cell 6: Prediktera flera bilder
    notebook["cells"].append({
        "cell_type": "code",
        "execution_count": None,
        "id": "predict_multiple_images",
        "metadata": {},
        "outputs": [],
        "source": [
            "# Prediktera flera bilder samtidigt\n",
            "image_paths = ['bild1.jpg', 'bild2.jpg', 'bild3.jpg']  # Ändra till dina bilder\n",
            "\n",
            "# Filtrera bort bilder som inte finns\n",
            "existing_images = [path for path in image_paths if os.path.exists(path)]\n",
            "\n",
            "if existing_images:\n",
            "    # Ladda och förbehandla alla bilder\n",
            "    imgs = []\n",
            "    for path in existing_images:\n",
            "        img = image.load_img(path, target_size=(224, 224))\n",
            "        img_array = image.img_to_array(img)\n",
            "        img_batch = np.expand_dims(img_array, axis=0)\n",
            "        img_preprocessed = preprocess_input(img_batch)\n",
            "        imgs.append(img_preprocessed)\n",
            "    \n",
            "    # Stacka alla bilder till en batch\n",
            "    batch = np.vstack(imgs)\n",
            "    \n",
            "    # Prediktera batch\n",
            "    predictions = model.predict(batch, verbose=2)\n",
            "    \n",
            "    # Visa resultat för varje bild\n",
            "    num_images = len(existing_images)\n",
            "    fig, axes = plt.subplots(num_images, 2, figsize=(12, 4*num_images))\n",
            "    \n",
            "    if num_images == 1:\n",
            "        axes = axes.reshape(1, -1)\n",
            "    \n",
            "    for i, path in enumerate(existing_images):\n",
            "        # Ladda originalbild för visning\n",
            "        img = image.load_img(path)\n",
            "        \n",
            "        # Dekodera prediktioner\n",
            "        decoded = decode_predictions(predictions[i:i+1], top=3)[0]\n",
            "        \n",
            "        # Visa bild\n",
            "        axes[i, 0].imshow(img)\n",
            "        axes[i, 0].axis('off')\n",
            "        axes[i, 0].set_title(f'Bild {i+1}: {os.path.basename(path)}')\n",
            "        \n",
            "        # Visa topp 3 prediktioner\n",
            "        axes[i, 1].axis('off')\n",
            "        prediction_text = '\\n'.join([\n",
            "            f\"{j+1}. {pred[1]}: {pred[2]:.4f}\" \n",
            "            for j, pred in enumerate(decoded)\n",
            "        ])\n",
            "        axes[i, 1].text(0.1, 0.5, f\"Topp 3 prediktioner:\\n\\n{prediction_text}\", \n",
            "                        fontsize=10, verticalalignment='center',\n",
            "                        bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))\n",
            "        \n",
            "        # Skriv ut i konsolen\n",
            "        print(f\"\\n{path}:\")\n",
            "        for j, (_, label, score) in enumerate(decoded, 1):\n",
            "            print(f\"  {j}. {label}: {score:.4f} ({score*100:.2f}%)\")\n",
            "    \n",
            "    plt.tight_layout()\n",
            "    plt.show()\n",
            "else:\n",
            "    print(\"Inga bilder hittades. Kontrollera sökvägarna.\")"
        ]
    })
    
    # Cell 7: Del B - Streamlit applikation
    notebook["cells"].append({
        "cell_type": "markdown",
        "id": "part_b_header",
        "metadata": {},
        "source": [
            "# Del B: Bygg Streamlit-applikation\n",
            "\n",
            "Vi skapar en avancerad Streamlit-applikation där användare kan ladda upp bilder och få prediktioner från förtränad modell."
        ]
    })
    
    notebook["cells"].append({
        "cell_type": "markdown",
        "id": "streamlit_code",
        "metadata": {},
        "source": [
            "## Avancerad Streamlit-applikationskod\n",
            "\n",
            "Koden nedan skapar en avancerad Streamlit-applikation med fler funktioner. Spara den i en separat fil (t.ex. `image_classifier_app.py`) och kör den med kommandot:\n",
            "\n",
            "```bash\n",
            "python -m streamlit run image_classifier_app.py\n",
            "```"
        ]
    })
    
    notebook["cells"].append({
        "cell_type": "code",
        "execution_count": None,
        "id": "create_streamlit_code",
        "metadata": {},
        "outputs": [],
        "source": [
            "# Här är koden för den avancerade Streamlit-applikationen\n",
            "# Denna kod skapar en fil: image_classifier_app.py\n",
            "\n",
            "streamlit_code = '''\n",
            "import streamlit as st\n",
            "import numpy as np\n",
            "from PIL import Image\n",
            "import tensorflow as tf\n",
            "from tensorflow.keras.applications.resnet50 import ResNet50\n",
            "from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions\n",
            "from tensorflow.keras.preprocessing import image\n",
            "import pandas as pd\n",
            "import matplotlib.pyplot as plt\n",
            "\n",
            "# Konfigurera Streamlit\n",
            "st.set_page_config(\n",
            "    page_title=\"Avancerad Bildklassificering\",\n",
            "    page_icon=\"\",\n",
            "    layout=\"wide\"\n",
            ")\n",
            "\n",
            "@st.cache_resource\n",
            "def load_model():\n",
            "    return ResNet50(weights='imagenet')\n",
            "\n",
            "def preprocess_image(uploaded_file):\n",
            "    img_pil = Image.open(uploaded_file)\n",
            "    if img_pil.mode != 'RGB':\n",
            "        img_pil = img_pil.convert('RGB')\n",
            "    img_resized = img_pil.resize((224, 224))\n",
            "    img_array = image.img_to_array(img_resized)\n",
            "    img_batch = np.expand_dims(img_array, axis=0)\n",
            "    img_preprocessed = preprocess_input(img_batch)\n",
            "    return img_preprocessed, img_pil\n",
            "\n",
            "def predict_image(model, img_preprocessed, top_n=5):\n",
            "    predictions = model.predict(img_preprocessed, verbose=0)\n",
            "    decoded = decode_predictions(predictions, top=top_n)[0]\n",
            "    return decoded\n",
            "\n",
            "def main():\n",
            "    st.title(\" Avancerad Bildklassificering\")\n",
            "    st.markdown(\"**Klassificera bilder med ResNet50 - Förtränad CNN-modell**\")\n",
            "    \n",
            "    # Sidebar med inställningar\n",
            "    with st.sidebar:\n",
            "        st.header(\" Inställningar\")\n",
            "        top_n = st.slider(\"Antal topprediktioner att visa\", 3, 10, 5)\n",
            "        show_details = st.checkbox(\"Visa detaljerad information\", value=True)\n",
            "        \n",
            "        st.markdown(\"---\")\n",
            "        st.header(\"ℹ Om applikationen\")\n",
            "        st.info(\"\"\"\n",
            "        Denna applikation använder ResNet50, en förtränad CNN-modell.\n",
            "        Modellen kan klassificera 1000 olika klasser från ImageNet.\n",
            "        \"\"\")\n",
            "    \n",
            "    # Ladda modell\n",
            "    with st.spinner('Laddar modell...'):\n",
            "        model = load_model()\n",
            "    st.success(' Modell laddad!')\n",
            "    \n",
            "    st.markdown(\"---\")\n",
            "    \n",
            "    # Huvudfunktionalitet\n",
            "    col1, col2 = st.columns([1, 1])\n",
            "    \n",
            "    with col1:\n",
            "        st.header(\" Ladda upp bild\")\n",
            "        uploaded_file = st.file_uploader(\n",
            "            \"Välj en bildfil\",\n",
            "            type=['png', 'jpg', 'jpeg'],\n",
            "            help=\"Ladda upp en bild för klassificering\"\n",
            "        )\n",
            "    \n",
            "    if uploaded_file is not None:\n",
            "        # Visa bild\n",
            "        img_pil = Image.open(uploaded_file)\n",
            "        \n",
            "        col1, col2 = st.columns(2)\n",
            "        \n",
            "        with col1:\n",
            "            st.subheader(\" Originalbild\")\n",
            "            st.image(img_pil, caption=\"Uppladdad bild\", use_container_width=True)\n",
            "            st.caption(f\"Storlek: {img_pil.size[0]}x{img_pil.size[1]} pixlar\")\n",
            "        \n",
            "        # Prediktera\n",
            "        with st.spinner('Analyserar bild...'):\n",
            "            img_preprocessed, _ = preprocess_image(uploaded_file)\n",
            "            predictions = predict_image(model, img_preprocessed, top_n=top_n)\n",
            "        \n",
            "        with col2:\n",
            "            st.subheader(\" Prediktionsresultat\")\n",
            "            \n",
            "            # Visa topprediktioner med progress bars\n",
            "            for i, (_, label, score) in enumerate(predictions, 1):\n",
            "                st.write(f\"**{i}. {label}**\")\n",
            "                st.progress(float(score))\n",
            "                st.caption(f\"{score:.4f} ({score*100:.2f}%)\")\n",
            "        \n",
            "        # Detaljerad information\n",
            "        if show_details:\n",
            "            st.markdown(\"---\")\n",
            "            \n",
            "            # Tabell\n",
            "            st.subheader(\" Detaljerad tabell\")\n",
            "            results_df = pd.DataFrame([\n",
            "                {'Rank': i+1, 'Klass': label, 'Sannolikhet': f\"{score:.4f}\", 'Procent': f\"{score*100:.2f}%\"}\n",
            "                for i, (_, label, score) in enumerate(predictions)\n",
            "            ])\n",
            "            st.dataframe(results_df, use_container_width=True)\n",
            "            \n",
            "            # Visualisering\n",
            "            st.subheader(\" Sannolikhetsfördelning\")\n",
            "            fig, ax = plt.subplots(figsize=(10, 6))\n",
            "            labels = [pred[1] for pred in predictions]\n",
            "            scores = [pred[2] for pred in predictions]\n",
            "            ax.barh(labels, scores)\n",
            "            ax.set_xlabel('Sannolikhet')\n",
            "            ax.set_title('Topp prediktioner')\n",
            "            plt.tight_layout()\n",
            "            st.pyplot(fig)\n",
            "        \n",
            "        # Bästa prediktion\n",
            "        best_pred = predictions[0]\n",
            "        st.success(f\" **Bästa prediktion:** {best_pred[1]} ({best_pred[2]*100:.2f}%)\")\n",
            "\n",
            "if __name__ == \"__main__\":\n",
            "    main()\n",
            "'''\n",
            "\n",
            "# Skriv koden till en fil\n",
            "with open('image_classifier_app.py', 'w', encoding='utf-8') as f:\n",
            "    f.write(streamlit_code)\n",
            "\n",
            "print(\"Avancerad Streamlit-applikationskod sparad i 'image_classifier_app.py'\")\n",
            "print(\"\\nFör att köra applikationen, använd kommandot:\")\n",
            "print(\"python -m streamlit run image_classifier_app.py\")"
        ]
    })
    
    # Cell 8: Var kan man hitta bilder att testa med?
    notebook["cells"].append({
        "cell_type": "markdown",
        "id": "where_to_find_images",
        "metadata": {},
        "source": [
            "## Var kan man hitta bilder att testa med?\n",
            "\n",
            "Här är några bra källor för att hitta bilder att testa med applikationen:\n",
            "\n",
            "### 1. **Egna bilder**\n",
            "- Ta bilder med din mobil eller kamera\n",
            "- Bilderna kan vara på djur, fordon, mat, elektronik, möbler, etc.\n",
            "- Spara bilderna i format PNG, JPG eller JPEG\n",
            "\n",
            "### 2. **Unsplash** (Gratis, hög kvalitet)\n",
            "- URL: https://unsplash.com/\n",
            "- Gratis bilder med hög kvalitet\n",
            "- Sök efter objekt som finns i ImageNet (hundar, katter, bilar, etc.)\n",
            "- Exempel: https://unsplash.com/s/photos/dog, https://unsplash.com/s/photos/car\n",
            "\n",
            "### 3. **Pexels** (Gratis)\n",
            "- URL: https://www.pexels.com/\n",
            "- Gratis stockfoton\n",
            "- Bra sökfunktion för att hitta specifika objekt\n",
            "\n",
            "### 4. **Pixabay** (Gratis)\n",
            "- URL: https://pixabay.com/\n",
            "- Gratis bilder och illustrationer\n",
            "- Stort urval av olika kategorier\n",
            "\n",
            "### 5. **ImageNet** (För testbilder)\n",
            "- URL: https://www.image-net.org/\n",
            "- Det dataset som ResNet50 tränades på\n",
            "- Du kan hitta exempelbilder för olika klasser\n",
            "- Lista över alla klasser: https://deeplearning.cms.waikato.ac.nz/user-guide/class-maps/IMAGENET/\n",
            "\n",
            "### 6. **Kaggle Datasets**\n",
            "- URL: https://www.kaggle.com/datasets\n",
            "- Många bilddatasets att välja mellan\n",
            "- Sök efter \"image classification\" eller specifika objekt\n",
            "\n",
            "### 7. **Google Images** (För test)\n",
            "- Sök efter objekt och ladda ner bilder\n",
            "- Tänk på upphovsrätt när du använder bilder\n",
            "- Använd \"Tools\" > \"Usage Rights\" > \"Labeled for reuse\"\n",
            "\n",
            "### Tips för bästa resultat:\n",
            "- Använd tydliga bilder med bra belysning\n",
            "- Objektet ska vara i fokus och tydligt synligt\n",
            "- Försök att ha objektet i mitten av bilden\n",
            "- Bilderna kan vara i olika storlekar - appen ändrar automatiskt till 224x224\n",
            "- Testa med olika typer av objekt för att se modellens förmåga"
        ]
    })
    
    # Cell 9: Exempel på bildkällor
    notebook["cells"].append({
        "cell_type": "code",
        "execution_count": None,
        "id": "example_image_sources",
        "metadata": {},
        "outputs": [],
        "source": [
            "# Exempel på hur du kan ladda ner bilder programmatiskt (valfritt)\n",
            "# Detta är endast ett exempel - du kan också ladda ner bilder manuellt\n",
            "\n",
            "import requests\n",
            "from io import BytesIO\n",
            "\n",
            "def download_image_from_url(url, save_path):\n",
            "    \"\"\"\n",
            "    Ladda ner en bild från en URL och spara den lokalt.\n",
            "    \n",
            "    Args:\n",
            "        url: URL till bilden\n",
            "        save_path: Var bilden ska sparas\n",
            "    \"\"\"\n",
            "    try:\n",
            "        response = requests.get(url, timeout=10)\n",
            "        response.raise_for_status()\n",
            "        \n",
            "        img = Image.open(BytesIO(response.content))\n",
            "        img.save(save_path)\n",
            "        print(f\"Bild sparad till: {save_path}\")\n",
            "        return True\n",
            "    except Exception as e:\n",
            "        print(f\"Fel vid nedladdning: {e}\")\n",
            "        return False\n",
            "\n",
            "# Exempel: Ladda ner en testbild (kommentera bort om du inte vill använda detta)\n",
            "# test_image_url = \"https://images.unsplash.com/photo-1583337130417-3346a1be7dee\"  # Exempel: katt\n",
            "# download_image_from_url(test_image_url, \"test_bild.jpg\")\n",
            "\n",
            "print(\"Funktion för att ladda ner bilder från URL definierad.\")\n",
            "print(\"\\nTips: Du kan också använda dina egna bilder eller ladda ner manuellt från Unsplash, Pexels, etc.\")"
        ]
    })
    
    # Cell 10: Instruktioner för att köra Streamlit-appen
    notebook["cells"].append({
        "cell_type": "markdown",
        "id": "run_instructions",
        "metadata": {},
        "source": [
            "from PIL import Image\n",
            "import tensorflow as tf\n",
            "from tensorflow.keras.applications.resnet50 import ResNet50\n",
            "from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions\n",
            "from tensorflow.keras.preprocessing import image\n",
            "import pandas as pd\n",
            "import matplotlib.pyplot as plt\n",
            "\n",
            "# Konfigurera Streamlit\n",
            "st.set_page_config(\n",
            "    page_title=\"Avancerad Bildklassificering\",\n",
            "    page_icon=\"\",\n",
            "    layout=\"wide\"\n",
            ")\n",
            "\n",
            "@st.cache_resource\n",
            "def load_model():\n",
            "    return ResNet50(weights='imagenet')\n",
            "\n",
            "def preprocess_image(uploaded_file):\n",
            "    img_pil = Image.open(uploaded_file)\n",
            "    if img_pil.mode != 'RGB':\n",
            "        img_pil = img_pil.convert('RGB')\n",
            "    img_resized = img_pil.resize((224, 224))\n",
            "    img_array = image.img_to_array(img_resized)\n",
            "    img_batch = np.expand_dims(img_array, axis=0)\n",
            "    img_preprocessed = preprocess_input(img_batch)\n",
            "    return img_preprocessed, img_pil\n",
            "\n",
            "def predict_image(model, img_preprocessed, top_n=5):\n",
            "    predictions = model.predict(img_preprocessed, verbose=0)\n",
            "    decoded = decode_predictions(predictions, top=top_n)[0]\n",
            "    return decoded\n",
            "\n",
            "def main():\n",
            "    st.title(\" Avancerad Bildklassificering\")\n",
            "    st.markdown(\"**Klassificera bilder med ResNet50 - Förtränad CNN-modell**\")\n",
            "    \n",
            "    # Sidebar med inställningar\n",
            "    with st.sidebar:\n",
            "        st.header(\" Inställningar\")\n",
            "        top_n = st.slider(\"Antal topprediktioner att visa\", 3, 10, 5)\n",
            "        show_details = st.checkbox(\"Visa detaljerad information\", value=True)\n",
            "        \n",
            "        st.markdown(\"---\")\n",
            "        st.header(\"ℹ Om applikationen\")\n",
            "        st.info(\"\"\"\n",
            "        Denna applikation använder ResNet50, en förtränad CNN-modell.\n",
            "        Modellen kan klassificera 1000 olika klasser från ImageNet.\n",
            "        \"\"\")\n",
            "    \n",
            "    # Ladda modell\n",
            "    with st.spinner('Laddar modell...'):\n",
            "        model = load_model()\n",
            "    st.success(' Modell laddad!')\n",
            "    \n",
            "    st.markdown(\"---\")\n",
            "    \n",
            "    # Huvudfunktionalitet\n",
            "    col1, col2 = st.columns([1, 1])\n",
            "    \n",
            "    with col1:\n",
            "        st.header(\" Ladda upp bild\")\n",
            "        uploaded_file = st.file_uploader(\n",
            "            \"Välj en bildfil\",\n",
            "            type=['png', 'jpg', 'jpeg'],\n",
            "            help=\"Ladda upp en bild för klassificering\"\n",
            "        )\n",
            "    \n",
            "    if uploaded_file is not None:\n",
            "        # Visa bild\n",
            "        img_pil = Image.open(uploaded_file)\n",
            "        \n",
            "        col1, col2 = st.columns(2)\n",
            "        \n",
            "        with col1:\n",
            "            st.subheader(\" Originalbild\")\n",
            "            st.image(img_pil, caption=\"Uppladdad bild\", use_container_width=True)\n",
            "            st.caption(f\"Storlek: {img_pil.size[0]}x{img_pil.size[1]} pixlar\")\n",
            "        \n",
            "        # Prediktera\n",
            "        with st.spinner('Analyserar bild...'):\n",
            "            img_preprocessed, _ = preprocess_image(uploaded_file)\n",
            "            predictions = predict_image(model, img_preprocessed, top_n=top_n)\n",
            "        \n",
            "        with col2:\n",
            "            st.subheader(\" Prediktionsresultat\")\n",
            "            \n",
            "            # Visa topprediktioner med progress bars\n",
            "            for i, (_, label, score) in enumerate(predictions, 1):\n",
            "                st.write(f\"**{i}. {label}**\")\n",
            "                st.progress(float(score))\n",
            "                st.caption(f\"{score:.4f} ({score*100:.2f}%)\")\n",
            "        \n",
            "        # Detaljerad information\n",
            "        if show_details:\n",
            "            st.markdown(\"---\")\n",
            "            \n",
            "            # Tabell\n",
            "            st.subheader(\" Detaljerad tabell\")\n",
            "            results_df = pd.DataFrame([\n",
            "                {'Rank': i+1, 'Klass': label, 'Sannolikhet': f\"{score:.4f}\", 'Procent': f\"{score*100:.2f}%\"}\n",
            "                for i, (_, label, score) in enumerate(predictions)\n",
            "            ])\n",
            "            st.dataframe(results_df, use_container_width=True)\n",
            "            \n",
            "            # Visualisering\n",
            "            st.subheader(\" Sannolikhetsfördelning\")\n",
            "            fig, ax = plt.subplots(figsize=(10, 6))\n",
            "            labels = [pred[1] for pred in predictions]\n",
            "            scores = [pred[2] for pred in predictions]\n",
            "            ax.barh(labels, scores)\n",
            "            ax.set_xlabel('Sannolikhet')\n",
            "            ax.set_title('Topp prediktioner')\n",
            "            plt.tight_layout()\n",
            "            st.pyplot(fig)\n",
            "        \n",
            "        # Bästa prediktion\n",
            "        best_pred = predictions[0]\n",
            "        st.success(f\" **Bästa prediktion:** {best_pred[1]} ({best_pred[2]*100:.2f}%)\")\n",
            "\n",
            "if __name__ == \"__main__\":\n",
            "    main()\n",
            "'''\n",
            "\n",
            "# Skriv koden till en fil\n",
            "with open('image_classifier_app.py', 'w', encoding='utf-8') as f:\n",
            "    f.write(streamlit_code)\n",
            "\n",
            "print(\"Avancerad Streamlit-applikationskod sparad i 'image_classifier_app.py'\")"
        ]
    })
    
    # Cell 9: Instruktioner för att köra Streamlit-appen
    notebook["cells"].append({
        "cell_type": "markdown",
        "id": "run_instructions",
        "metadata": {},
        "source": [
            "## Instruktioner för att köra Streamlit-applikationen\n",
            "\n",
            "1. **Installera Streamlit** (om du inte redan har det):\n",
            "   ```bash\n",
            "   pip install streamlit\n",
            "   ```\n",
            "\n",
            "2. **Kör applikationen:**\n",
            "   ```bash\n",
            "   python -m streamlit run image_classifier_app.py\n",
            "   ```\n",
            "   \n",
            "   **Notera:** I Git Bash/MINGW64, använd `python -m streamlit` istället för bara `streamlit`.\n",
            "\n",
            "3. Applikationen öppnas automatiskt i din webbläsare på `http://localhost:8501`\n",
            "\n",
            "4. Ladda upp en bild och se prediktionerna!\n",
            "\n",
            "5. **Tips:** Se cellen ovan för var du kan hitta bilder att testa med."
        ]
    })
    
    # Cell 10: Sammanfattning och reflektioner
    notebook["cells"].append({
        "cell_type": "markdown",
        "id": "summary",
        "metadata": {},
        "source": [
            "# Sammanfattning och reflektioner\n",
            "\n",
            "## Vad har vi lärt oss?\n",
            "\n",
            "1. **Förtränade modeller (Del A)**: Vi använde ResNet50, en förtränad CNN-modell, för att prediktera egna bilder. Detta är mycket kraftfullt eftersom vi inte behöver träna modellen själva.\n",
            "\n",
            "2. **Streamlit-applikationer (Del B)**: Vi byggde en interaktiv webapplikation där användare kan ladda upp bilder och få omedelbara prediktioner. Detta visar hur man kan göra ML-modeller tillgängliga för användare utan teknisk kunskap.\n",
            "\n",
            "## Fördelar med förtränade modeller:\n",
            "\n",
            "- **Snabbhet**: Ingen träning krävs, modellen är redan tränad\n",
            "- **Prestanda**: Förtränade modeller har ofta mycket bra prestanda eftersom de tränats på stora dataset\n",
            "- **Användbarhet**: Kan användas direkt för många olika uppgifter\n",
            "- **Resursbesparing**: Sparar tid och beräkningsresurser\n",
            "\n",
            "## Begränsningar:\n",
            "\n",
            "- **Begränsade klasser**: ResNet50 kan endast klassificera de 1000 klasser som finns i ImageNet\n",
            "- **Ingen anpassning**: Modellen kan inte läras nya klasser utan träning\n",
            "- **Preprocessing**: Bilder måste förbehandlas på ett specifikt sätt (224x224, RGB, etc.)\n",
            "\n",
            "## Tips för vidareutveckling:\n",
            "\n",
            "- **Transfer Learning**: Använd förtränade modeller som bas och träna om de sista lagren för dina egna klasser\n",
            "- **Fler modeller**: Testa andra förtränade modeller (VGG16, MobileNet, EfficientNet, etc.)\n",
            "- **Data augmentation**: Lägg till funktioner för att rotera, zooma eller ändra ljusstyrka på bilder\n",
            "- **Batch processing**: Tillåt användare att ladda upp flera bilder samtidigt\n",
            "- **Exportfunktioner**: Låt användare spara resultat som CSV eller PDF\n",
            "- **Visualiseringar**: Lägg till mer avancerade visualiseringar av prediktionerna\n",
            "\n",
            "## Ytterligare läsning:\n",
            "\n",
            "- Keras Applications: https://keras.io/api/applications/\n",
            "- Streamlit dokumentation: https://docs.streamlit.io/\n",
            "- ImageNet: https://www.image-net.org/\n",
            "- ResNet paper: https://arxiv.org/abs/1512.03385"
        ]
    })
    
    return notebook

def main():
    """Huvudfunktion som skapar notebooken."""
    notebook = create_notebook()
    
    # Spara notebooken
    output_file = "Kapitel8_Uppgift9_Fortranade_Modeller_Streamlit.ipynb"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(notebook, f, indent=1, ensure_ascii=False)
    
    print(f"Notebook skapad: {output_file}")
    print(f"Totalt antal celler: {len(notebook['cells'])}")

if __name__ == "__main__":
    main()
