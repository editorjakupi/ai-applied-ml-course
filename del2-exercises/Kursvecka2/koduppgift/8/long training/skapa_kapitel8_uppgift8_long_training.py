"""
Python-skript som genererar en Jupyter notebook för Koduppgift 8, Kapitel 8.
OPTIMERAD VERSION för bästa resultat:
- Använder hela CIFAR-100 datasetet (inga begränsningar)
- Ökade epoker för alla delar
- Callbacks för bättre träning (EarlyStopping, ReduceLROnPlateau)
- Ökade max_trials för KerasTuner
"""

import json

def create_notebook():
    """Skapar en komplett notebook för Koduppgift 8, Kapitel 8 - OPTIMERAD VERSION."""
    
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
            "# Koduppgift 8 - Kapitel 8: CNN med CIFAR-100 (OPTIMERAD VERSION)\n",
            "\n",
            "**OPTIMERAD VERSION FÖR BÄSTA RESULTAT:**\n",
            "- Använder hela CIFAR-100 datasetet (50 000 träningsbilder, 10 000 testbilder)\n",
            "- Ökade epoker för bättre träning\n",
            "- Callbacks för att förhindra överträning och optimera learning rate\n",
            "- Ökade max_trials för KerasTuner\n",
            "\n",
            "I denna uppgift arbetar vi med CIFAR-100 datasetet som gicks igenom i kodexempel 1.\n",
            "\n",
            "**Uppgiften består av tre delar:**\n",
            "\n",
            "a) Skapa en CNN-modell för att prediktera datasetet.\n",
            "\n",
            "b) Om du justerar hyperparametrar med KerasTuner, får du bättre resultat?\n",
            "\n",
            "c) Prova använd transfer learning för att genomföra prediktioner, får du bättre resultat?\n",
            "\n",
            "Mer information om CIFAR-100: https://www.cs.toronto.edu/~kriz/cifar.html\n",
            "\n",
            "**VIKTIGT:** Denna version tar betydligt längre tid att köra än standardversionen, men ger bättre resultat!"
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
            "import tensorflow as tf\n",
            "from tensorflow import keras\n",
            "from tensorflow.keras.datasets import cifar100\n",
            "from tensorflow.keras.models import Sequential\n",
            "from tensorflow.keras.layers import (\n",
            "    Input, Conv2D, MaxPooling2D, Flatten, \n",
            "    Dropout, Dense\n",
            ")\n",
            "from tensorflow.keras.applications import MobileNetV2, ResNet50\n",
            "from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau\n",
            "import keras_tuner as kt\n",
            "import os\n",
            "import tempfile\n",
            "\n",
            "print(f\"TensorFlow version: {tf.__version__}\")\n",
            "print(f\"Keras version: {keras.__version__}\")"
        ]
    })
    
    # Cell 3: Del A - Ladda och förbereda data
    notebook["cells"].append({
        "cell_type": "markdown",
        "id": "part_a_header",
        "metadata": {},
        "source": [
            "# Del A: Skapa en CNN-modell för att prediktera datasetet"
        ]
    })
    
    notebook["cells"].append({
        "cell_type": "markdown",
        "id": "data_loading",
        "metadata": {},
        "source": [
            "## Ladda och förbereda data\n",
            "\n",
            "**OPTIMERING:** Vi använder hela CIFAR-100 datasetet för bästa resultat!"
        ]
    })
    
    notebook["cells"].append({
        "cell_type": "code",
        "execution_count": None,
        "id": "load_data",
        "metadata": {},
        "outputs": [],
        "source": [
            "# Ladda CIFAR-100 dataset\n",
            "(x_train, y_train), (x_test, y_test) = cifar100.load_data()\n",
            "\n",
            "# OPTIMERING: Använd hela datasetet för bästa resultat!\n",
            "# Ingen begränsning - vi använder alla 50 000 träningsbilder och 10 000 testbilder\n",
            "# Detta ger bättre resultat men tar längre tid att träna\n",
            "\n",
            "# Normalisera pixlar till intervallet [0, 1]\n",
            "x_train = x_train / 255.0\n",
            "x_test = x_test / 255.0\n",
            "\n",
            "# Kontrollera dimensioner\n",
            "print(f\"Träningsdata shape: {x_train.shape}\")\n",
            "print(f\"Träningslabels shape: {y_train.shape}\")\n",
            "print(f\"Testdata shape: {x_test.shape}\")\n",
            "print(f\"Testlabels shape: {y_test.shape}\")\n",
            "print(f\"\\nCIFAR-100 har 100 klasser.\")\n",
            "print(f\"Varje bild är 32x32 pixlar med 3 kanaler (RGB).\")\n",
            "print(f\"\\nVi använder hela datasetet: {len(x_train)} träningsbilder och {len(x_test)} testbilder!\")"
        ]
    })
    
    # Cell 4: Visualisera data
    notebook["cells"].append({
        "cell_type": "code",
        "execution_count": None,
        "id": "visualize_data",
        "metadata": {},
        "outputs": [],
        "source": [
            "# Visualisera några bilder från datasetet\n",
            "plt.figure(figsize=(10, 10))\n",
            "for i in range(9):\n",
            "    plt.subplot(3, 3, i + 1)\n",
            "    plt.imshow(x_train[i])\n",
            "    plt.axis('off')\n",
            "    plt.title(f'Label: {y_train[i][0]}')\n",
            "plt.tight_layout()\n",
            "plt.show()"
        ]
    })
    
    # Cell 5: Bygg CNN-modell
    notebook["cells"].append({
        "cell_type": "markdown",
        "id": "build_model_a",
        "metadata": {},
        "source": [
            "## Bygg CNN-modell\n",
            "\n",
            "Vi bygger en CNN-modell baserad på kodexempel 1 från kapitel 8."
        ]
    })
    
    notebook["cells"].append({
        "cell_type": "code",
        "execution_count": None,
        "id": "create_model_a",
        "metadata": {},
        "outputs": [],
        "source": [
            "# Skapa CNN-modell\n",
            "model_a = Sequential()\n",
            "\n",
            "# Input layer\n",
            "model_a.add(Input(shape=(32, 32, 3)))\n",
            "\n",
            "# Convolutional layers med MaxPooling\n",
            "# Första blocket: 32 filter\n",
            "model_a.add(Conv2D(32, kernel_size=(3, 3), padding='same', activation='relu'))\n",
            "model_a.add(MaxPooling2D(pool_size=(2, 2)))\n",
            "\n",
            "# Andra blocket: 64 filter\n",
            "model_a.add(Conv2D(64, kernel_size=(3, 3), padding='same', activation='relu'))\n",
            "model_a.add(MaxPooling2D(pool_size=(2, 2)))\n",
            "\n",
            "# Tredje blocket: 128 filter\n",
            "model_a.add(Conv2D(128, kernel_size=(3, 3), padding='same', activation='relu'))\n",
            "model_a.add(MaxPooling2D(pool_size=(2, 2)))\n",
            "\n",
            "# Fjärde blocket: 256 filter\n",
            "model_a.add(Conv2D(256, kernel_size=(3, 3), padding='same', activation='relu'))\n",
            "model_a.add(MaxPooling2D(pool_size=(2, 2)))\n",
            "\n",
            "# Flatten och Dense layers\n",
            "model_a.add(Flatten())\n",
            "model_a.add(Dropout(0.5))  # Regularisering\n",
            "model_a.add(Dense(512, activation='relu'))\n",
            "model_a.add(Dense(100, activation='softmax'))  # 100 klasser\n",
            "\n",
            "# Visa modellstruktur\n",
            "model_a.summary()"
        ]
    })
    
    # Cell 6: Kompilera och träna modell A
    notebook["cells"].append({
        "cell_type": "markdown",
        "id": "train_a_header",
        "metadata": {},
        "source": [
            "## Kompilera och träna modell\n",
            "\n",
            "**OPTIMERING:** Ökade epoker till 30 med callbacks för bättre träning!"
        ]
    })
    
    notebook["cells"].append({
        "cell_type": "code",
        "execution_count": None,
        "id": "compile_train_a",
        "metadata": {},
        "outputs": [],
        "source": [
            "# Kompilera modellen\n",
            "model_a.compile(\n",
            "    loss='sparse_categorical_crossentropy',\n",
            "    optimizer='adam',\n",
            "    metrics=['accuracy']\n",
            ")\n",
            "\n",
            "# OPTIMERING: Definiera callbacks för bättre träning\n",
            "# EarlyStopping: Stoppa träning om val_loss inte förbättras\n",
            "# ReduceLROnPlateau: Minska learning rate om val_loss stagnerar\n",
            "callbacks_a = [\n",
            "    EarlyStopping(\n",
            "        monitor='val_loss',\n",
            "        patience=5,  # Vänta 5 epoker innan stopp\n",
            "        restore_best_weights=True,  # Återställ till bästa vikter\n",
            "        verbose=1\n",
            "    ),\n",
            "    ReduceLROnPlateau(\n",
            "        monitor='val_loss',\n",
            "        factor=0.5,  # Halvera learning rate\n",
            "        patience=3,  # Vänta 3 epoker\n",
            "        min_lr=1e-7,  # Minsta learning rate\n",
            "        verbose=1\n",
            "    )\n",
            "]\n",
            "\n",
            "# Träna modellen\n",
            "# OPTIMERING: Ökade till 30 epoker för bättre träning\n",
            "print(\"Börjar träning av Del A...\")\n",
            "print(\"Detta kan ta en stund eftersom vi använder hela datasetet!\")\n",
            "history_a = model_a.fit(\n",
            "    x_train, y_train,\n",
            "    epochs=30,  # Ökade från 10 till 30\n",
            "    batch_size=128,\n",
            "    validation_split=0.2,\n",
            "    callbacks=callbacks_a,  # Lägg till callbacks\n",
            "    verbose=2\n",
            ")"
        ]
    })
    
    # Cell 7: Utvärdera modell A
    notebook["cells"].append({
        "cell_type": "code",
        "execution_count": None,
        "id": "evaluate_a",
        "metadata": {},
        "outputs": [],
        "source": [
            "# Utvärdera modellen på testdata\n",
            "y_pred_a = model_a.predict(x_test, verbose=2)\n",
            "y_pred_labels_a = np.argmax(y_pred_a, axis=1)\n",
            "accuracy_a = np.mean(y_pred_labels_a == y_test.flatten())\n",
            "\n",
            "print(f\"\\nAccuracy på testdata (Del A): {accuracy_a:.4f}\")\n",
            "\n",
            "# Visualisera träningshistorik\n",
            "plt.figure(figsize=(12, 4))\n",
            "\n",
            "plt.subplot(1, 2, 1)\n",
            "plt.plot(history_a.history['accuracy'], label='Training Accuracy')\n",
            "plt.plot(history_a.history['val_accuracy'], label='Validation Accuracy')\n",
            "plt.xlabel('Epoch')\n",
            "plt.ylabel('Accuracy')\n",
            "plt.title('Model Accuracy - Del A')\n",
            "plt.legend()\n",
            "\n",
            "plt.subplot(1, 2, 2)\n",
            "plt.plot(history_a.history['loss'], label='Training Loss')\n",
            "plt.plot(history_a.history['val_loss'], label='Validation Loss')\n",
            "plt.xlabel('Epoch')\n",
            "plt.ylabel('Loss')\n",
            "plt.title('Model Loss - Del A')\n",
            "plt.legend()\n",
            "\n",
            "plt.tight_layout()\n",
            "plt.show()"
        ]
    })
    
    # Cell 8: Del B - KerasTuner
    notebook["cells"].append({
        "cell_type": "markdown",
        "id": "part_b_header",
        "metadata": {},
        "source": [
            "# Del B: Justera hyperparametrar med KerasTuner\n",
            "\n",
            "Vi använder KerasTuner för att automatiskt hitta bästa hyperparametrarna för vår CNN-modell.\n",
            "\n",
            "**OPTIMERING:** Ökade max_trials till 30 och epoker per trial till 10 för bättre resultat!"
        ]
    })
    
    notebook["cells"].append({
        "cell_type": "code",
        "execution_count": None,
        "id": "define_model_builder",
        "metadata": {},
        "outputs": [],
        "source": [
            "# Definiera funktion som bygger modell med hyperparametrar från tuner\n",
            "def build_model_cnn(hp):\n",
            "    model = Sequential()\n",
            "    model.add(Input(shape=(32, 32, 3)))\n",
            "    \n",
            "    # Första Conv2D blocket - antal filter kan variera\n",
            "    filters_1 = hp.Int('filters_1', min_value=16, max_value=64, step=16)\n",
            "    model.add(Conv2D(filters_1, kernel_size=(3, 3), padding='same', activation='relu'))\n",
            "    model.add(MaxPooling2D(pool_size=(2, 2)))\n",
            "    \n",
            "    # Andra Conv2D blocket\n",
            "    filters_2 = hp.Int('filters_2', min_value=32, max_value=128, step=32)\n",
            "    model.add(Conv2D(filters_2, kernel_size=(3, 3), padding='same', activation='relu'))\n",
            "    model.add(MaxPooling2D(pool_size=(2, 2)))\n",
            "    \n",
            "    # Tredje Conv2D blocket\n",
            "    filters_3 = hp.Int('filters_3', min_value=64, max_value=256, step=64)\n",
            "    model.add(Conv2D(filters_3, kernel_size=(3, 3), padding='same', activation='relu'))\n",
            "    model.add(MaxPooling2D(pool_size=(2, 2)))\n",
            "    \n",
            "    # Fjärde Conv2D blocket\n",
            "    filters_4 = hp.Int('filters_4', min_value=128, max_value=512, step=128)\n",
            "    model.add(Conv2D(filters_4, kernel_size=(3, 3), padding='same', activation='relu'))\n",
            "    model.add(MaxPooling2D(pool_size=(2, 2)))\n",
            "    \n",
            "    # Flatten och Dense layers\n",
            "    model.add(Flatten())\n",
            "    \n",
            "    # Dropout rate kan variera\n",
            "    dropout_rate = hp.Float('dropout_rate', min_value=0.3, max_value=0.7, step=0.1)\n",
            "    model.add(Dropout(dropout_rate))\n",
            "    \n",
            "    # Antal neuroner i Dense layer kan variera\n",
            "    dense_units = hp.Int('dense_units', min_value=256, max_value=1024, step=256)\n",
            "    model.add(Dense(dense_units, activation='relu'))\n",
            "    \n",
            "    # Output layer\n",
            "    model.add(Dense(100, activation='softmax'))\n",
            "    \n",
            "    # Learning rate kan variera\n",
            "    learning_rate = hp.Choice('learning_rate', values=[1e-3, 5e-4, 1e-4, 5e-5])\n",
            "    \n",
            "    model.compile(\n",
            "        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),\n",
            "        loss='sparse_categorical_crossentropy',\n",
            "        metrics=['accuracy']\n",
            "    )\n",
            "    \n",
            "    return model\n",
            "\n",
            "print(\"Model builder-funktion definierad!\")"
        ]
    })
    
    # Cell 9: Skapa och köra tuner
    notebook["cells"].append({
        "cell_type": "code",
        "execution_count": None,
        "id": "create_tuner",
        "metadata": {},
        "outputs": [],
        "source": [
            "# Skapa RandomSearch tuner\n",
            "tuner_dir = os.path.join(tempfile.gettempdir(), 'keras_tuner_cifar100')\n",
            "os.makedirs(tuner_dir, exist_ok=True)\n",
            "\n",
            "# OPTIMERING: Ökade max_trials från 10 till 30 för bättre hyperparameteroptimering\n",
            "max_trials = 30  # Antal modellkonfigurationer att testa\n",
            "# Detta tar längre tid men ger bättre resultat\n",
            "\n",
            "tuner = kt.RandomSearch(\n",
            "    build_model_cnn,\n",
            "    objective='val_accuracy',  # Maximera validation accuracy\n",
            "    max_trials=max_trials,\n",
            "    executions_per_trial=1,\n",
            "    directory=tuner_dir,\n",
            "    project_name='cifar100_cnn_optimization',\n",
            "    overwrite=True\n",
            ")\n",
            "\n",
            "print(f\"KerasTuner skapad! Kommer testa {max_trials} olika konfigurationer.\")\n",
            "print(\"Detta kommer ta betydligt längre tid än standardversionen!\")"
        ]
    })
    
    # Cell 10: Kör tuner search
    notebook["cells"].append({
        "cell_type": "code",
        "execution_count": None,
        "id": "run_tuner",
        "metadata": {},
        "outputs": [],
        "source": [
            "# Kör hyperparameter search\n",
            "# OPTIMERING: Ökade epoker per trial från 5 till 10\n",
            "print(\"Börjar hyperparameter search...\")\n",
            "print(\"Detta kan ta flera timmar beroende på din hårdvara!\")\n",
            "tuner.search(\n",
            "    x_train, y_train,\n",
            "    epochs=10,  # Ökade från 5 till 10 epoker per trial\n",
            "    validation_split=0.2,\n",
            "    batch_size=128,\n",
            "    verbose=2\n",
            ")\n",
            "\n",
            "# Visa sammanfattning av resultat\n",
            "tuner.results_summary()"
        ]
    })
    
    # Cell 11: Hämta bästa modellen och träna den
    notebook["cells"].append({
        "cell_type": "code",
        "execution_count": None,
        "id": "get_best_model",
        "metadata": {},
        "outputs": [],
        "source": [
            "# Hämta bästa modellen\n",
            "best_model_b = tuner.get_best_models(num_models=1)[0]\n",
            "\n",
            "# Visa bästa hyperparametrarna\n",
            "best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]\n",
            "print(\"\\nBästa hyperparametrarna:\")\n",
            "print(f\"Filters 1: {best_hps.get('filters_1')}\")\n",
            "print(f\"Filters 2: {best_hps.get('filters_2')}\")\n",
            "print(f\"Filters 3: {best_hps.get('filters_3')}\")\n",
            "print(f\"Filters 4: {best_hps.get('filters_4')}\")\n",
            "print(f\"Dropout rate: {best_hps.get('dropout_rate')}\")\n",
            "print(f\"Dense units: {best_hps.get('dense_units')}\")\n",
            "print(f\"Learning rate: {best_hps.get('learning_rate')}\")\n",
            "\n",
            "# OPTIMERING: Definiera callbacks för bästa modellen\n",
            "callbacks_b = [\n",
            "    EarlyStopping(\n",
            "        monitor='val_loss',\n",
            "        patience=5,\n",
            "        restore_best_weights=True,\n",
            "        verbose=1\n",
            "    ),\n",
            "    ReduceLROnPlateau(\n",
            "        monitor='val_loss',\n",
            "        factor=0.5,\n",
            "        patience=3,\n",
            "        min_lr=1e-7,\n",
            "        verbose=1\n",
            "    )\n",
            "]\n",
            "\n",
            "# Träna bästa modellen med fler epoker\n",
            "# OPTIMERING: Ökade till 30 epoker med callbacks\n",
            "print(\"\\nTränar bästa modellen med fler epoker...\")\n",
            "history_b = best_model_b.fit(\n",
            "    x_train, y_train,\n",
            "    epochs=30,  # Ökade från 10 till 30\n",
            "    batch_size=128,\n",
            "    validation_split=0.2,\n",
            "    callbacks=callbacks_b,  # Lägg till callbacks\n",
            "    verbose=2\n",
            ")"
        ]
    })
    
    # Cell 12: Utvärdera modell B
    notebook["cells"].append({
        "cell_type": "code",
        "execution_count": None,
        "id": "evaluate_b",
        "metadata": {},
        "outputs": [],
        "source": [
            "# Utvärdera bästa modellen på testdata\n",
            "y_pred_b = best_model_b.predict(x_test, verbose=2)\n",
            "y_pred_labels_b = np.argmax(y_pred_b, axis=1)\n",
            "accuracy_b = np.mean(y_pred_labels_b == y_test.flatten())\n",
            "\n",
            "print(f\"\\nAccuracy på testdata (Del B - KerasTuner): {accuracy_b:.4f}\")\n",
            "print(f\"Förbättring jämfört med Del A: {accuracy_b - accuracy_a:.4f}\")\n",
            "\n",
            "# Visualisera träningshistorik\n",
            "plt.figure(figsize=(12, 4))\n",
            "\n",
            "plt.subplot(1, 2, 1)\n",
            "plt.plot(history_b.history['accuracy'], label='Training Accuracy')\n",
            "plt.plot(history_b.history['val_accuracy'], label='Validation Accuracy')\n",
            "plt.xlabel('Epoch')\n",
            "plt.ylabel('Accuracy')\n",
            "plt.title('Model Accuracy - Del B (KerasTuner)')\n",
            "plt.legend()\n",
            "\n",
            "plt.subplot(1, 2, 2)\n",
            "plt.plot(history_b.history['loss'], label='Training Loss')\n",
            "plt.plot(history_b.history['val_loss'], label='Validation Loss')\n",
            "plt.xlabel('Epoch')\n",
            "plt.ylabel('Loss')\n",
            "plt.title('Model Loss - Del B (KerasTuner)')\n",
            "plt.legend()\n",
            "\n",
            "plt.tight_layout()\n",
            "plt.show()"
        ]
    })
    
    # Cell 13: Del C - Transfer Learning
    notebook["cells"].append({
        "cell_type": "markdown",
        "id": "part_c_header",
        "metadata": {},
        "source": [
            "# Del C: Transfer Learning\n",
            "\n",
            "Vi använder en förtränad modell (MobileNetV2) och anpassar den för CIFAR-100 genom transfer learning.\n",
            "\n",
            "**OPTIMERING:** Ökade epoker till 20 med callbacks för bättre träning!"
        ]
    })
    
    notebook["cells"].append({
        "cell_type": "code",
        "execution_count": None,
        "id": "prepare_data_transfer",
        "metadata": {},
        "outputs": [],
        "source": [
            "# För transfer learning behöver vi ändra bildstorleken\n",
            "# MobileNetV2 förväntar sig större bilder (minst 32x32, men 128x128 är bättre)\n",
            "# Vi använder TensorFlow för att ändra storlek\n",
            "x_train_resized = tf.image.resize(x_train, (128, 128))\n",
            "x_test_resized = tf.image.resize(x_test, (128, 128))\n",
            "\n",
            "print(f\"Ny träningsdata shape: {x_train_resized.shape}\")\n",
            "print(f\"Ny testdata shape: {x_test_resized.shape}\")\n",
            "print(f\"\\nVi använder hela datasetet: {len(x_train_resized)} träningsbilder!\")"
        ]
    })
    
    # Cell 14: Ladda förtränad modell
    notebook["cells"].append({
        "cell_type": "code",
        "execution_count": None,
        "id": "load_pretrained",
        "metadata": {},
        "outputs": [],
        "source": [
            "# Ladda MobileNetV2 som base model\n",
            "# include_top=False betyder att vi exkluderar output-lagret\n",
            "# Vi vill lägga till vårt eget output-lager för 100 klasser\n",
            "base_model = MobileNetV2(\n",
            "    weights='imagenet',  # Använd vikter tränade på ImageNet\n",
            "    include_top=False,  # Exkludera output-lagret\n",
            "    input_shape=(128, 128, 3)  # Input-storlek\n",
            ")\n",
            "\n",
            "# Frysa vikterna i base model (vi tränar inte om dessa initialt)\n",
            "base_model.trainable = False\n",
            "\n",
            "print(\"Förtränad MobileNetV2 modell laddad!\")\n",
            "print(f\"Antal lager i base model: {len(base_model.layers)}\")"
        ]
    })
    
    # Cell 15: Bygg modell med transfer learning
    notebook["cells"].append({
        "cell_type": "code",
        "execution_count": None,
        "id": "build_transfer_model",
        "metadata": {},
        "outputs": [],
        "source": [
            "# Bygg modell med transfer learning\n",
            "model_c = Sequential([\n",
            "    base_model,  # Förtränad base model\n",
            "    Flatten(),  # Konvertera till 1D\n",
            "    Dense(100, activation='softmax')  # Output layer för 100 klasser\n",
            "])\n",
            "\n",
            "# Kompilera modellen\n",
            "model_c.compile(\n",
            "    loss='sparse_categorical_crossentropy',\n",
            "    optimizer='adam',\n",
            "    metrics=['accuracy']\n",
            ")\n",
            "\n",
            "# Visa modellstruktur\n",
            "model_c.summary()"
        ]
    })
    
    # Cell 16: Träna transfer learning modell
    notebook["cells"].append({
        "cell_type": "code",
        "execution_count": None,
        "id": "train_transfer",
        "metadata": {},
        "outputs": [],
        "source": [
            "# OPTIMERING: Definiera callbacks för transfer learning\n",
            "callbacks_c = [\n",
            "    EarlyStopping(\n",
            "        monitor='val_loss',\n",
            "        patience=5,\n",
            "        restore_best_weights=True,\n",
            "        verbose=1\n",
            "    ),\n",
            "    ReduceLROnPlateau(\n",
            "        monitor='val_loss',\n",
            "        factor=0.5,\n",
            "        patience=3,\n",
            "        min_lr=1e-7,\n",
            "        verbose=1\n",
            "    )\n",
            "]\n",
            "\n",
            "# Träna modellen\n",
            "# Vi tränar endast de nya lagren (Flatten och Dense)\n",
            "# Base model-vikterna är frysta\n",
            "# OPTIMERING: Ökade till 20 epoker med callbacks\n",
            "print(\"Börjar träning av Transfer Learning modell...\")\n",
            "print(\"Detta kan ta en stund eftersom vi använder hela datasetet!\")\n",
            "history_c = model_c.fit(\n",
            "    x_train_resized, y_train,\n",
            "    epochs=20,  # Ökade från 5 till 20\n",
            "    batch_size=128,\n",
            "    validation_split=0.2,\n",
            "    callbacks=callbacks_c,  # Lägg till callbacks\n",
            "    verbose=2\n",
            ")"
        ]
    })
    
    # Cell 17: Utvärdera transfer learning modell
    notebook["cells"].append({
        "cell_type": "code",
        "execution_count": None,
        "id": "evaluate_c",
        "metadata": {},
        "outputs": [],
        "source": [
            "# Utvärdera modellen på testdata\n",
            "y_pred_c = model_c.predict(x_test_resized, verbose=2)\n",
            "y_pred_labels_c = np.argmax(y_pred_c, axis=1)\n",
            "accuracy_c = np.mean(y_pred_labels_c == y_test.flatten())\n",
            "\n",
            "print(f\"\\nAccuracy på testdata (Del C - Transfer Learning): {accuracy_c:.4f}\")\n",
            "print(f\"Förbättring jämfört med Del A: {accuracy_c - accuracy_a:.4f}\")\n",
            "\n",
            "# Visualisera träningshistorik\n",
            "plt.figure(figsize=(12, 4))\n",
            "\n",
            "plt.subplot(1, 2, 1)\n",
            "plt.plot(history_c.history['accuracy'], label='Training Accuracy')\n",
            "plt.plot(history_c.history['val_accuracy'], label='Validation Accuracy')\n",
            "plt.xlabel('Epoch')\n",
            "plt.ylabel('Accuracy')\n",
            "plt.title('Model Accuracy - Del C (Transfer Learning)')\n",
            "plt.legend()\n",
            "\n",
            "plt.subplot(1, 2, 2)\n",
            "plt.plot(history_c.history['loss'], label='Training Loss')\n",
            "plt.plot(history_c.history['val_loss'], label='Validation Loss')\n",
            "plt.xlabel('Epoch')\n",
            "plt.ylabel('Loss')\n",
            "plt.title('Model Loss - Del C (Transfer Learning)')\n",
            "plt.legend()\n",
            "\n",
            "plt.tight_layout()\n",
            "plt.show()"
        ]
    })
    
    # Cell 18: Jämförelse av alla tre metoder
    notebook["cells"].append({
        "cell_type": "markdown",
        "id": "comparison_header",
        "metadata": {},
        "source": [
            "# Jämförelse av resultat"
        ]
    })
    
    notebook["cells"].append({
        "cell_type": "code",
        "execution_count": None,
        "id": "compare_results",
        "metadata": {},
        "outputs": [],
        "source": [
            "# Jämför resultaten från alla tre delar\n",
            "results = {\n",
            "    'Del A (Baseline CNN)': accuracy_a,\n",
            "    'Del B (KerasTuner)': accuracy_b,\n",
            "    'Del C (Transfer Learning)': accuracy_c\n",
            "}\n",
            "\n",
            "print(\"\\n\" + \"=\"*50)\n",
            "print(\"JÄMFÖRELSE AV RESULTAT\")\n",
            "print(\"=\"*50)\n",
            "for method, acc in results.items():\n",
            "    print(f\"{method}: {acc:.4f}\")\n",
            "\n",
            "# Visualisera jämförelse\n",
            "plt.figure(figsize=(10, 6))\n",
            "methods = list(results.keys())\n",
            "accuracies = list(results.values())\n",
            "\n",
            "bars = plt.bar(methods, accuracies, color=['skyblue', 'lightgreen', 'lightcoral'])\n",
            "plt.ylabel('Accuracy')\n",
            "plt.title('Jämförelse av Accuracy för olika metoder (OPTIMERAD VERSION)')\n",
            "plt.ylim([0, max(accuracies) * 1.2])\n",
            "\n",
            "# Lägg till värden på staplarna\n",
            "for bar, acc in zip(bars, accuracies):\n",
            "    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,\n",
            "             f'{acc:.4f}', ha='center', va='bottom')\n",
            "\n",
            "plt.xticks(rotation=15, ha='right')\n",
            "plt.tight_layout()\n",
            "plt.show()\n",
            "\n",
            "# Analysera resultat\n",
            "print(\"\\n\" + \"=\"*50)\n",
            "print(\"ANALYS\")\n",
            "print(\"=\"*50)\n",
            "if accuracy_b > accuracy_a:\n",
            "    print(f\"KerasTuner förbättrade resultatet med {accuracy_b - accuracy_a:.4f}\")\n",
            "else:\n",
            "    print(f\"KerasTuner försämrade resultatet med {accuracy_a - accuracy_b:.4f}\")\n",
            "\n",
            "if accuracy_c > accuracy_a:\n",
            "    print(f\"Transfer Learning förbättrade resultatet med {accuracy_c - accuracy_a:.4f}\")\n",
            "else:\n",
            "    print(f\"Transfer Learning försämrade resultatet med {accuracy_a - accuracy_c:.4f}\")\n",
            "\n",
            "best_method = max(results, key=results.get)\n",
            "print(f\"\\nBästa metoden: {best_method} med accuracy {results[best_method]:.4f}\")"
        ]
    })
    
    # Cell 19: Sammanfattning och reflektioner
    notebook["cells"].append({
        "cell_type": "markdown",
        "id": "summary",
        "metadata": {},
        "source": [
            "# Sammanfattning och reflektioner\n",
            "\n",
            "## Vad har vi lärt oss?\n",
            "\n",
            "1. **Baseline CNN (Del A)**: Vi byggde en CNN-modell från grunden med Conv2D och MaxPooling2D lager.\n",
            "\n",
            "2. **Hyperparameteroptimering (Del B)**: Vi använde KerasTuner för att automatiskt hitta bästa hyperparametrarna. Detta kan förbättra resultatet, men kräver mer beräkningstid.\n",
            "\n",
            "3. **Transfer Learning (Del C)**: Vi använde en förtränad modell (MobileNetV2) och anpassade den för vår specifika uppgift. Detta är ofta mycket effektivt och ger bra resultat med mindre träning.\n",
            "\n",
            "## Optimeringar i denna version:\n",
            "\n",
            "-  **Hela datasetet**: Använder alla 50 000 träningsbilder och 10 000 testbilder\n",
            "-  **Ökade epoker**: Del A (30), Del B (30), Del C (20)\n",
            "-  **Callbacks**: EarlyStopping och ReduceLROnPlateau för bättre träning\n",
            "-  **Ökade max_trials**: 30 istället för 10 för KerasTuner\n",
            "-  **Ökade epoker per trial**: 10 istället för 5 i KerasTuner\n",
            "\n",
            "## Ytterligare förbättringsmöjligheter:\n",
            "\n",
            "- Experimentera med olika förtränade modeller (ResNet50, VGG16, EfficientNet, etc.)\n",
            "- Använd data augmentation för att öka storleken på träningsdatan\n",
            "- Prova att träna om de sista lagren i förtränade modeller (fine-tuning)\n",
            "- Experimentera med olika optimizers (AdamW, RMSprop, etc.)\n",
            "- Prova olika learning rate schedules\n",
            "\n",
            "## Ytterligare läsning:\n",
            "\n",
            "- Keras dokumentation: https://keras.io/\n",
            "- KerasTuner dokumentation: https://keras.io/guides/keras_tuner/\n",
            "- Transfer Learning guide: https://keras.io/guides/transfer_learning/"
        ]
    })
    
    return notebook

def main():
    """Huvudfunktion som skapar notebooken."""
    notebook = create_notebook()
    
    # Spara notebooken
    output_file = "Kapitel8_Uppgift8_CIFAR100_CNN_Long_Training.ipynb"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(notebook, f, indent=1, ensure_ascii=False)
    
    print(f"Notebook skapad: {output_file}")
    print(f"Totalt antal celler: {len(notebook['cells'])}")
    print("\n" + "="*60)
    print("VIKTIGT: Denna optimerade version tar betydligt längre tid att köra!")
    print("Förväntad träningstid:")
    print("- Del A: ~30-60 minuter (beroende på hårdvara)")
    print("- Del B: ~2-6 timmar (beroende på hårdvara)")
    print("- Del C: ~1-3 timmar (beroende på hårdvara)")
    print("="*60)

if __name__ == "__main__":
    main()
