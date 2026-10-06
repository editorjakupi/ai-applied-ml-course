"""
Skript för att generera Jupyter notebook med lösning till Koduppgift 10, Kapitel 7
Uppgift: Träna ANN-modell på MNIST och optimera med KerasTuner
Styl: Enkel och ren kod enligt deep_learning_koncist_exempel
"""
import json
import os
import tempfile

# Definiera notebook-strukturen
notebook = {
    "cells": [
        {
            "cell_type": "markdown",
            "id": "title",
            "metadata": {},
            "source": [
                "# Koduppgift 10 - Kapitel 7: ANN på MNIST med KerasTuner\n",
                "\n",
                "**Uppgift:**\n",
                "a) Träna en ANN-modell på MNIST-datan. Vad får du för resultat?\n",
                "\n",
                "b) Prova justera hyperparametrarna med KerasTuner. Får du bättre resultat?"
            ]
        },
        {
            "cell_type": "markdown",
            "id": "imports_section",
            "metadata": {},
            "source": [
                "# Importera Bibliotek\n",
                "\n",
                "Importerar nödvändiga bibliotek för att arbeta med MNIST, bygga neurala nätverk och optimera hyperparametrar."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "id": "imports",
            "metadata": {},
            "outputs": [],
            "source": [
                "import numpy as np\n",
                "import matplotlib.pyplot as plt\n",
                "import os\n",
                "import tempfile\n",
                "\n",
                "from tensorflow import keras\n",
                "from tensorflow.keras.models import Sequential\n",
                "from tensorflow.keras.layers import Dense, Flatten, Dropout, Input\n",
                "from tensorflow.keras.callbacks import EarlyStopping\n",
                "\n",
                "import keras_tuner as kt\n",
                "\n",
                "from sklearn.metrics import classification_report, confusion_matrix\n",
                "\n",
                "print(f\"TensorFlow version: {keras.__version__}\")"
            ]
        },
        {
            "cell_type": "markdown",
            "id": "random_seed_section",
            "metadata": {},
            "source": [
                "# Sätt Random Seed för Reproducerbarhet\n",
                "\n",
                "Sätter random seeds för att säkerställa att resultaten blir reproducerbara. Detta är viktigt för att kunna jämföra resultat mellan olika körningar."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "id": "random_seed",
            "metadata": {},
            "outputs": [],
            "source": [
                "# Sätt random seed för reproducerbarhet\n",
                "# np.random.seed(42): Säkerställer att NumPy genererar samma slumptal varje gång\n",
                "# keras.utils.set_random_seed(42): Säkerställer att TensorFlow/Keras använder samma initialisering\n",
                "# Varför 42? Det är en konvention inom ML-communityn (från \"Hitchhiker's Guide to the Galaxy\")\n",
                "np.random.seed(42)\n",
                "keras.utils.set_random_seed(42)\n",
                "\n",
                "print(\"Random seeds satta för reproducerbarhet!\")"
            ]
        },
        {
            "cell_type": "markdown",
            "id": "load_data_section",
            "metadata": {},
            "source": [
                "# Ladda och Förbereda Data\n",
                "\n",
                "Laddar MNIST-datasetet, normaliserar pixlar till [0,1] och delar upp i träning, validering och test."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "id": "load_mnist",
            "metadata": {},
            "outputs": [],
            "source": [
                "# Ladda MNIST-datasetet\n",
                "(X_train_full, y_train_full), (X_test, y_test) = keras.datasets.mnist.load_data()\n",
                "\n",
                "print(f\"Träningsdata form: {X_train_full.shape}\")\n",
                "print(f\"Testdata form: {X_test.shape}\")"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "id": "split_data",
            "metadata": {},
            "outputs": [],
            "source": [
                "# Dela upp träningsdata i träning och validering\n",
                "# Normalisera pixlar genom att dividera med 255.0 (pixlar är 0-255)\n",
                "X_valid, X_train = X_train_full[:5000] / 255.0, X_train_full[5000:] / 255.0\n",
                "y_valid, y_train = y_train_full[:5000], y_train_full[5000:]\n",
                "X_test = X_test / 255.0\n",
                "\n",
                "print(f\"Träningsdata: {X_train.shape[0]} bilder\")\n",
                "print(f\"Valideringsdata: {X_valid.shape[0]} bilder\")\n",
                "print(f\"Testdata: {X_test.shape[0]} bilder\")"
            ]
        },
        {
            "cell_type": "markdown",
            "id": "eda_section",
            "metadata": {},
            "source": [
                "# Visualisera Data\n",
                "\n",
                "Visar några exempelbilder och klassfördelningen för att förstå datan."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "id": "visualize_examples",
            "metadata": {},
            "outputs": [],
            "source": [
                "# Visualisera några exempelbilder\n",
                "fig, axes = plt.subplots(2, 5, figsize=(12, 5))\n",
                "fig.suptitle('Exempelbilder från MNIST-datasetet', fontsize=16)\n",
                "for i in range(10):\n",
                "    row = i // 5\n",
                "    col = i % 5\n",
                "    axes[row, col].imshow(X_train[i], cmap='gray')\n",
                "    axes[row, col].set_title(f'Etikett: {y_train[i]}', fontsize=12)\n",
                "    axes[row, col].axis('off')\n",
                "plt.tight_layout()\n",
                "plt.show()"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "id": "visualize_distribution",
            "metadata": {},
            "outputs": [],
            "source": [
                "# Visa fördelning av klasser\n",
                "unique, counts = np.unique(y_train, return_counts=True)\n",
                "plt.figure(figsize=(10, 5))\n",
                "plt.bar(unique, counts)\n",
                "plt.xlabel('Siffra')\n",
                "plt.ylabel('Antal bilder')\n",
                "plt.title('Fördelning av klasser i träningsdatan')\n",
                "plt.xticks(unique)\n",
                "plt.grid(axis='y', alpha=0.3)\n",
                "plt.show()"
            ]
        },
        {
            "cell_type": "markdown",
            "id": "model1_section",
            "metadata": {},
            "source": [
                "# Del A: Bygg och Träna ANN-modell\n",
                "\n",
                "Bygger en neural nätverksmodell med manuellt valda hyperparametrar för att klassificera MNIST-siffror."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "id": "build_model1",
            "metadata": {},
            "outputs": [],
            "source": [
                "# Bygg modell med Sequential API\n",
                "model = Sequential([\n",
                "    Input(shape=[28, 28]),  # Definierar input-shape (28x28 bilder)\n",
                "    Flatten(),  # Konverterar 28x28 bilder till 784 features\n",
                "    Dense(128, activation='relu'),  # Hidden layer 1: 128 neuroner med ReLU\n",
                "    Dropout(0.3),  # Regularisering: stänger av 30% av neuronerna slumpmässigt\n",
                "    Dense(64, activation='relu'),  # Hidden layer 2: 64 neuroner\n",
                "    Dropout(0.3),  # Ytterligare regularisering\n",
                "    Dense(10, activation='softmax')  # Output layer: 10 klasser (siffror 0-9)\n",
                "])\n",
                "\n",
                "model.summary()"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "id": "compile_model1",
            "metadata": {},
            "outputs": [],
            "source": [
                "# Kompilera modellen\n",
                "# Adam: adaptiv optimizer som justerar learning rate automatiskt\n",
                "# SparseCategoricalCrossentropy: används när labels är heltal (0-9)\n",
                "model.compile(\n",
                "    optimizer='adam',\n",
                "    loss='sparse_categorical_crossentropy',\n",
                "    metrics=['accuracy']\n",
                ")"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "id": "train_model1",
            "metadata": {},
            "outputs": [],
            "source": [
                "# Early stopping: stoppar träningen om val_loss inte förbättras\n",
                "early_stopping = EarlyStopping(\n",
                "    monitor='val_loss',\n",
                "    patience=5,  # Vänta 5 epoker innan stopp\n",
                "    restore_best_weights=True  # Återställ till bästa vikter\n",
                ")\n",
                "\n",
                "# Träna modellen\n",
                "history = model.fit(\n",
                "    X_train, y_train,\n",
                "    epochs=30,\n",
                "    batch_size=128,\n",
                "    validation_data=(X_valid, y_valid),\n",
                "    callbacks=[early_stopping],\n",
                "    verbose=1\n",
                ")"
            ]
        },
        {
            "cell_type": "markdown",
            "id": "evaluate_model1_section",
            "metadata": {},
            "source": [
                "# Utvärdera Manuellt Tränad Modell\n",
                "\n",
                "Utvärderar modellen på testdatan för att se hur bra den presterar."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "id": "evaluate_model1",
            "metadata": {},
            "outputs": [],
            "source": [
                "# Utvärdera på testdata\n",
                "test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)\n",
                "\n",
                "print(\"=\" * 60)\n",
                "print(\"RESULTAT - Manuellt Tränad Modell\")\n",
                "print(\"=\" * 60)\n",
                "print(f\"Test Loss: {test_loss:.4f}\")\n",
                "print(f\"Test Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)\")\n",
                "print(\"=\" * 60)"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "id": "plot_history1",
            "metadata": {},
            "outputs": [],
            "source": [
                "# Visualisera träningshistorik\n",
                "plt.figure(figsize=(12, 4))\n",
                "\n",
                "plt.subplot(1, 2, 1)\n",
                "plt.plot(history.history['loss'], 'r', label='Training Loss')\n",
                "plt.plot(history.history['val_loss'], 'b', label='Validation Loss')\n",
                "plt.xlabel('Epochs')\n",
                "plt.ylabel('Loss')\n",
                "plt.title('Model Loss')\n",
                "plt.legend()\n",
                "plt.grid(True, alpha=0.3)\n",
                "\n",
                "plt.subplot(1, 2, 2)\n",
                "plt.plot(history.history['accuracy'], 'r', label='Training Accuracy')\n",
                "plt.plot(history.history['val_accuracy'], 'b', label='Validation Accuracy')\n",
                "plt.xlabel('Epochs')\n",
                "plt.ylabel('Accuracy')\n",
                "plt.title('Model Accuracy')\n",
                "plt.legend()\n",
                "plt.grid(True, alpha=0.3)\n",
                "\n",
                "plt.tight_layout()\n",
                "plt.show()"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "id": "confusion_matrix1",
            "metadata": {},
            "outputs": [],
            "source": [
                "# Gör prediktioner och visa classification report\n",
                "y_pred_proba = model.predict(X_test, verbose=0)\n",
                "y_pred = np.argmax(y_pred_proba, axis=1)  # Konvertera sannolikheter till klasser\n",
                "\n",
                "print(\"Classification Report:\")\n",
                "print(classification_report(y_test, y_pred))"
            ]
        },
        {
            "cell_type": "markdown",
            "id": "kerastuner_section",
            "metadata": {},
            "source": [
                "# Del B: Optimera Hyperparametrar med KerasTuner\n",
                "\n",
                "Använder KerasTuner för att automatiskt hitta bästa hyperparametrarna. Vi optimerar antal neuroner i hidden layers, dropout rate och learning rate."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "id": "define_model_builder",
            "metadata": {},
            "outputs": [],
            "source": [
                "# Definiera funktion som bygger modell med hyperparametrar från tuner\n",
                "def build_model(hp):\n",
                "    model = Sequential([\n",
                "        Input(shape=[28, 28]),  # Definierar input-shape (28x28 bilder)\n",
                "        Flatten(),  # Konverterar 28x28 bilder till 784 features\n",
                "        # Antal neuroner i första hidden layer väljs av tuner (64-256)\n",
                "        Dense(units=hp.Int('units_1', min_value=64, max_value=256, step=64), activation='relu'),\n",
                "        # Dropout rate väljs av tuner (0.2-0.4)\n",
                "        Dropout(rate=hp.Float('dropout_1', min_value=0.2, max_value=0.4, step=0.1)),\n",
                "        # Antal neuroner i andra hidden layer (32-128)\n",
                "        Dense(units=hp.Int('units_2', min_value=32, max_value=128, step=32), activation='relu'),\n",
                "        Dropout(rate=hp.Float('dropout_2', min_value=0.2, max_value=0.4, step=0.1)),\n",
                "        Dense(10, activation='softmax')  # Output layer: 10 klasser\n",
                "    ])\n",
                "    \n",
                "    # Learning rate väljs av tuner\n",
                "    learning_rate = hp.Choice('learning_rate', values=[1e-3, 1e-4, 1e-5])\n",
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
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "id": "create_tuner",
            "metadata": {},
            "outputs": [],
            "source": [
                "# Skapa RandomSearch tuner\n",
                "# Använd tempfile för att undvika problem med svenska tecken i sökvägar\n",
                "tuner_dir = os.path.join(tempfile.gettempdir(), 'keras_tuner_mnist')\n",
                "os.makedirs(tuner_dir, exist_ok=True)\n",
                "\n",
                "max_trials = 10  # Antal modellkonfigurationer att testa\n",
                "\n",
                "tuner = kt.RandomSearch(\n",
                "    build_model,\n",
                "    objective='val_accuracy',  # Maximera validation accuracy\n",
                "    max_trials=max_trials,\n",
                "    executions_per_trial=1,\n",
                "    directory=tuner_dir,\n",
                "    project_name='mnist_ann_optimization',\n",
                "    overwrite=True\n",
                ")\n",
                "\n",
                "print(f\"KerasTuner skapad! Kommer testa {max_trials} olika konfigurationer.\")"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "id": "run_tuner",
            "metadata": {},
            "outputs": [],
            "source": [
                "# Kör hyperparametersökningen\n",
                "print(\"Börjar söka efter bästa hyperparametrar...\")\n",
                "print(\"Detta kan ta en stund beroende på din hårdvara.\\n\")\n",
                "\n",
                "tuner.search(\n",
                "    X_train, y_train,\n",
                "    epochs=20,  # Färre epoker per modell för snabbare sökning\n",
                "    batch_size=128,\n",
                "    validation_data=(X_valid, y_valid),\n",
                "    callbacks=[EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)],\n",
                "    verbose=1\n",
                ")\n",
                "\n",
                "print(\"\\nHyperparametersökning klar!\")"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "id": "get_best_model",
            "metadata": {},
            "outputs": [],
            "source": [
                "# Hämta bästa modellen och dess hyperparametrar\n",
                "best_model = tuner.get_best_models(num_models=1)[0]\n",
                "best_hyperparameters = tuner.get_best_hyperparameters(num_trials=1)[0]\n",
                "\n",
                "print(\"=\" * 60)\n",
                "print(\"BÄSTA HYPERPARAMETRARNA\")\n",
                "print(\"=\" * 60)\n",
                "print(f\"Units i första hidden layer: {best_hyperparameters.get('units_1')}\")\n",
                "print(f\"Units i andra hidden layer: {best_hyperparameters.get('units_2')}\")\n",
                "print(f\"Dropout rate 1: {best_hyperparameters.get('dropout_1')}\")\n",
                "print(f\"Dropout rate 2: {best_hyperparameters.get('dropout_2')}\")\n",
                "print(f\"Learning rate: {best_hyperparameters.get('learning_rate')}\")\n",
                "print(\"=\" * 60)"
            ]
        },
        {
            "cell_type": "markdown",
            "id": "retrain_best_section",
            "metadata": {},
            "source": [
                "# Träna Bästa Modellen på Hela Träningsdatan\n",
                "\n",
                "Tränar modellen med bästa hyperparametrarna på hela träningsdatan (inklusive valideringsdata) för bästa resultat."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "id": "retrain_best",
            "metadata": {},
            "outputs": [],
            "source": [
                "# Kombinera träning och valideringsdata\n",
                "X_train_full_combined = np.concatenate([X_train, X_valid], axis=0)\n",
                "y_train_full_combined = np.concatenate([y_train, y_valid], axis=0)\n",
                "\n",
                "# Bygg modell med bästa hyperparametrarna\n",
                "model_optimized = build_model(best_hyperparameters)\n",
                "\n",
                "# Träna på hela träningsdatan\n",
                "history_optimized = model_optimized.fit(\n",
                "    X_train_full_combined, y_train_full_combined,\n",
                "    epochs=30,\n",
                "    batch_size=128,\n",
                "    validation_split=0.1,  # Använd 10% för validering\n",
                "    callbacks=[EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)],\n",
                "    verbose=1\n",
                ")\n",
                "\n",
                "print(\"\\nTräning klar!\")"
            ]
        },
        {
            "cell_type": "markdown",
            "id": "evaluate_optimized_section",
            "metadata": {},
            "source": [
                "# Utvärdera Optimerad Modell\n",
                "\n",
                "Utvärderar den optimerade modellen på testdatan och jämför med den manuellt tränade modellen."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "id": "evaluate_optimized",
            "metadata": {},
            "outputs": [],
            "source": [
                "# Utvärdera optimerad modell\n",
                "test_loss_opt, test_accuracy_opt = model_optimized.evaluate(X_test, y_test, verbose=0)\n",
                "\n",
                "print(\"=\" * 60)\n",
                "print(\"RESULTAT - Optimerad Modell (KerasTuner)\")\n",
                "print(\"=\" * 60)\n",
                "print(f\"Test Loss: {test_loss_opt:.4f}\")\n",
                "print(f\"Test Accuracy: {test_accuracy_opt:.4f} ({test_accuracy_opt*100:.2f}%)\")\n",
                "print(\"=\" * 60)"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "id": "compare_models",
            "metadata": {},
            "outputs": [],
            "source": [
                "# Jämför de två modellerna\n",
                "print(\"=\" * 60)\n",
                "print(\"JÄMFÖRELSE MELLAN MODELLERNA\")\n",
                "print(\"=\" * 60)\n",
                "print(f\"\\nManuellt Tränad Modell:\")\n",
                "print(f\"  Test Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)\")\n",
                "print(f\"  Test Loss: {test_loss:.4f}\")\n",
                "print(f\"\\nOptimerad Modell (KerasTuner):\")\n",
                "print(f\"  Test Accuracy: {test_accuracy_opt:.4f} ({test_accuracy_opt*100:.2f}%)\")\n",
                "print(f\"  Test Loss: {test_loss_opt:.4f}\")\n",
                "\n",
                "improvement = test_accuracy_opt - test_accuracy\n",
                "print(f\"\\nFörbättring: {improvement:.4f} ({improvement*100:.2f} procentenheter)\")\n",
                "\n",
                "if improvement > 0:\n",
                "    print(\"\\nKerasTuner förbättrade modellen!\")\n",
                "elif improvement < 0:\n",
                "    print(\"\\nManuell modell presterade bättre.\")\n",
                "else:\n",
                "    print(\"\\nModellerna presterade lika bra.\")\n",
                "\n",
                "print(\"=\" * 60)"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "id": "classification_report_optimized",
            "metadata": {},
            "outputs": [],
            "source": [
                "# Classification report för optimerad modell\n",
                "y_pred_opt_proba = model_optimized.predict(X_test, verbose=0)\n",
                "y_pred_opt = np.argmax(y_pred_opt_proba, axis=1)\n",
                "\n",
                "print(\"Classification Report - Optimerad Modell:\")\n",
                "print(classification_report(y_test, y_pred_opt))"
            ]
        }
    ],
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "codemirror_mode": {
                "name": "ipython",
                "version": 3
            },
            "file_extension": ".py",
            "mimetype": "text/x-python",
            "name": "python",
            "nbconvert_exporter": "python",
            "pygments_lexer": "ipython3",
            "version": "3.11.9"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 5
}

# Spara notebooken
output_file = os.path.join(
    os.path.dirname(__file__),
    "Kapitel7_Uppgift10_MNIST_ANN_KerasTuner.ipynb"
)

with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(notebook, f, indent=1, ensure_ascii=False)

print(f"Notebook skapad: {output_file}")
print("\nNotebooken innehåller:")
print("- Komplett lösning för Koduppgift 10, Kapitel 7")
print("- ANN-modell på MNIST (Del A)")
print("- Hyperparameteroptimering med KerasTuner (Del B)")
print("- Jämförelse mellan modellerna")
print("\nKör skriptet för att generera notebooken!")
