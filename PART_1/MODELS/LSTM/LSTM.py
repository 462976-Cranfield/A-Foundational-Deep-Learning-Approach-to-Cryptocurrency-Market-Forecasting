import numpy as np
import pandas as pd
import tensorflow as tf
import os
import keras_tuner as kt
import json
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, Callback
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt

# === 1. Données d'entrée : lecture et préparation ===
def load_and_prepare_data(file_path, window_size=24):
    df = pd.read_csv(file_path)
    df.drop(columns=['date'], inplace=True, errors='ignore')
    df.drop(columns=['target_class'], inplace=True, errors='ignore')

    X, y = [], []
    for i in range(window_size, len(df)):
        X.append(df.iloc[i-window_size:i, :-3].values)  # Features (excluant Buy, Hold, Sell)
        y.append(df.iloc[i, -3:].values)  # Labels one-hot [Buy, Hold, Sell]

    X = np.array(X)
    y = np.array(y)
    return X, y

# === 2. Modèle LSTM pour Keras Tuner ===
def build_lstm_model(hp, window_size=24, input_dim=23):
    model = Sequential()
    model.add(Input(shape=(window_size, input_dim)))
    model.add(LSTM(units=hp.Int('units_1', 32, 128, step=32), return_sequences=True, activation='tanh'))
    model.add(BatchNormalization())
    model.add(Dropout(hp.Float('dropout_1', 0.1, 0.5, step=0.1)))
    model.add(LSTM(units=hp.Int('units_2', 16, 64, step=16), activation='tanh'))
    model.add(BatchNormalization())
    model.add(Dropout(hp.Float('dropout_2', 0.1, 0.5, step=0.1)))
    model.add(Dense(units=hp.Int('dense_units', 8, 32, step=8), activation='relu'))
    model.add(Dense(3, activation='softmax'))

    model.compile(optimizer=Adam(hp.Float('learning_rate', 1e-4, 1e-2, sampling='log')),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

# === 3. Modèle final avec hyperparamètres ===
def build_final_model(window_size=24, input_dim=23, hp_values=None):
    model = Sequential()
    model.add(Input(shape=(window_size, input_dim)))
    units_1 = hp_values.get('units_1', 96) if hp_values else 96
    dropout_1 = hp_values.get('dropout_1', 0.3) if hp_values else 0.3
    units_2 = hp_values.get('units_2', 32) if hp_values else 32
    dropout_2 = hp_values.get('dropout_2', 0.3) if hp_values else 0.3
    dense_units = hp_values.get('dense_units', 16) if hp_values else 16
    learning_rate = hp_values.get('learning_rate', 0.001) if hp_values else 0.001

    model.add(LSTM(units=units_1, return_sequences=True, activation='tanh'))
    model.add(BatchNormalization())
    model.add(Dropout(dropout_1))
    model.add(LSTM(units=units_2, activation='tanh'))
    model.add(BatchNormalization())
    model.add(Dropout(dropout_2))
    model.add(Dense(units=dense_units, activation='relu'))
    model.add(Dense(3, activation='softmax'))

    model.compile(optimizer=Adam(learning_rate=learning_rate),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

# === 4. Callback pour suivre le learning rate ===
class LearningRateLogger(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        lr = tf.keras.backend.get_value(self.model.optimizer.learning_rate)
        print(f"Learning rate at epoch {epoch + 1}: {lr:.6f}")

# === 5. Compilation et entraînement ===
def compile_and_train(model, X_train, y_train, X_val, y_val, batch_size=16, patience=20, epochs=200, output_dir="results"):
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=0.00001, verbose=1),
        LearningRateLogger(),
        tf.keras.callbacks.ModelCheckpoint(
            os.path.join(output_dir, "best_lstm_model.keras"),
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        )
    ]

    class_weights = compute_class_weight('balanced', classes=np.array([0, 1, 2]), y=np.argmax(y_train, axis=1))
    class_weights = dict(enumerate(class_weights))
    print(f"Poids des classes : {class_weights}")

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        batch_size=batch_size,
        epochs=epochs,
        callbacks=callbacks,
        shuffle=False,
        class_weight=class_weights
    )

    return model, history

# === 6. Matrice de confusion ===
def plot_confusion_matrix(y_true, y_pred, output_dir="results", title="model", class_names=None):
    if class_names is None:
        class_names = ['Buy', 'Hold', 'Sell']

    y_true_labels = np.argmax(y_true, axis=1)
    y_pred_labels = np.argmax(y_pred, axis=1)

    cm = confusion_matrix(y_true_labels, y_pred_labels)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
    ax.figure.colorbar(im, ax=ax)

    ax.set(
        xticks=np.arange(len(class_names)),
        yticks=np.arange(len(class_names)),
        xticklabels=class_names,
        yticklabels=class_names,
        title=f"Matrice de confusion - {title}",
        ylabel='True label',
        xlabel='Predicted label'
    )

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")

    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    fig_path = os.path.join(output_dir, f"confusion_matrix_{title}.png")
    plt.savefig(fig_path)
    plt.close()
    print(f"✔️ Matrice de confusion sauvegardée dans : {fig_path}")

# === 7. Calcul des métriques ===
def print_classification_report(y_true, y_pred, title="Classification Report", output_dir="results"):
    print(f"\n📊 {title}")
    report = classification_report(
        np.argmax(y_true, axis=1),
        np.argmax(y_pred, axis=1),
        target_names=["Buy", "Hold", "Sell"],
        digits=4,
        output_dict=True
    )
    # Sauvegarde du rapport dans un fichier .txt
    os.makedirs(output_dir, exist_ok=True)
    report_path = os.path.join(output_dir, f"{title.replace(' ', '_')}_report.txt")
    with open(report_path, 'w') as f:
        f.write(f"{title}\n")
        f.write(classification_report(
            np.argmax(y_true, axis=1),
            np.argmax(y_pred, axis=1),
            target_names=["Buy", "Hold", "Sell"],
            digits=4
        ))
    print(f"✔️ Rapport de classification sauvegardé dans : {report_path}")
    return report

# === 8. Fonction principale ===
def main(file_path, window_size=24, output_dir="results"):
    # Charger et préparer les données
    X, y = load_and_prepare_data(file_path, window_size=window_size)
    total_size = len(X)

    # Division statique : 70% entraînement, 15% validation, 15% test
    train_idx = int(total_size * 0.7)
    val_idx = int(total_size * 0.85)

    X_train, y_train = X[:train_idx], y[:train_idx]
    X_val, y_val = X[train_idx:val_idx], y[train_idx:val_idx]
    X_test, y_test = X[val_idx:], y[val_idx:]

    print(f"Données : Entraînement {X_train.shape}, Validation {X_val.shape}, Test {X_test.shape}")

    # Créer le dossier output_dir
    os.makedirs(output_dir, exist_ok=True)

    # Étape Keras Tuner pour optimiser les hyperparamètres
    tuner = kt.RandomSearch(
        lambda hp: build_lstm_model(hp, window_size=window_size, input_dim=X.shape[2]),
        objective='val_accuracy',
        max_trials=10,
        directory='tuner_dir',
        project_name='lstm_tuning',
        overwrite=True
    )
    tuner.search(X_train, y_train, epochs=50, validation_data=(X_val, y_val),
                 callbacks=[EarlyStopping(patience=10)])

    # Récupérer et sauvegarder les meilleurs hyperparamètres
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    hp_values = best_hps.values
    hp_path = os.path.join(output_dir, "best_hyperparameters.json")
    with open(hp_path, 'w') as f:
        json.dump(hp_values, f, indent=4)
    print(f"✔️ Meilleurs hyperparamètres sauvegardés dans : {hp_path}")
    print(f"Meilleurs hyperparamètres : {hp_values}")

    # Construire et entraîner le modèle final
    model = build_final_model(window_size=window_size, input_dim=X.shape[2], hp_values=hp_values)
    model, history = compile_and_train(model, X_train, y_train, X_val, y_val, output_dir=output_dir)

    # Charger le meilleur modèle sauvegardé
    best_model_path = os.path.join(output_dir, "best_lstm_model.keras")
    if os.path.exists(best_model_path):
        model = tf.keras.models.load_model(best_model_path)
        print(f"✔️ Meilleur modèle chargé depuis : {best_model_path}")

    # Prédictions
    y_val_pred = model.predict(X_val)
    y_test_pred = model.predict(X_test)

    # Sauvegarde des prédictions
    preds_df = pd.DataFrame(np.argmax(y_test_pred, axis=1), columns=["Predicted_Class"])
    preds_df.to_csv(os.path.join(output_dir, "test_predictions.csv"), index=False)
    print(f"✔️ Prédictions sauvegardées dans : {os.path.join(output_dir, 'test_predictions.csv')}")

    # Sauvegarde du modèle final (optionnel, car best_lstm_model.keras est déjà sauvegardé)
    model.save(os.path.join(output_dir, "lstm_model.keras"))
    print(f"✔️ Modèle sauvegardé dans : {os.path.join(output_dir, 'lstm_model.keras')}")

    # Sauvegarde des courbes d'entraînement
    plt.figure()
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.savefig(os.path.join(output_dir, "loss_plot.png"))
    plt.close()

    plt.figure()
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()
    plt.savefig(os.path.join(output_dir, "accuracy_plot.png"))
    plt.close()

    # Sauvegarde de l'historique
    history_df = pd.DataFrame(history.history)
    history_df.to_csv(os.path.join(output_dir, "training_history.csv"), index=False)
    print(f"✔️ Historique d'entraînement sauvegardé dans : {os.path.join(output_dir, 'training_history.csv')}")

    # Rapports de classification
    val_report = print_classification_report(y_val, y_val_pred, title="Validation Set", output_dir=output_dir)
    test_report = print_classification_report(y_test, y_test_pred, title="Test Set", output_dir=output_dir)

    # Matrice de confusion
    plot_confusion_matrix(y_test, y_test_pred, output_dir=output_dir, title="test_set")

    # Sauvegarde des résultats
    results_df = pd.DataFrame([{
        'F1-Score': test_report['weighted avg']['f1-score'],
        'Precision': test_report['weighted avg']['precision'],
        'Recall': test_report['weighted avg']['recall'],
        'Buy_Precision': test_report['Buy']['precision'],
        'Buy_Recall': test_report['Buy']['recall'],
        'Buy_F1-Score': test_report['Buy']['f1-score'],
        'Buy_Support': test_report['Buy']['support'],
        'Hold_Precision': test_report['Hold']['precision'],
        'Hold_Recall': test_report['Hold']['recall'],
        'Hold_F1-Score': test_report['Hold']['f1-score'],
        'Hold_Support': test_report['Hold']['support'],
        'Sell_Precision': test_report['Sell']['precision'],
        'Sell_Recall': test_report['Sell']['recall'],
        'Sell_F1-Score': test_report['Sell']['f1-score'],
        'Sell_Support': test_report['Sell']['support']
    }])
    results_df.to_csv(os.path.join(output_dir, "results.csv"), index=False)
    print(f"✔️ Résultats sauvegardés dans : {os.path.join(output_dir, 'results.csv')}")

# Exécution
if __name__ == "__main__":
    main("/content/data/btc_4H_onehot_thresh_0.47.csv")
