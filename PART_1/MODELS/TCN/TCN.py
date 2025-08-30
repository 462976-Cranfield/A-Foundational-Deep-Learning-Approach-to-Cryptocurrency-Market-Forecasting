import numpy as np
import pandas as pd
import tensorflow as tf
import os
import keras_tuner as kt
import json
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, BatchNormalization, ReLU, Conv1D
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, Callback
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
from tensorflow.keras.layers import GlobalAveragePooling1D
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.callbacks import LearningRateScheduler


# === 1. Données d'entrée : lecture et préparation ===
def load_and_prepare_data(file_path, window_size=24):
    df = pd.read_csv(file_path)
    df.drop(columns=['date'], inplace=True, errors='ignore')
    df.drop(columns=['target_class'], inplace=True, errors='ignore')

    if len(df) < window_size:
        raise ValueError(f"Le DataFrame a moins de {window_size} lignes, insuffisant pour créer des séquences.")
    if df.shape[1] < 4:  # 3 colonnes pour les labels one-hot + au moins 1 feature
        raise ValueError("Le DataFrame doit contenir au moins 4 colonnes (features + 3 colonnes one-hot).")

    X, y = [], []
    for i in range(window_size, len(df)):
        X.append(df.iloc[i-window_size:i, :-3].values)  # Features (excluant Buy, Hold, Sell)
        y.append(df.iloc[i, -3:].values)  # Labels one-hot [Buy, Hold, Sell]

    X = np.array(X)
    y = np.array(y)
    if X.shape[0] != y.shape[0]:
        raise ValueError("Les dimensions de X et y ne correspondent pas.")

    return X, y

def lr_scheduler(epoch, lr):
    if epoch % 20 == 0 and epoch:  # Réduction toutes les 20 époques
        lr = lr * 0.9
        print(f"Réduction du learning rate à {lr:.6f} à l'époque {epoch}")
    return lr

def build_model(hp, model_type='tcn', window_size=24, input_dim=23):
    model = Sequential()
    model.add(Input(shape=(window_size, input_dim)))

    if model_type == 'tcn':
        # Hyperparamètres à tuner
        filters = hp.Int('filters', min_value=16, max_value=256, step=16)
        kernel_size = hp.Choice('kernel_size', values=[3, 5, 7])
        dilation_rates = [1, 2, 4, 8, 16]  # Profondeur TCN
        dropout_rate = hp.Float('dropout_rate', min_value=0.1, max_value=0.5, step=0.1)
        dense_units = hp.Int('dense_units', min_value=8, max_value=128, step=8)
        learning_rate = hp.Float('learning_rate', min_value=1e-4, max_value=1e-2, sampling='log')

        # Blocs TCN dilatés
        for i, dilation in enumerate(dilation_rates):
            model.add(Conv1D(filters=filters,
                             kernel_size=kernel_size,
                             dilation_rate=dilation,
                             padding='causal',
                             activation=None))
            model.add(BatchNormalization())
            model.add(ReLU())
            model.add(Dropout(dropout_rate))

        # Classification
        model.add(GlobalAveragePooling1D())
        model.add(Dense(dense_units, activation='relu'))
        model.add(Dense(3, activation='softmax'))

        # Compilation
        model.compile(optimizer=Adam(learning_rate=learning_rate),
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])
        return model

    else:
        raise ValueError("model_type must be 'tcn'")

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
        ModelCheckpoint(filepath=os.path.join(output_dir, "tcn_model.keras"),
                        monitor='val_loss', save_best_only=True, verbose=1)
    ]

    # Vérification des classes présentes dans y_train
    y_train_labels = np.argmax(y_train, axis=1)
    unique_classes = np.unique(y_train_labels)
    expected_classes = np.array([0, 1, 2])
    if not np.all(np.isin(expected_classes, unique_classes)):
        missing_classes = expected_classes[~np.isin(expected_classes, unique_classes)]
        print(f"⚠️ Attention : Les classes suivantes sont absentes dans y_train : {missing_classes}")

    # Calcul des poids des classes
    class_weights = compute_class_weight(class_weight='balanced', classes=unique_classes, y=y_train_labels)
    class_weights_dict = {cls: weight for cls, weight in zip(unique_classes, class_weights)}
    for cls in expected_classes:
        if cls not in class_weights_dict:
            class_weights_dict[cls] = 1.0
            print(f"Poids par défaut (1.0) attribué à la classe absente : {cls}")
    class_weights_dict = {cls: min(max(weight, 0.1), 10.0) for cls, weight in class_weights_dict.items()}
    print(f"Poids des classes : {class_weights_dict}")
    class_counts = {i: np.sum(y_train_labels == i) for i in expected_classes}
    print(f"Distribution des classes dans y_train : {class_counts}")

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        batch_size=batch_size,
        epochs=epochs,
        callbacks=callbacks,
        shuffle=False,
        class_weight=class_weights_dict
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

    os.makedirs(output_dir, exist_ok=True)
    os.environ['TF_CUDNN_DETERMINISTIC'] = '1'  # Pour GPU
    np.random.seed(42)
    tf.random.set_seed(42)
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

    # Étape Keras Tuner pour optimiser les hyperparamètres
    tuner = kt.RandomSearch(
        lambda hp: build_model(hp, model_type='tcn', window_size=window_size, input_dim=X.shape[2]),
        objective='val_accuracy',
        max_trials=10,
        directory='tuner_dir',
        project_name='tcn_tuning',  # Changé de 'lstm_tuning' à 'tcn_tuning' pour clarté
        overwrite=True
    )
    tuner.search(X_train, y_train, epochs=50, validation_data=(X_val, y_val),
                 callbacks=[EarlyStopping(patience=10)])

    # Récupérer et sauvegarder les meilleurs hyperparamètres
    best_hps = tuner.get_best_hyperparameters(num_trials=1)
    if not best_hps:
        raise ValueError("Aucun hyperparamètre optimal trouvé.")
    best_hps = best_hps[0]
    hp_values = best_hps.values
    hp_path = os.path.join(output_dir, "best_hyperparameters.json")
    with open(hp_path, 'w') as f:
        json.dump(hp_values, f, indent=4)
    print(f"✔️ Meilleurs hyperparamètres sauvegardés dans : {hp_path}")
    print(f"Meilleurs hyperparamètres : {hp_values}")

    # Construire le modèle final avec les meilleurs hyperparamètres
    model = build_model(best_hps, model_type='tcn', window_size=window_size, input_dim=X.shape[2])
    model, history = compile_and_train(model, X_train, y_train, X_val, y_val, output_dir=output_dir)

    # Sauvegarde de l'historique
    history_df = pd.DataFrame(history.history)
    history_df.to_csv(os.path.join(output_dir, "training_history.csv"), index=False)
    print(f"✔️ Historique d'entraînement sauvegardé dans : {os.path.join(output_dir, 'training_history.csv')}")

    # Prédictions
    y_val_pred = model.predict(X_val)
    y_test_pred = model.predict(X_test)

    # Sauvegarde des prédictions
    preds_df = pd.DataFrame(np.argmax(y_test_pred, axis=1), columns=["Predicted_Class"])
    preds_df.to_csv(os.path.join(output_dir, "test_predictions.csv"), index=False)
    print(f"✔️ Prédictions sauvegardées dans : {os.path.join(output_dir, 'test_predictions.csv')}")

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