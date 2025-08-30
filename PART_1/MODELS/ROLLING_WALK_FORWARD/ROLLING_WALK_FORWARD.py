import numpy as np
import pandas as pd
import tensorflow as tf
import os
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

# === 2. Modèle LSTM ===
def build_final_model(window_size=24, input_dim=23):
    model = Sequential()
    model.add(Input(shape=(window_size, input_dim)))
    model.add(LSTM(units=96, return_sequences=True, activation='tanh'))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    model.add(LSTM(units=32, activation='tanh'))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    model.add(Dense(units=16, activation='relu'))
    model.add(Dense(3, activation='softmax'))

    model.compile(optimizer=Adam(learning_rate=0.001),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

# === 3. Callback pour suivre le learning rate ===
class LearningRateLogger(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        lr = tf.keras.backend.get_value(self.model.optimizer.learning_rate)
        print(f"Learning rate at epoch {epoch + 1}: {lr:.6f}")

# === 4. Compilation et entraînement ===
def compile_and_train(model, X_train, y_train, X_val, y_val, batch_size=16, patience=20, epochs=200):
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=0.00001, verbose=1),
        LearningRateLogger()
    ]

    # Calcul dynamique des poids des classes
    class_weights = compute_class_weight('balanced', classes=np.array([0, 1, 2]), y=np.argmax(y_train, axis=1))
    class_weights = dict(enumerate(class_weights))
    print(f"Poids des classes pour cette fenêtre : {class_weights}")

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        batch_size=batch_size,
        epochs=epochs,
        callbacks=callbacks,
        shuffle=False,  # PAS de shuffle (ordre temporel)
        class_weight=class_weights
    )

    return model, history

# === 5. Matrice de confusion ===
def plot_confusion_matrix(y_true, y_pred, output_dir="results", title="window", class_names=None):
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

# === 6. Calcul des métriques ===
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

# === 7. Rolling Walk-Forward ===
def rolling_walk_forward(file_path, window_size=24, train_years=3, val_months=6, test_months=3, step_months=1, output_dir="results"):
    X, y = load_and_prepare_data(file_path, window_size=window_size)
    total_size = len(X)

    # Calcul des tailles des fenêtres (en observations)
    obs_per_day = 6  # 24 heures ÷ 4 heures
    obs_per_year = 365 * obs_per_day  # ≈ 2190
    train_size = int(train_years * obs_per_year)  # 3 ans ≈ 6570 observations
    val_size = int(val_months * obs_per_day * 30.42)  # 6 mois ≈ 1095 observations
    test_size = int(test_months * obs_per_day * 30.42)  # 3 mois ≈ 540 observations
    step_size = int(step_months * obs_per_day * 30.42)  # 1 mois ≈ 180 observations

    results = []
    os.makedirs(output_dir, exist_ok=True)

    # Boucle sur les fenêtres glissantes
    for start in range(0, total_size - (train_size + val_size + test_size), step_size):
        # Indices des fenêtres
        train_end = start + train_size
        val_end = train_end + val_size
        test_end = val_end + test_size

        # Séparation des données
        X_train, y_train = X[start:train_end], y[start:train_end]
        X_val, y_val = X[train_end:val_end], y[train_end:val_end]
        X_test, y_test = X[val_end:test_end], y[val_end:test_end]

        print(f"\nFenêtre {start//step_size + 1}: Entraînement [{start}:{train_end}], Validation [{train_end}:{val_end}], Test [{val_end}:{test_end}]")

        # Construire et entraîner le modèle
        model = build_final_model(window_size=window_size, input_dim=X.shape[2])
        model, history = compile_and_train(model, X_train, y_train, X_val, y_val)

        # Prédictions
        y_val_pred = model.predict(X_val)
        y_test_pred = model.predict(X_test)

        # Sauvegarde des prédictions
        window_id = f"window_{start//step_size + 1}"
        preds_df = pd.DataFrame(np.argmax(y_test_pred, axis=1), columns=["Predicted_Class"])
        preds_df.to_csv(os.path.join(output_dir, f"test_predictions_{window_id}.csv"), index=False)

        # Sauvegarde du modèle
        model.save(os.path.join(output_dir, f"lstm_model_{window_id}.keras"))

        # Sauvegarde des courbes d'entraînement
        plt.figure()
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title(f'Training and Validation Loss - {window_id}')
        plt.legend()
        plt.savefig(os.path.join(output_dir, f"loss_plot_{window_id}.png"))
        plt.close()

        plt.figure()
        plt.plot(history.history['accuracy'], label='Training Accuracy')
        plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
        plt.title(f'Training and Validation Accuracy - {window_id}')
        plt.legend()
        plt.savefig(os.path.join(output_dir, f"accuracy_plot_{window_id}.png"))
        plt.close()

        # Sauvegarde de l'historique (loss, val_loss, accuracy, val_accuracy)
        history_df = pd.DataFrame(history.history)
        history_df.to_csv(os.path.join(output_dir, f"training_history_{window_id}.csv"), index=False)
        print(f"✔️ Historique d'entraînement sauvegardé dans : {os.path.join(output_dir, f'training_history_{window_id}.csv')}")

        # Rapports de classification
        val_report = print_classification_report(y_val, y_val_pred, title=f"Validation Set - {window_id}", output_dir=output_dir)
        test_report = print_classification_report(y_test, y_test_pred, title=f"Test Set - {window_id}", output_dir=output_dir)

        # Matrice de confusion
        plot_confusion_matrix(y_test, y_test_pred, output_dir=output_dir, title=window_id)

        # Sauvegarde des résultats
        results.append({
            'window_id': window_id,
            'train_start': start,
            'test_end': test_end,
            'val_report': val_report,
            'test_report': test_report
        })

    # Résumé des performances
    results_df = pd.DataFrame([{
        'Window': r['window_id'],
        'F1-Score': r['test_report']['weighted avg']['f1-score'],
        'Precision': r['test_report']['weighted avg']['precision'],
        'Recall': r['test_report']['weighted avg']['recall'],
        'Buy_Precision': r['test_report']['Buy']['precision'],
        'Buy_Recall': r['test_report']['Buy']['recall'],
        'Buy_F1-Score': r['test_report']['Buy']['f1-score'],
        'Buy_Support': r['test_report']['Buy']['support'],
        'Hold_Precision': r['test_report']['Hold']['precision'],
        'Hold_Recall': r['test_report']['Hold']['recall'],
        'Hold_F1-Score': r['test_report']['Hold']['f1-score'],
        'Hold_Support': r['test_report']['Hold']['support'],
        'Sell_Precision': r['test_report']['Sell']['precision'],
        'Sell_Recall': r['test_report']['Sell']['recall'],
        'Sell_F1-Score': r['test_report']['Sell']['f1-score'],
        'Sell_Support': r['test_report']['Sell']['support']
    } for r in results])
    results_df.to_csv(os.path.join(output_dir, "walk_forward_results.csv"), index=False)
    print(f"✔️ Résultats agrégés sauvegardés dans : {os.path.join(output_dir, 'walk_forward_results.csv')}")

    # Afficher la moyenne des métriques
    avg_f1 = np.mean([r['test_report']['weighted avg']['f1-score'] for r in results])
    print(f"\n📊 Moyenne du F1-score sur toutes les fenêtres : {avg_f1:.4f}")

    # Visualisation des F1-scores par fenêtre
    plt.figure(figsize=(10, 6))
    plt.plot(results_df['Window'], results_df['F1-Score'], marker='o', label='F1-Score')
    plt.title('F1-Score par fenêtre')
    plt.xlabel('Fenêtre')
    plt.ylabel('F1-Score')
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'f1_score_windows.png'))
    plt.close()

    return results

# Exécution
if __name__ == "__main__":
    rolling_walk_forward("/content/data/btc_4H_onehot_thresh_0.47.csv")