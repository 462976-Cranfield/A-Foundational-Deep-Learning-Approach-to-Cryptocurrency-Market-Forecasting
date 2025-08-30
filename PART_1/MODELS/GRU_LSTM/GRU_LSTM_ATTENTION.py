import numpy as np
import pandas as pd
import tensorflow as tf
import os
import keras_tuner as kt
import json
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout, Input, BatchNormalization, Attention, Multiply
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, Callback
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt

# === 1. Données d'entrée : lecture et préparation ===
def load_and_prepare_data(file_path, window_size=24, key_features=None):
    print("Chargement et préparation des données...")
    df = pd.read_csv(file_path)
    df.drop(columns=['date'], inplace=True, errors='ignore')

    # Colonnes de caractéristiques
    feature_cols = [
        'open_norm', 'high_norm', 'low_norm', 'close_norm', 'volume_norm',
        'RSI_norm', 'MACD_norm', 'MACD_signal_norm', 'rsi_oversold', 'rsi_overbought',
        'macd_cross', 'strong_macd_bull', 'above_cloud', 'below_cloud', 'price_in_kumo',
        'kumo_thick', 'thick_kumo_when_price_inside', 'tenkan_kijun_cross_up',
        'tenkan_kijun_cross_down', 'bullish_cross_above_cloud', 'price_above_kijun',
        'price_above_tenkan', 'chikou_above_price'
    ]

    # Filtrer les colonnes existantes
    feature_cols = [col for col in feature_cols if col in df.columns]
    print(f"Colonnes utilisées ({len(feature_cols)}): {feature_cols}")

    # Sélectionner les caractéristiques clés si spécifiées
    if key_features:
        key_features = [col for col in key_features if col in feature_cols]
        print(f"Caractéristiques clés sélectionnées : {key_features}")
        feature_cols = key_features

    # Vérifier la présence de target_class
    if 'target_class' not in df.columns:
        raise ValueError("Colonne 'target_class' non trouvée.")

    # Convertir target_class en one-hot encoding
    label_map = {'Buy': 0, 'Hold': 1, 'Sell': 2}
    df['target_class'] = df['target_class'].map(label_map)
    y = tf.keras.utils.to_categorical(df['target_class'], num_classes=3)

    print("Distribution des classes target_class:")
    print(df['target_class'].value_counts())

    print("Statistiques des caractéristiques:")
    print(df[feature_cols].describe())

    # Préparer les séquences
    X = []
    for i in range(window_size, len(df)):
        X.append(df[feature_cols].iloc[i-window_size:i].values)
    X = np.array(X)
    y = y[window_size:]

    print(f"Données : X shape {X.shape}, y shape {y.shape}")
    return X, y, feature_cols

# === 2. Modèle GRU + LSTM + Attention avec Keras Tuner ===
def build_lstm_model(hp, model_type='lstm_attention', window_size=24, input_dim=7, key_feature_indices=None):
    inputs = Input(shape=(window_size, input_dim))

    # Masque pour prioriser les caractéristiques clés
    if key_feature_indices is not None:
        mask = np.ones(input_dim)
        for idx in key_feature_indices:
            mask[idx] = 2.0  # Poids plus élevé pour les caractéristiques clés
        mask = tf.constant(mask, dtype=tf.float32)
        mask = tf.expand_dims(tf.expand_dims(mask, 0), 0)  # (1, 1, input_dim)
        masked_inputs = Multiply()([inputs, mask])
    else:
        masked_inputs = inputs

    # GRU Layer
    gru_units = hp.Int('gru_units', min_value=32, max_value=256, step=32)
    x = GRU(units=gru_units, return_sequences=True)(masked_inputs)
    x = BatchNormalization()(x)
    x = Dropout(hp.Float('gru_dropout', 0.1, 0.5, step=0.1))(x)

    # LSTM Layer
    lstm_units = hp.Int('lstm_units', min_value=32, max_value=256, step=32)
    x = LSTM(units=lstm_units, return_sequences=True)(x)
    x = BatchNormalization()(x)
    x = Dropout(hp.Float('lstm_dropout', 0.1, 0.5, step=0.1))(x)

    # Attention Layer
    attention = Attention()([x, x])
    x = tf.keras.layers.GlobalAveragePooling1D()(attention)

    # Dense Layers
    dense_units = hp.Int('dense_units', min_value=16, max_value=128, step=16)
    x = Dense(units=dense_units, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(hp.Float('dense_dropout', 0.1, 0.5, step=0.1))(x)

    # Output Layer
    outputs = Dense(3, activation='softmax')(x)

    model = Model(inputs=inputs, outputs=outputs)

    # Learning rate optimization
    learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model

def build_final_model(window_size=24, input_dim=7, hp_values=None, key_feature_indices=None):
    inputs = Input(shape=(window_size, input_dim))

    # Masque pour prioriser les caractéristiques clés
    if key_feature_indices is not None:
        mask = np.ones(input_dim)
        for idx in key_feature_indices:
            mask[idx] = 2.0  # Poids plus élevé
        mask = tf.constant(mask, dtype=tf.float32)
        mask = tf.expand_dims(tf.expand_dims(mask, 0), 0)  # (1, 1, input_dim)
        masked_inputs = Multiply()([inputs, mask])
    else:
        masked_inputs = inputs

    x = GRU(units=hp_values['gru_units'], return_sequences=True)(masked_inputs)
    x = BatchNormalization()(x)
    x = Dropout(hp_values['gru_dropout'])(x)

    x = LSTM(units=hp_values['lstm_units'], return_sequences=True)(x)
    x = BatchNormalization()(x)
    x = Dropout(hp_values['lstm_dropout'])(x)

    attention = Attention()([x, x])
    x = tf.keras.layers.GlobalAveragePooling1D()(attention)

    x = Dense(units=hp_values['dense_units'], activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(hp_values['dense_dropout'])(x)

    outputs = Dense(3, activation='softmax')(x)

    model = Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=Adam(learning_rate=hp_values['learning_rate']),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

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
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=1e-5, verbose=1),
        LearningRateLogger(),
        tf.keras.callbacks.ModelCheckpoint(
            os.path.join(output_dir, "best_gru_lstm_model.keras"),
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
    print("Début du traitement...")

    # Caractéristiques clés à prioriser
    key_features = ['macd_cross', 'tenkan_kijun_cross_up', 'tenkan_kijun_cross_down',
                    'open_norm', 'close_norm', 'volume_norm', 'bullish_cross_above_cloud']

    # Charger les données
    X, y, feature_cols = load_and_prepare_data(file_path, window_size=window_size, key_features=key_features)
    key_feature_indices = [feature_cols.index(f) for f in key_features if f in feature_cols]

    total_size = len(X)
    train_idx = int(total_size * 0.7)
    val_idx = int(total_size * 0.85)

    X_train, y_train = X[:train_idx], y[:train_idx]
    X_val, y_val = X[train_idx:val_idx], y[train_idx:val_idx]
    X_test, y_test = X[val_idx:], y[val_idx:]

    print(f"Données : Entraînement {X_train.shape}, Validation {X_val.shape}, Test {X_test.shape}")

    os.makedirs(output_dir, exist_ok=True)

    tuner = kt.RandomSearch(
        lambda hp: build_lstm_model(hp, model_type='lstm_attention', window_size=window_size,
                                   input_dim=len(feature_cols), key_feature_indices=key_feature_indices),
        objective='val_accuracy',
        max_trials=10,
        directory='tuner_dir',
        project_name='gru_lstm_tuning',
        overwrite=True
    )
    tuner.search(X_train, y_train, epochs=50, validation_data=(X_val, y_val),
                 callbacks=[EarlyStopping(patience=10)])

    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    hp_values = best_hps.values
    hp_path = os.path.join(output_dir, "best_hyperparameters.json")
    with open(hp_path, 'w') as f:
        json.dump(hp_values, f, indent=4)
    print(f"✔️ Meilleurs hyperparamètres sauvegardés dans : {hp_path}")
    print(f"Meilleurs hyperparamètres : {hp_values}")

    model = build_final_model(window_size=window_size, input_dim=len(feature_cols),
                             hp_values=hp_values, key_feature_indices=key_feature_indices)
    model, history = compile_and_train(model, X_train, y_train, X_val, y_val, output_dir=output_dir)

    best_model_path = os.path.join(output_dir, "best_gru_lstm_model.keras")
    if os.path.exists(best_model_path):
        model = tf.keras.models.load_model(best_model_path)
        print(f"✔️ Meilleur modèle chargé depuis : {best_model_path}")

    y_val_pred = model.predict(X_val)
    y_test_pred = model.predict(X_test)

    preds_df = pd.DataFrame(np.argmax(y_test_pred, axis=1), columns=["Predicted_Class"])
    preds_df['Predicted_Class'] = preds_df['Predicted_Class'].map({0: 'Buy', 1: 'Hold', 2: 'Sell'})
    preds_df.to_csv(os.path.join(output_dir, "test_predictions.csv"), index=False)
    print(f"✔️ Prédictions sauvegardées dans : {os.path.join(output_dir, 'test_predictions.csv')}")

    model.save(os.path.join(output_dir, "gru_lstm_model.keras"))
    print(f"✔️ Modèle sauvegardé dans : {os.path.join(output_dir, 'gru_lstm_model.keras')}")

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

    history_df = pd.DataFrame(history.history)
    history_df.to_csv(os.path.join(output_dir, "training_history.csv"), index=False)
    print(f"✔️ Historique d'entraînement sauvegardé dans : {os.path.join(output_dir, 'training_history.csv')}")

    val_report = print_classification_report(y_val, y_val_pred, title="Validation Set", output_dir=output_dir)
    test_report = print_classification_report(y_test, y_test_pred, title="Test Set", output_dir=output_dir)

    plot_confusion_matrix(y_test, y_test_pred, output_dir=output_dir, title="test_set")

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

if __name__ == "__main__":
    main("/content/data/btc_4H_onehot_thresh_0.47.csv", window_size=24)