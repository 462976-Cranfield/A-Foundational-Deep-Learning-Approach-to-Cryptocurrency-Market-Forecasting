import numpy as np
import pandas as pd
import tensorflow as tf
import os
import keras_tuner as kt
import json
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, Input, LayerNormalization, MultiHeadAttention, GRU, GlobalAveragePooling1D, BatchNormalization, Attention
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, Callback
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt

# === Positional Encoding Layer ===
class PositionalEncoding(tf.keras.layers.Layer):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.d_model = d_model
        pe = np.zeros((max_len, d_model))
        position = np.arange(0, max_len, dtype=np.float32)[:, np.newaxis]
        div_term = np.exp(np.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = np.sin(position * div_term)
        pe[:, 1::2] = np.cos(position * div_term)
        pe = pe[np.newaxis, :, :]
        self.pos_encoding = tf.constant(pe, dtype=tf.float32)

    def call(self, inputs):
        return inputs + self.pos_encoding[:, :tf.shape(inputs)[1], :]

# === 1. Données d'entrée : lecture et préparation ===
def load_and_prepare_data(file_path, window_size=24):
    df = pd.read_csv(file_path)
    df.drop(columns=['date'], inplace=True, errors='ignore')
    df.drop(columns=['target_class'], inplace=True, errors='ignore')

    X, y = [], []
    for i in range(window_size, len(df)):
        X.append(df.iloc[i-window_size:i, :-3].values)
        y.append(df.iloc[i, -3:].values)
    X = np.array(X)
    y = np.array(y)
    return X, y

# === 2. Modèle Transformer + GRU avec Keras Tuner ===
def build_transformer_gru_model(hp, window_size=24, input_dim=23):
    inputs = Input(shape=(window_size, input_dim))

    # Embedding Layer
    d_model = hp.Int('d_model', min_value=32, max_value=128, step=32)
    x = Dense(d_model, activation='relu')(inputs)  # Linear projection to d_model
    x = PositionalEncoding(d_model)(x)

    # Transformer Layers
    num_transformer_layers = hp.Int('num_transformer_layers', min_value=1, max_value=3, step=1)
    num_heads = hp.Int('num_heads', min_value=2, max_value=8, step=2)
    d_ff = hp.Int('d_ff', min_value=128, max_value=512, step=128)

    for _ in range(num_transformer_layers):
        # Multi-Head Self-Attention
        attn_output = MultiHeadAttention(num_heads=num_heads, key_dim=d_model // num_heads)(x, x)
        x = LayerNormalization(epsilon=1e-6)(x + attn_output)  # Residual + LayerNorm
        # Feed-Forward Network
        ffn_output = Dense(d_ff, activation='relu')(x)
        ffn_output = Dense(d_model)(ffn_output)
        x = LayerNormalization(epsilon=1e-6)(x + ffn_output)  # Residual + LayerNorm
        x = Dropout(hp.Float('transformer_dropout', 0.1, 0.5, step=0.1))(x)

    # GRU Decoder with Attention
    gru_units = hp.Int('gru_units', min_value=32, max_value=256, step=32)
    x = GRU(units=gru_units, return_sequences=True)(x)
    x = BatchNormalization()(x)
    x = Dropout(hp.Float('gru_dropout', 0.1, 0.5, step=0.1))(x)

    # Attention Layer
    attention = Attention()([x, x])

    # Global Average Pooling
    x = GlobalAveragePooling1D()(attention)

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

def build_final_model(window_size=24, input_dim=23, hp_values=None):
    inputs = Input(shape=(window_size, input_dim))

    x = Dense(hp_values['d_model'], activation='relu')(inputs)
    x = PositionalEncoding(hp_values['d_model'])(x)

    for _ in range(hp_values['num_transformer_layers']):
        attn_output = MultiHeadAttention(num_heads=hp_values['num_heads'], key_dim=hp_values['d_model'] // hp_values['num_heads'])(x, x)
        x = LayerNormalization(epsilon=1e-6)(x + attn_output)
        ffn_output = Dense(hp_values['d_ff'], activation='relu')(x)
        ffn_output = Dense(hp_values['d_model'])(ffn_output)
        x = LayerNormalization(epsilon=1e-6)(x + ffn_output)
        x = Dropout(hp_values['transformer_dropout'])(x)

    x = GRU(units=hp_values['gru_units'], return_sequences=True)(x)
    x = BatchNormalization()(x)
    x = Dropout(hp_values['gru_dropout'])(x)

    attention = Attention()([x, x])
    x = GlobalAveragePooling1D()(attention)

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
            os.path.join(output_dir, "best_transformer_gru_model.keras"),
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
    X, y = load_and_prepare_data(file_path, window_size=window_size)
    total_size = len(X)

    train_idx = int(total_size * 0.7)
    val_idx = int(total_size * 0.85)

    X_train, y_train = X[:train_idx], y[:train_idx]
    X_val, y_val = X[train_idx:val_idx], y[train_idx:val_idx]
    X_test, y_test = X[val_idx:], y[val_idx:]

    print(f"Données : Entraînement {X_train.shape}, Validation {X_val.shape}, Test {X_test.shape}")

    os.makedirs(output_dir, exist_ok=True)

    tuner = kt.RandomSearch(
        lambda hp: build_transformer_gru_model(hp, window_size=window_size, input_dim=23),
        objective='val_accuracy',
        max_trials=20,
        directory='tuner_dir',
        project_name='transformer_gru_tuning',
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

    model = build_final_model(window_size=window_size, input_dim=23, hp_values=hp_values)
    model, history = compile_and_train(model, X_train, y_train, X_val, y_val, output_dir=output_dir)

    best_model_path = os.path.join(output_dir, "best_transformer_gru_model.keras")
    if os.path.exists(best_model_path):
        model = tf.keras.models.load_model(best_model_path, custom_objects={'PositionalEncoding': PositionalEncoding})
        print(f"✔️ Meilleur modèle chargé depuis : {best_model_path}")

    y_val_pred = model.predict(X_val)
    y_test_pred = model.predict(X_test)

    preds_df = pd.DataFrame(np.argmax(y_test_pred, axis=1), columns=["Predicted_Class"])
    preds_df.to_csv(os.path.join(output_dir, "test_predictions.csv"), index=False)
    print(f"✔️ Prédictions sauvegardées dans : {os.path.join(output_dir, 'test_predictions.csv')}")

    model.save(os.path.join(output_dir, "transformer_gru_model.keras"))
    print(f"✔️ Modèle sauvegardé dans : {os.path.join(output_dir, 'transformer_gru_model.keras')}")

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
    main("/content/data/btc_4H_onehot_thresh_0.38.csv")