import numpy as np
import pandas as pd
import os
import json
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import RandomizedSearchCV
import matplotlib.pyplot as plt
from scipy.stats import randint, uniform
import time

# === 1. Données d'entrée : lecture et préparation ===
def load_and_prepare_data(file_path, window_size=12):
    print("Chargement et préparation des données...")
    start_time = time.time()
    df = pd.read_csv(file_path)
    df.drop(columns=['date'], inplace=True, errors='ignore')

    # Colonnes de caractéristiques (18 normalisées, pour X.shape=(16836, 90))
    feature_cols = [
        'open_norm', 'high_norm', 'low_norm', 'close_norm', 'volume_norm',
        'RSI_norm', 'MACD_norm', 'MACD_signal_norm', 'macd_divergence_norm',
        'kumo_thickness_norm', 'price_to_cloud_distance_norm', 'price_in_kumo_distance_norm',
        'tenkan_kijun_diff_norm', 'price_vs_kijun_norm', 'price_vs_tenkan_norm',
        'chikou_vs_price_norm', 'ATR_norm', 'ADX_norm'
    ]

    # Vérifier que les colonnes existent
    feature_cols = [col for col in feature_cols if col in df.columns]
    if len(feature_cols) != 18:
        print(f"Colonnes trouvées : {feature_cols}")
        raise ValueError(f"Attendu 18 colonnes de caractéristiques, trouvé {len(feature_cols)}.")

    # Vérifier la présence de target_class
    if 'target_class' not in df.columns:
        raise ValueError("Colonne 'target_class' non trouvée dans le fichier CSV.")

    # Convertir target_class en labels numériques (Buy=0, Hold=1, Sell=2)
    label_map = {'Buy': 0, 'Hold': 1, 'Sell': 2}
    df['target_class'] = df['target_class'].map(label_map)
    if df['target_class'].isna().any():
        raise ValueError("Valeurs non valides dans target_class. Attendu : 'Buy', 'Hold', 'Sell'.")

    # Vectorisation pour accélérer
    X, y = [], []
    data = df[feature_cols].values
    labels = df['target_class'].values

    for i in range(window_size, len(df)):
        window = data[i-window_size:i]  # (window_size, n_features)
        features = np.hstack([
            np.mean(window, axis=0),  # Moyenne
            np.std(window, axis=0),   # Écart-type
            np.min(window, axis=0),   # Minimum
            np.max(window, axis=0),   # Maximum
            np.array([np.polyfit(np.arange(window_size), window[:, col], 1)[0] for col in range(window.shape[1])])  # Pente
        ])
        X.append(features)
        y.append(labels[i])
    X = np.array(X)  # (N, 18*5=90)
    y = np.array(y)  # (N,)
    elapsed = (time.time() - start_time) / 60
    print(f"Données préparées en {elapsed:.2f} minutes")
    print(f"Données : X shape {X.shape}, y shape {y.shape}")
    return X, y

# === 2. Modèle XGBoost avec RandomizedSearchCV ===
def build_xgboost_model(hp):
    model = XGBClassifier(
        objective='multi:softmax',
        num_class=3,
        n_estimators=hp['n_estimators'],
        max_depth=hp['max_depth'],
        learning_rate=hp['learning_rate'],
        subsample=hp['subsample'],
        colsample_bytree=hp['colsample_bytree'],
        random_state=42,
        tree_method='hist',  # Optimisé pour CPU
        eval_metric=['mlogloss', 'merror']
    )
    return model

def tune_xgboost_model(X_train, y_train, X_val, y_val, max_trials=10, output_dir="results"):
    param_dist = {
        'n_estimators': randint(50, 200),  # Réduit pour CPU
        'max_depth': randint(3, 6),        # Profondeur réduite
        'learning_rate': uniform(0.01, 0.19),  # [0.01, 0.2]
        'subsample': uniform(0.6, 0.4),    # [0.6, 1.0]
        'colsample_bytree': uniform(0.6, 0.4)  # [0.6, 1.0]
    }
    xgb = XGBClassifier(
        objective='multi:softmax',
        num_class=3,
        random_state=42,
        tree_method='hist',
        eval_metric=['mlogloss', 'merror']
    )

    tuner = RandomizedSearchCV(
        xgb,
        param_distributions=param_dist,
        n_iter=max_trials,
        scoring='accuracy',
        cv=3,
        verbose=2,
        n_jobs=-1,  # Utiliser tous les cœurs CPU
        random_state=42
    )
    print("Début de l'optimisation des hyperparamètres...")
    start_time = time.time()
    tuner.fit(X_train, y_train)  # Sans early_stopping_rounds
    elapsed = (time.time() - start_time) / 60
    print(f"Optimisation terminée en {elapsed:.2f} minutes")
    return tuner

# === 3. Compilation et entraînement ===
def compile_and_train(model, X_train, y_train, X_val, y_val, output_dir="results"):
    print("Entraînement du modèle final...")
    start_time = time.time()
    class_weights = compute_class_weight('balanced', classes=np.array([0, 1, 2]), y=y_train)
    sample_weight = np.array([class_weights[label] for label in y_train])

    eval_set = [(X_train, y_train), (X_val, y_val)]
    history = {'loss': [], 'val_loss': [], 'accuracy': [], 'val_accuracy': []}
    try:
        model.fit(
            X_train,
            y_train,
            sample_weight=sample_weight,
            eval_set=eval_set,
            early_stopping_rounds=10,
            verbose=100
        )
        history = {
            'loss': model.evals_result()['validation_0']['mlogloss'],
            'val_loss': model.evals_result()['validation_1']['mlogloss'],
            'accuracy': [1 - x for x in model.evals_result()['validation_0']['merror']],
            'val_accuracy': [1 - x for x in model.evals_result()['validation_1']['merror']]
        }
    except TypeError:
        print("early_stopping_rounds non supporté, entraînement sans early stopping...")
        model.fit(
            X_train,
            y_train,
            sample_weight=sample_weight,
            verbose=100
        )

    os.makedirs(output_dir, exist_ok=True)
    model_path = os.path.join(output_dir, "best_xgboost_model.json")
    model.save_model(model_path)
    print(f"✔️ Modèle sauvegardé dans : {model_path}")
    elapsed = (time.time() - start_time) / 60
    print(f"Entraînement terminé en {elapsed:.2f} minutes")
    return model, history

# === 4. Matrice de confusion ===
def plot_confusion_matrix(y_true, y_pred, output_dir="results", title="model", class_names=None):
    if class_names is None:
        class_names = ['Buy', 'Hold', 'Sell']

    cm = confusion_matrix(y_true, y_pred)

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

# === 5. Calcul des métriques ===
def print_classification_report(y_true, y_pred, title="Classification Report", output_dir="results"):
    print(f"\n📊 {title}")
    report = classification_report(
        y_true,
        y_pred,
        target_names=["Buy", "Hold", "Sell"],
        digits=4,
        output_dict=True
    )
    os.makedirs(output_dir, exist_ok=True)
    report_path = os.path.join(output_dir, f"{title.replace(' ', '_')}_report.txt")
    with open(report_path, 'w') as f:
        f.write(f"{title}\n")
        f.write(classification_report(
            y_true,
            y_pred,
            target_names=["Buy", "Hold", "Sell"],
            digits=4
        ))
    print(f"✔️ Rapport de classification sauvegardé dans : {report_path}")
    return report

# === 6. Fonction principale ===
def main(file_path, window_size=12, output_dir="results"):
    print("Début du traitement...")
    start_time = time.time()

    X, y = load_and_prepare_data(file_path, window_size=window_size)
    total_size = len(X)

    train_idx = int(total_size * 0.7)
    val_idx = int(total_size * 0.85)

    X_train, y_train = X[:train_idx], y[:train_idx]
    X_val, y_val = X[train_idx:val_idx], y[train_idx:val_idx]
    X_test, y_test = X[val_idx:], y[val_idx:]

    print(f"Données : Entraînement {X_train.shape}, Validation {X_val.shape}, Test {X_test.shape}")

    os.makedirs(output_dir, exist_ok=True)

    tuner = tune_xgboost_model(X_train, y_train, X_val, y_val, max_trials=10, output_dir=output_dir)

    best_hps = tuner.best_params_
    hp_path = os.path.join(output_dir, "best_hyperparameters.json")
    with open(hp_path, 'w') as f:
        json.dump(best_hps, f, indent=4)
    print(f"✔️ Meilleurs hyperparamètres sauvegardés dans : {hp_path}")
    print(f"Meilleurs hyperparamètres : {best_hps}")

    print("Entraînement du modèle final...")
    model = XGBClassifier(
        objective='multi:softmax',
        num_class=3,
        n_estimators=best_hps['n_estimators'],
        max_depth=best_hps['max_depth'],
        learning_rate=best_hps['learning_rate'],
        subsample=best_hps['subsample'],
        colsample_bytree=best_hps['colsample_bytree'],
        random_state=42,
        tree_method='hist',
        eval_metric=['mlogloss', 'merror']
    )
    model, history = compile_and_train(model, X_train, y_train, X_val, y_val, output_dir=output_dir)

    best_model_path = os.path.join(output_dir, "best_xgboost_model.json")
    if os.path.exists(best_model_path):
        model.load_model(best_model_path)
        print(f"✔️ Meilleur modèle chargé depuis : {best_model_path}")

    print("Génération des prédictions...")
    y_val_pred = model.predict(X_val)
    y_test_pred = model.predict(X_test)

    preds_df = pd.DataFrame(y_test_pred, columns=["Predicted_Class"])
    preds_df['Predicted_Class'] = preds_df['Predicted_Class'].map({0: 'Buy', 1: 'Hold', 2: 'Sell'})
    preds_df.to_csv(os.path.join(output_dir, "test_predictions.csv"), index=False)
    print(f"✔️ Prédictions sauvegardées dans : {os.path.join(output_dir, 'test_predictions.csv')}")

    if history['loss']:  # Vérifier si l'historique est non vide
        plt.figure()
        plt.plot(history['loss'], label='Training Loss')
        plt.plot(history['val_loss'], label='Validation Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.savefig(os.path.join(output_dir, "loss_plot.png"))
        plt.close()
        print(f"✔️ Courbe de perte sauvegardée dans : {os.path.join(output_dir, 'loss_plot.png')}")

        plt.figure()
        plt.plot(history['accuracy'], label='Training Accuracy')
        plt.plot(history['val_accuracy'], label='Validation Accuracy')
        plt.title('Training and Validation Accuracy')
        plt.legend()
        plt.savefig(os.path.join(output_dir, "accuracy_plot.png"))
        plt.close()
        print(f"✔️ Courbe d'accuracy sauvegardée dans : {os.path.join(output_dir, 'accuracy_plot.png')}")

    history_df = pd.DataFrame(history)
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

    elapsed = (time.time() - start_time) / 60
    print(f"✔️ Traitement terminé en {elapsed:.2f} minutes")

if __name__ == "__main__":
    main("/content/data/btc_4H_onehot_thresh_0.47.csv", window_size=12)