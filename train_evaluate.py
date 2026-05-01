# Import necessary libraries
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score, average_precision_score, f1_score
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
import argparse
import joblib
import os
import json
from feature_extract import compute_overthinking


def balance_data(df):
    min_size = df['target'].value_counts().min()
    df0 = df[df['target'] == 0].sample(min_size, random_state=42)
    df1 = df[df['target'] == 1].sample(min_size, random_state=42)
    df_balanced = pd.concat([df0, df1]).sample(frac=1, random_state=42).reset_index(drop=True)
    return df_balanced


def norm_data(train_df, val_df, test_df):
    """Fit scaler on train only, transform val and test."""
    X_train = train_df.drop(columns=['target'])
    y_train = train_df['target']
    X_val = val_df.drop(columns=['target'])
    y_val = val_df['target']
    X_test = test_df.drop(columns=['target'])
    y_test = test_df['target']

    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    def rebuild(X_orig, X_scaled, y):
        df_s = X_orig.copy()
        df_s.loc[:, :] = X_scaled
        df_s['target'] = y.values
        return df_s

    return (
        rebuild(X_train, X_train_scaled, y_train),
        rebuild(X_val, X_val_scaled, y_val),
        rebuild(X_test, X_test_scaled, y_test),
        scaler,
    )


def load_data(train_path, test_path):
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    layer_cols = [col for col in train_df.columns if col.startswith("H_")]
    layers = len(layer_cols)
    return train_df, test_df, layers


def preprocess_data(train_df_full, test_df_full, layers):
    tk = 10
    entropy_features = [f"H_{i}" for i in range(layers)]
    img_attn_features = [f"IA_{i}" for i in range(layers)]
    txt_attn_features = [f"TA_{i}" for i in range(layers)]

    if 'overthinking_score' not in train_df_full.columns:
        train_df_full['overthinking_score'] = train_df_full.apply(
            lambda row: compute_overthinking(row), axis=1)
        test_df_full['overthinking_score'] = test_df_full.apply(
            lambda row: compute_overthinking(row), axis=1)

    train_df_full['repeat'] = train_df_full.duplicated(
        subset=['image_id', 'next_token']).astype(int)
    test_df_full['repeat'] = test_df_full.duplicated(
        subset=['image_id', 'next_token']).astype(int)

    columns = (entropy_features + img_attn_features + txt_attn_features
               + ["overthinking_score", "repeat", "target"])

    train_df = train_df_full[columns].copy()
    test_df = test_df_full[columns].copy()

    # Split train into train and val
    train_part, val_part = train_test_split(
        train_df, test_size=0.10, random_state=42, stratify=train_df['target'])
    train_part = train_part.reset_index(drop=True)
    val_part = val_part.reset_index(drop=True)

    # Keep unscaled originals for GB 
    train_orig = train_part.copy()
    val_orig = val_part.copy()
    test_orig = test_df.copy()

    # Scale 
    train_scaled, val_scaled, test_scaled, scaler = norm_data(
        train_part, val_part, test_df)

    # Balance only the training split
    train_scaled = balance_data(train_scaled)
    train_orig = balance_data(train_orig)

    return (train_scaled, val_scaled, test_scaled,
            train_orig, val_orig, test_orig, scaler)



def find_best_threshold(y_true, y_probs):
    """Return threshold in [0.05, 0.95] that maximises macro F1."""
    thresholds = np.arange(0.05, 0.96, 0.01)
    best_thresh, best_f1 = 0.5, 0.0
    for t in thresholds:
        preds = (y_probs >= t).astype(int)
        f = f1_score(y_true, preds, average='macro', zero_division=0)
        if f > best_f1:
            best_f1, best_thresh = f, t
    return best_thresh, best_f1



def compute_metrics(y_true, y_pred, y_prob):
    report = classification_report(y_true, y_pred, digits=4, output_dict=True)
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'f1_macro': f1_score(y_true, y_pred, average='macro'),
        'f1_weighted': f1_score(y_true, y_pred, average='weighted'),
        'auc': roc_auc_score(y_true, y_prob),
        'ap': average_precision_score(y_true, y_prob),
        'precision_class1': report.get('1', {}).get('precision', None),
        'recall_class1': report.get('1', {}).get('recall', None),
        'f1_class1': report.get('1', {}).get('f1-score', None),
    }


def save_results(metrics: dict, best_params: dict, model_type: str,
                 output_csv_path: str):
    row = {'model_type': model_type}
    row.update(best_params)
    row.update(metrics)
    df_out = pd.DataFrame([row])

    if os.path.exists(output_csv_path):
        df_existing = pd.read_csv(output_csv_path)
        df_out = pd.concat([df_existing, df_out], ignore_index=True)

    os.makedirs(os.path.dirname(os.path.abspath(output_csv_path)), exist_ok=True)
    df_out.to_csv(output_csv_path, index=False)
    print(f"\nResults saved to: {output_csv_path}")



def train_models(model_type, train_df, val_df, test_df,
                 train_orig, val_orig, test_orig,
                 output_csv_path, model_save_dir="./saved_models"):

    os.makedirs(model_save_dir, exist_ok=True)

    if model_type == "LR":
        param_grid = {
            'C': [0.001, 0.01, 0.1, 1.0, 10.0],
            'penalty': ['l2'],
        }
        best_f1, best_params, best_model = 0.0, {}, None

        for C in param_grid['C']:
            model = LogisticRegression(
                C=C, max_iter=2000, solver='lbfgs', random_state=42)
            model.fit(train_df.drop(['target'], axis=1), train_df['target'])
            y_val_pred = model.predict(val_df.drop(['target'], axis=1))
            val_f1 = f1_score(val_df['target'], y_val_pred, average='macro')
            print(f"  LR C={C:.4f}  val_f1={val_f1:.4f}")
            if val_f1 > best_f1:
                best_f1, best_params, best_model = val_f1, {'C': C}, model

        print(f"\nBest LR params: {best_params}  val_f1={best_f1:.4f}")

        # Test evaluation
        y_pred = best_model.predict(test_df.drop(['target'], axis=1))
        y_prob = best_model.predict_proba(test_df.drop(['target'], axis=1))[:, 1]

        print("\n=== Logistic Regression (best model) ===")
        print("Accuracy:", accuracy_score(test_df['target'], y_pred))
        print(classification_report(test_df['target'], y_pred, digits=4))
        print(f"AUC: {roc_auc_score(test_df['target'], y_pred):.4f}")
        print(f"AP:  {average_precision_score(test_df['target'], y_pred):.4f}")

        metrics = compute_metrics(test_df['target'], y_pred, y_prob)
        save_results(metrics, best_params, model_type, output_csv_path)

        model_path = os.path.join(model_save_dir, "best_LR.pkl")
        joblib.dump(best_model, model_path)
        print(f"Model saved to: {model_path}")


    elif model_type == "GB":
        n_estimators_list = [100, 200, 300]
        max_depth_list = [3, 5, 10]
        lr_list = [0.01, 0.05, 0.1]

        best_f1, best_params, best_model, best_threshold = 0.0, {}, None, 0.5

        for n_est in n_estimators_list:
            for depth in max_depth_list:
                for lr in lr_list:
                    model = GradientBoostingClassifier(
                        n_estimators=n_est, learning_rate=lr,
                        max_depth=depth, random_state=42)
                    model.fit(
                        train_orig.drop(['target'], axis=1), train_orig['target'])
                    y_val_prob = model.predict_proba(
                        val_orig.drop(['target'], axis=1))[:, 1]

                    # Find threshold that maximises val F1
                    thresh, val_f1 = find_best_threshold(val_orig['target'], y_val_prob)
                    print(f"  GB n_est={n_est} depth={depth} lr={lr}  "
                          f"best_thresh={thresh:.2f}  val_f1={val_f1:.4f}")

                    if val_f1 > best_f1:
                        best_f1 = val_f1
                        best_params = {'n_estimators': n_est,
                                       'max_depth': depth,
                                       'learning_rate': lr,
                                       'best_threshold': thresh}
                        best_model = model
                        best_threshold = thresh

        print(f"\nBest GB params: {best_params}  val_f1={best_f1:.4f}")

        # Test evaluation using the best threshold
        y_prob = best_model.predict_proba(
            test_orig.drop(['target'], axis=1))[:, 1]
        y_pred = (y_prob >= best_threshold).astype(int)

        print("\n=== Gradient Boosting (best model, tuned threshold) ===")
        print(f"Threshold used: {best_threshold:.2f}")
        print("Accuracy:", accuracy_score(test_orig['target'], y_pred))
        print(classification_report(test_orig['target'], y_pred, digits=4))
        print(f"AUC: {roc_auc_score(test_orig['target'], y_prob):.4f}")
        print(f"AP:  {average_precision_score(test_orig['target'], y_prob):.4f}")

        metrics = compute_metrics(test_orig['target'], y_pred, y_prob)
        save_results(metrics, best_params, model_type, output_csv_path)

        model_path = os.path.join(model_save_dir, "best_GB.pkl")
        joblib.dump({'model': best_model, 'threshold': best_threshold}, model_path)
        print(f"Model + threshold saved to: {model_path}")

    elif model_type == "MLP":
        hidden_units_list = [32, 64, 128]
        lr_list = [0.1, 0.01, 0.001]
        solver_list = ['adam', 'sgd']

        best_f1, best_params, best_model = 0.0, {}, None

        X_train = train_df.drop(['target'], axis=1)
        y_train = train_df['target']
        X_val = val_df.drop(['target'], axis=1)
        y_val = val_df['target']

        for units in hidden_units_list:
            for lr in lr_list:
                for solver in solver_list:
                    model = MLPClassifier(
                        hidden_layer_sizes=(units,),
                        activation='relu',
                        solver=solver,
                        max_iter=2000,
                        random_state=42,
                        learning_rate_init=lr,
                    )
                    model.fit(X_train, y_train)
                    y_val_pred = model.predict(X_val)
                    val_f1 = f1_score(y_val, y_val_pred, average='macro')
                    print(f"  MLP units={units} lr={lr} solver={solver}  "
                          f"val_f1={val_f1:.4f}")
                    if val_f1 > best_f1:
                        best_f1 = val_f1
                        best_params = {
                            'hidden_units': units, 'lr': lr, 'solver': solver}
                        best_model = model

        print(f"\nBest MLP params: {best_params}  val_f1={best_f1:.4f}")

        X_test = test_df.drop(['target'], axis=1)
        y_test = test_df['target']
        y_pred = best_model.predict(X_test)
        y_prob = best_model.predict_proba(X_test)[:, 1]

        print("\n=== MLP Classifier (best model) ===")
        print("Accuracy:", accuracy_score(y_test, y_pred))
        print(classification_report(y_test, y_pred, digits=4))
        print(f"AUC: {roc_auc_score(y_test, y_prob):.4f}")
        print(f"AP:  {average_precision_score(y_test, y_prob):.4f}")

        metrics = compute_metrics(y_test, y_pred, y_prob)
        save_results(metrics, best_params, model_type, output_csv_path)

        model_path = os.path.join(model_save_dir, "best_MLP.pkl")
        joblib.dump(best_model, model_path)
        print(f"Model saved to: {model_path}")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train classifiers for overthinking detection")
    parser.add_argument("--train_path", type=str,
                        default="./data/LLaVA-1.5/train.csv")
    parser.add_argument("--test_path", type=str,
                        default="./data/LLaVA-1.5/test.csv")
    parser.add_argument("--model_type", type=str, default="LR",
                        choices=["LR", "GB", "MLP"])
    parser.add_argument("--output_csv", type=str,
                        default="./results/metrics.csv",
                        help="Path to CSV file where test metrics are saved")
    parser.add_argument("--model_save_dir", type=str,
                        default="./saved_models",
                        help="Directory to save the best model")
    args = parser.parse_args()

    # Load
    train_df_full, test_df_full, layers = load_data(args.train_path, args.test_path)

    # Preprocess
    (train_df, val_df, test_df,
     train_orig, val_orig, test_orig, scaler) = preprocess_data(
        train_df_full, test_df_full, layers)

    # Train with HP tuning
    train_models(
        args.model_type,
        train_df, val_df, test_df,
        train_orig, val_orig, test_orig,
        output_csv_path=args.output_csv,
        model_save_dir=args.model_save_dir,
    )