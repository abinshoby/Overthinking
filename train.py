# Import libraries
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score, roc_curve, average_precision_score, f1_score
from sklearn.neural_network import MLPClassifier

def balance_data(df):
    min_size = df['target'].value_counts().min()
    # Split by class
    df0 = df[df['target'] == 0].sample(min_size, random_state=42)
    df1 = df[df['target'] == 1].sample(min_size, random_state=42)
    
    # Balanced dataset
    df_balanced = pd.concat([df0, df1]).sample(frac=1, random_state=42).reset_index(drop=True)
    return df_balanced

def norm_data(train_df, test_df):
    # Separate features and target
    X_train = train_df.drop(columns=['target'])
    y_train = train_df['target']
    
    X_test = test_df.drop(columns=['target'])
    y_test = test_df['target']
    
    # Initialize scaler and fit only on training data
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Recombine scaled features with target
    train_scaled = X_train.copy()
    train_scaled.loc[:, :] = X_train_scaled
    train_scaled['target'] = y_train.values
    
    test_scaled = X_test.copy()
    test_scaled.loc[:, :] = X_test_scaled
    test_scaled['target'] = y_test.values
    return train_scaled, test_scaled, scaler


train_df_full = pd.read_csv("/workspace/data/Detection/overthink/data3/with_score_train.csv")
test_df_full = pd.read_csv("/workspace/data/Detection/overthink/data3/with_score_test.csv")

# Select columns required for training and testing
layers=32
tk=10
entropy_features = [f"H_{i}" for i in range(layers)]
img_attn_features = [f"IA_{i}" for i in range(layers)]
txt_attn_features = [f"TA_{i}" for i in range(layers)]
columns = entropy_features + img_attn_features + txt_attn_features + ["overthinking_score", "target"]

train_df = train_df_full[columns]
test_df = test_df_full[columns]

# scale the data
train_df_orig = train_df.copy()
test_df_orig = test_df.copy()
train_df, test_df, scaler = norm_data(train_df, test_df)

# Balance the training data
train_df = balance_data(train_df)

# Train Logistic Regression
lr_model = LogisticRegression(max_iter=2000, solver='lbfgs',random_state=42)
lr_model.fit(train_df.drop(['target'], axis=1), train_df['target'])

y_pred_lr = lr_model.predict(test_df.drop(['target'], axis=1))
y_prob_lr = lr_model.predict_proba(test_df.drop(['target'], axis=1))[:, 1]

print("=== Logistic Regression ===")
print("Accuracy:", accuracy_score(test_df['target'], y_pred_lr))
print(classification_report(test_df['target'], y_pred_lr, digits=4))

# --- AUC ---
auc_score = roc_auc_score(test_df['target'], y_pred_lr)
ap = average_precision_score(test_df['target'], y_pred_lr)
print(f"AUC: {auc_score:.4f}")
print(f"AP: {ap:.4f}")


# Train Gradient Boosting Classifier
gb_model = GradientBoostingClassifier(n_estimators=200, learning_rate=0.1, max_depth=10, random_state=42)
gb_model.fit(train_df_orig.drop(['target'], axis=1), train_df_orig['target'])
y_pred_gb = gb_model.predict(test_df_orig.drop(['target'], axis=1))
y_probs_gb = gb_model.predict_proba(test_df_orig.drop(['target'], axis=1))[:, 1]
print("=== Gradient Boosting Classifier ===")
print("Accuracy:", accuracy_score(test_df_orig['target'], y_pred_gb))
print(classification_report(test_df_orig['target'], y_pred_gb, digits=4))
auc_score = roc_auc_score(test_df_orig['target'], y_probs_gb)
ap = average_precision_score(test_df_orig['target'], y_probs_gb)
print(f"AUC: {auc_score:.4f}")
print(f"AP: {ap:.4f}")


# Train MLP Classifier
X_train = train_df.drop(['target'], axis=1)
y_train = train_df['target']
X_test = test_df.drop(['target'], axis=1)
y_test = test_df['target']

mlp_model = MLPClassifier(
    hidden_layer_sizes=(128,),
    activation='relu',
    solver='sgd',
    max_iter=2000,
    random_state=42,
    learning_rate_init=0.01
)

mlp_model.fit(X_train, y_train)

y_pred_mlp = mlp_model.predict(X_test)
y_prob_mlp = mlp_model.predict_proba(X_test)[:, 1]

print("=== MLP Classifier ===")
print("Accuracy:", accuracy_score(test_df_orig['target'], y_pred_mlp))
print(classification_report(test_df_orig['target'], y_pred_mlp, digits=4))
auc_score = roc_auc_score(test_df_orig['target'], y_prob_mlp)
ap = average_precision_score(test_df_orig['target'], y_prob_mlp)
print(f"AUC: {auc_score:.4f}")
print(f"AP: {ap:.4f}")