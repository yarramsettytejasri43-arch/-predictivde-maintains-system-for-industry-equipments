import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report
from data_preprocessing import preprocess_data

def train():

    # Load correct dataset (based on your folder)
    df = pd.read_csv("machine_maintenance.csv")

    df = preprocess_data(df)

    X = df.drop(["machine_id", "failure_within_7days"], axis=1)
    y = df["failure_within_7days"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Train XGBoost
    model = XGBClassifier(eval_metric='logloss')
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    print("Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))

    # Create models folder automatically
    os.makedirs("models", exist_ok=True)

    # Save model
    joblib.dump(model, "models/model.pkl")

    print("Model saved successfully inside models folder!")

if __name__ == "__main__":
    train()
