import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

def generate_sample_data():
    """Generates synthetic customer dataset for training."""
    data = {
        'tenure_months': [2, 24, 1, 12, 36, 3, 48, 6, 18, 30],
        'monthly_charges': [70.5, 20.0, 85.0, 50.0, 25.0, 90.0, 15.0, 65.0, 55.0, 30.0],
        'total_support_calls': [4, 0, 5, 2, 0, 6, 0, 3, 1, 0],
        'contract_type': [0, 1, 0, 1, 1, 0, 1, 0, 1, 1], # 0: Month-to-Month, 1: Annual
        'churn': [1, 0, 1, 0, 0, 1, 0, 1, 0, 0]          # 1: Churned, 0: Retained
    }
    return pd.DataFrame(data)

def main():
    print("Generating dataset...")
    df = generate_sample_data()

    # Split features and target variable
    X = df[['tenure_months', 'monthly_charges', 'total_support_calls', 'contract_type']]
    y = df['churn']

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("Training Random Forest Classifier...")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Evaluate model
    predictions = model.predict(X_test)
    acc = accuracy_score(y_test, predictions)
    print(f"Model Training Complete! Accuracy: {acc * 100:.2f}%")

    # Save the trained model to disk
    model_filename = "churn_model.pkl"
    joblib.dump(model, model_filename)
    print(f"Model successfully saved to '{model_filename}'")

if __name__ == "__main__":
    main()