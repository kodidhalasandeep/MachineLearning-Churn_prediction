import joblib
import pandas as pd

def predict_customer_churn(customer_data):
    """Loads saved model and predicts churn status."""
    try:
        model = joblib.load("churn_model.pkl")
    except FileNotFoundError:
        print("[Error] Model file not found. Please run train_model.py first.")
        return

    # Convert input dict to DataFrame
    df = pd.DataFrame([customer_data])
    
    # Predict
    prediction = model.predict(df)[0]
    probability = model.predict_proba(df)[0][1] # Probability of churning

    status = "CHURNED (At Risk)" if prediction == 1 else "RETAINED (Safe)"
    print("\n--- Churn Prediction Result ---")
    print(f"Customer Status : {status}")
    print(f"Risk Probability: {probability * 100:.2f}%")

if __name__ == "__main__":
    # Test with a high-risk customer profile
    # (Low tenure, high monthly bills, multiple support calls, month-to-month contract)
    new_customer = {
        'tenure_months': 2,
        'monthly_charges': 88.0,
        'total_support_calls': 5,
        'contract_type': 0
    }
    
    predict_customer_churn(new_customer)