import pandas as pd
import joblib  # For loading the trained model
from sklearn.preprocessing import StandardScaler  # Modify if needed

# Load the trained model (Ensure this file exists)
model_cardio = joblib.load(r"C:\Users\ANUHYA ADUSUMILLI\Desktop\ret&nef-2\ret&nef\ensemble_model(cardio3).pkl")

# Define preprocessing (manual feature scaling if needed)
def preprocess_input(input_data):
    input_df = pd.DataFrame([input_data])
    
    # Define the features that need scaling (Modify if necessary)
    features_to_scale = ["age", "totChol", "sysBP", "diaBP", "BMI", "heartRate", "glucose"]
    
    # Apply Standard Scaling
    scaler = StandardScaler()
    input_df[features_to_scale] = scaler.fit_transform(input_df[features_to_scale])  # fit_transform for single use

    return input_df


# Function for cardiovascular disease prediction
def make_cardio_prediction(input_data):
    try:
        # Preprocess input data manually
        processed_data = preprocess_input(input_data)

        # Predict using the loaded model
        prediction = model_cardio.predict(processed_data)

        # Return readable output
        return "Cardiovascular disease detected" if prediction[0] == 1 else "No cardiovascular disease detected"

    except Exception as e:
        return f"Error in prediction: {str(e)}"


# Example Test Cases
if __name__ == "__main__":
    test_data_1 = {
        "male": 1, "age": 39, "education": 4, "currentSmoker": 0, "cigsPerDay": 0, 
        "BPMeds": 0, "prevalentStroke": 0, "prevalentHyp": 0, "diabetes": 0, 
        "totChol": 195, "sysBP": 106, "diaBP": 70, "BMI": 26.97, 
        "heartRate": 80, "glucose": 77
    }
    
    test_data_2 = {
        "male": 0, "age": 46, "education": 2, "currentSmoker": 0, "cigsPerDay": 0, 
        "BPMeds": 0, "prevalentStroke": 0, "prevalentHyp": 0, "diabetes": 0, 
        "totChol": 250, "sysBP": 121, "diaBP": 81, "BMI": 28.73, 
        "heartRate": 95, "glucose": 76
    }
    
    # New Test Data
    test_data_3 = {
        "male": 0, "age": 61, "education": 3, "currentSmoker": 1, "cigsPerDay": 30, 
        "BPMeds": 0, "prevalentStroke": 0, "prevalentHyp": 1, "diabetes": 0, 
        "totChol": 225, "sysBP": 150, "diaBP": 95, "BMI": 28.58, 
        "heartRate": 65, "glucose": 103
        }


    # Run predictions
    print("Test Case 1:", make_cardio_prediction(test_data_1))
    print("Test Case 2:", make_cardio_prediction(test_data_2))
    print("Test Case 3:", make_cardio_prediction(test_data_3))
