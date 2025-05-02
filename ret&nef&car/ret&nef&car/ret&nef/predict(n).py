import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

# Define the path to your trained model
model_path = r'C:\Users\ANUHYA ADUSUMILLI\Desktop\Nephropathy\ensemble_model_diabetic_nephropathy(120924-1457).pkl'

# Load the trained ensemble model
model = joblib.load(model_path)

# Define numerical and categorical features
numerical_features = ['Age', 'Diabetes duration (y)', 'Height(cm)', 'Weight(kg)', 'BMI (kg/m2)', 'SBP (mmHg)', 'DBP (mmHg)', 'HbA1c (%)', 'FBG (mmol/L)', 'TG（mmoll）', 'C-peptide (ng/ml）', 'TC（mmoll）', 'HDLC（mmoll）', 'LDLC（mmoll）']
categorical_features = ['Diabetic retinopathy (DR)', 'Insulin', 'Metformin', 'Lipid lowering drugs']

# Define preprocessing for numerical and categorical data
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Combine preprocessing steps
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Fit the preprocessor on the training data (you need to have access to training data for this step)
def fit_preprocessor(training_data_path):
    df_train = pd.read_excel(training_data_path)
    df_train.columns = [col.strip() for col in df_train.columns]
    
    X_train = df_train.drop(columns=['Diabetic nephropathy (DN)'])
    preprocessor.fit(X_train)

# Example path to the training data
training_data_path = r'C:\Users\ANUHYA ADUSUMILLI\Desktop\Nephropathy\Diabetic_Nephropathy_v1.xlsx'
fit_preprocessor(training_data_path)

# Function to make a prediction
def make_prediction(input_data):
    # Convert input data to DataFrame
    input_df = pd.DataFrame([input_data], columns=numerical_features + categorical_features)
    
    # Preprocess the input data
    processed_data = preprocessor.transform(input_df)

    # Make prediction
    prediction = model.predict(processed_data)
    
    return prediction[0]

# Example input data with all 22 features
input_data = {
    'Age': 57,
    'Diabetes duration (y)': 10,
    'Height(cm)': 178,
    'Weight(kg)': 60,
    'BMI (kg/m2)': 18.9370029,
    'SBP (mmHg)': 101,
    'DBP (mmHg)': 69,
    'HbA1c (%)': 14.1,
    'FBG (mmol/L)': 17.42,
    'TG（mmoll）': 1.95,
    'C-peptide (ng/ml）': 0.98,
    'TC（mmoll）': 5.51,
    'HDLC（mmoll）': 1.08,
    'LDLC（mmoll）': 3.71,
    'Diabetic retinopathy (DR)': 'Yes',
    'Insulin': 'Yes',
    'Metformin': 'Yes',
    'Lipid lowering drugs': 'Yes'
}


# Make the prediction
result = make_prediction(input_data)
print(f'Prediction: {"Diabetic nephropathy detected" if result == 1 else "No diabetic nephropathy detected"}')
