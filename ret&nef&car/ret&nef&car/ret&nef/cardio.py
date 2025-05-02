import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Step 1: Load the dataset (Update the path)
data = pd.read_csv(r"C:\Users\ANUHYA ADUSUMILLI\Desktop\ret&nef-2\ret&nef\dataset\cardiovascular.csv")  # Ensure the CSV file is in the same directory

# Step 2: Handle missing values
imputer = SimpleImputer(strategy='mean')
X = data.drop(columns=['TenYearCHD'])  # Features
y = data['TenYearCHD']  # Target

X_imputed = imputer.fit_transform(X)  # Fill missing values

# Step 3: Train/test split
X_train, X_test, y_train, y_test = train_test_split(X_imputed, y, test_size=0.2, random_state=42)

# Step 4: Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Step 5: Create an ensemble model
log_clf = LogisticRegression(random_state=42, max_iter=1000)
rf_clf = RandomForestClassifier(random_state=42, n_estimators=100)
svm_clf = SVC(probability=True, random_state=42)

ensemble_model = VotingClassifier(estimators=[
    ('lr', log_clf),
    ('rf', rf_clf),
    ('svc', svm_clf)], voting='soft')

# Step 6: Train the ensemble model
ensemble_model.fit(X_train_scaled, y_train)

# Step 7: Evaluate the model
y_pred = ensemble_model.predict(X_test_scaled)
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
print(classification_report(y_test, y_pred))

# Step 8: Save the trained model
joblib.dump(ensemble_model, "ensemble_model_cardio(14-03-25).pkl")
print("✅ Model saved as 'ensemble_model_cardio.pkl'")
