import os
import cv2
import numpy as np
import tensorflow as tf
import pandas as pd
import joblib
from flask import Flask, render_template, request, redirect, url_for, flash
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from flask_wtf import FlaskForm
from wtforms import FileField, SubmitField, FloatField, SelectField
from wtforms.validators import DataRequired
from flask_sqlalchemy import SQLAlchemy
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from flask_wtf import FlaskForm
from wtforms import IntegerField, FloatField, BooleanField, SubmitField  # ✅ Import BooleanField
from wtforms.validators import DataRequired


# Initialize Flask app
app = Flask(__name__)
app.secret_key = 'your_secret_key'

# Configure SQLAlchemy for MySQL
app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql://root:@localhost/retinopathy'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False 
db = SQLAlchemy(app)

# Define User model
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(100), unique=True, nullable=False)
    password = db.Column(db.String(128), nullable=False)
    name = db.Column(db.String(100), nullable=False)
    phone = db.Column(db.String(20), nullable=False)
    address = db.Column(db.String(255), nullable=False)

# Define image upload form
class UploadForm(FlaskForm):
    image = FileField('Image', validators=[DataRequired()])
    submit = SubmitField('Submit')

# Define nephropathy prediction form
class NephropathyForm(FlaskForm):
    age = FloatField('Age', validators=[DataRequired()])
    diabetes_duration = FloatField('Diabetes duration (y)', validators=[DataRequired()])
    height = FloatField('Height (cm)', validators=[DataRequired()])
    weight = FloatField('Weight (kg)', validators=[DataRequired()])
    bmi = FloatField('BMI (kg/m2)', validators=[DataRequired()])
    sbp = FloatField('SBP (mmHg)', validators=[DataRequired()])
    dbp = FloatField('DBP (mmHg)', validators=[DataRequired()])
    hba1c = FloatField('HbA1c (%)', validators=[DataRequired()])
    fbg = FloatField('FBG (mmol/L)', validators=[DataRequired()])
    tg = FloatField('TG (mmoll)', validators=[DataRequired()])
    c_peptide = FloatField('C-peptide (ng/ml)', validators=[DataRequired()])
    tc = FloatField('TC (mmoll)', validators=[DataRequired()])
    hdlc = FloatField('HDLC (mmoll)', validators=[DataRequired()])
    ldlc = FloatField('LDLC (mmoll)', validators=[DataRequired()])
    retinopathy = SelectField('Diabetic retinopathy (DR)', choices=[('Yes', 'Yes'), ('No', 'No')], validators=[DataRequired()])
    insulin = SelectField('Insulin', choices=[('Yes', 'Yes'), ('No', 'No')], validators=[DataRequired()])
    metformin = SelectField('Metformin', choices=[('Yes', 'Yes'), ('No', 'No')], validators=[DataRequired()])
    lipid_lowering_drugs = SelectField('Lipid lowering drugs', choices=[('Yes', 'Yes'), ('No', 'No')], validators=[DataRequired()])
    submit = SubmitField('Predict')

class CardioForm(FlaskForm):
    male = BooleanField('Male')  # Change from gender to male
    age = IntegerField('Age', validators=[DataRequired()])
    education = IntegerField('Education Level', validators=[DataRequired()])
    currentSmoker = BooleanField('Current Smoker')
    cigsPerDay = IntegerField('Cigarettes Per Day')
    BPMeds = BooleanField('On BP Medication')
    prevalentStroke = BooleanField('Had a Stroke')
    prevalentHyp = BooleanField('Hypertension')
    diabetes = BooleanField('Diabetes')
    totChol = FloatField('Total Cholesterol', validators=[DataRequired()])
    sysBP = FloatField('Systolic BP', validators=[DataRequired()])
    diaBP = FloatField('Diastolic BP', validators=[DataRequired()])
    BMI = FloatField('Body Mass Index', validators=[DataRequired()])
    heartRate = IntegerField('Heart Rate', validators=[DataRequired()])
    glucose = FloatField('Glucose Level', validators=[DataRequired()])
    
    submit = SubmitField('Predict')

# Load the pre-trained models
model_path_retino = r'C:\Users\ANUHYA ADUSUMILLI\Desktop\ret&nef\my_model(260624-1650).h5'
model_retino = tf.keras.models.load_model(model_path_retino, compile=False)

model_path_nephro = r'C:\Users\ANUHYA ADUSUMILLI\Desktop\ret&nef\ensemble_model_diabetic_nephropathy(120924-2009).pkl'
model_nephro = joblib.load(model_path_nephro)

# Load models
model_path_cardio = r'C:\Users\ANUHYA ADUSUMILLI\Desktop\ret&nef-2\ensemble_model_cardio(14-03-25).pkl'
model_cardio = joblib.load(model_path_cardio)

training_data_path = r'C:\Users\ANUHYA ADUSUMILLI\Desktop\ret&nef\dataset\Diabetic_Nephropathy_v1.xlsx'
df_train = pd.read_excel(training_data_path)
df_train.columns = [col.strip() for col in df_train.columns]
X_train = df_train.drop(columns=['Diabetic nephropathy (DN)'])

# Define preprocessing for cardiovascular disease
cardio_features = [
    'male', 'age', 'education', 'currentSmoker', 'cigsPerDay', 'BPMeds', 
    'prevalentStroke', 'prevalentHyp', 'diabetes', 'totChol', 'sysBP', 
    'diaBP', 'BMI', 'heartRate', 'glucose'
]

num_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])
cat_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

cardio_preprocessor = ColumnTransformer(transformers=[
    ('num', StandardScaler(), ['age', 'totChol', 'sysBP', 'diaBP'])
])


# Define preprocessing for nephropathy
numerical_features = ['Age', 'Diabetes duration (y)', 'Height(cm)', 'Weight(kg)', 'BMI (kg/m2)', 'SBP (mmHg)', 'DBP (mmHg)', 'HbA1c (%)', 'FBG (mmol/L)', 'TG（mmoll）', 'C-peptide (ng/ml）', 'TC（mmoll）', 'HDLC（mmoll）', 'LDLC（mmoll）']
categorical_features = ['Diabetic retinopathy (DR)', 'Insulin', 'Metformin', 'Lipid lowering drugs']

numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ])

preprocessor.fit(X_train)

def preprocess_image(image, target_size=(128, 128)):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, target_size)
    image = image.astype('float32') / 255.0
    image = np.expand_dims(image, axis=0)
    return image

def analyze_image(image):
    preprocessed_image = preprocess_image(image)
    prediction = model_retino.predict(preprocessed_image)
    prediction_str = ', '.join(str(x) for x in prediction[0])
    class_label = np.argmax(prediction)
    return prediction_str, class_label

def generate_preventive_measures(class_label):
    preventive_measures = {
        0: [
            "1) Encourage regular eye exams to detect early signs of retinopathy.",
            "2) Maintain stable blood sugar levels through medication, diet, and exercise.",
            "3) Promote a healthy lifestyle with a balanced diet and regular exercise."
        ],
        1: [
            "1) Regularly monitor blood sugar levels and adhere to treatment plans.",
            "2) Control blood pressure and cholesterol levels to reduce the risk of complications.",
            "3) Manage stress and adopt healthy coping mechanisms."
        ],
        2: [
            "1) Manage diabetes through proper medication, diet, and exercise.",
            "2) Monitor blood sugar levels regularly and seek medical attention for any abnormalities.",
            "3) Follow recommended lifestyle changes to prevent diabetic retinopathy."
        ],
        3: [
            "1) Seek immediate medical attention if experiencing symptoms of proliferative diabetic retinopathy.",
            "2) Undergo regular eye exams to monitor the progression of the condition.",
            "3) Follow treatment plans recommended by healthcare professionals."
        ],
        4: [
            "1) Adopt healthy lifestyle habits to manage diabetes and prevent complications.",
            "2) Attend regular eye exams to detect and address diabetic retinopathy early.",
            "3) Follow medical advice and treatment plans to minimize the impact of retinopathy."
        ]
    }
    return preventive_measures.get(class_label, [])

def make_prediction(input_data):
    input_df = pd.DataFrame([input_data])
    processed_data = preprocessor.transform(input_df)
    prediction = model_nephro.predict(processed_data)
    return prediction[0]

# Function for cardiovascular disease prediction
def make_cardio_prediction(input_data):
    input_df = pd.DataFrame([input_data])
    processed_data = cardio_preprocessor.transform(input_df)
    prediction = model_cardio.predict(processed_data)
    return 'Cardiovascular disease detected' if prediction[0] == 1 else 'No cardiovascular disease detected'


@app.route('/cardio', methods=['GET', 'POST'])
def cardio():
    form = CardioForm()

    if form.validate_on_submit():
        input_data = pd.DataFrame([{
            'male': int(form.male.data),
            'age': int(form.age.data),
            'education': int(form.education.data),
            'currentSmoker': int(form.currentSmoker.data),
            'cigsPerDay': int(form.cigsPerDay.data),
            'BPMeds': int(form.BPMeds.data),
            'prevalentStroke': int(form.prevalentStroke.data),
            'prevalentHyp': int(form.prevalentHyp.data),
            'diabetes': int(form.diabetes.data),
            'totChol': float(form.totChol.data),
            'sysBP': float(form.sysBP.data),
            'diaBP': float(form.diaBP.data),
            'BMI': float(form.BMI.data),
            'heartRate': int(form.heartRate.data),
            'glucose': float(form.glucose.data)
        }])

        print("Input Data Columns:", input_data.columns.tolist())  # Debugging
        print("Input Data Preview:\n", input_data.head())  # Debugging

        try:
            # Get prediction
            predicted_value = model_cardio.predict(input_data.values)[0]
           


            # Convert prediction to Yes/No
            prediction_result = "Yes" if predicted_value == 1 else "No"

            return render_template('cardio_output.html', prediction=prediction_result)

        except ValueError as e:
            return f"Error in model prediction: {e}"

    return render_template('cardio_form.html', form=form)


# Initialize Flask-Login
login_manager = LoginManager()
login_manager.init_app(app)

@login_manager.user_loader
def load_user(user_id):
   return db.session.get(User, int(user_id))


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        user = User.query.filter_by(username=username).first()
        if user and user.password == password:
            login_user(user)
            flash('Logged in successfully.', 'success')
            return redirect(url_for('upload'))
        else:
            flash('Invalid username or password', 'error')
    return render_template('login.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    flash('Logged out successfully.', 'success')
    return redirect(url_for('index'))

@app.route('/upload', methods=['GET', 'POST'])
@login_required
def upload():
    form = UploadForm()
    if form.validate_on_submit():
        f = form.image.data
        if f:
            if not os.path.exists('uploads'):
                os.makedirs('uploads', exist_ok=True)
                
            filename = f.filename
            image_path = os.path.join('uploads', filename)
            f.save(image_path)
            image = cv2.imread(image_path)
            
            if image is None:
                flash('Invalid image file. Please upload a valid image.', 'error')
            else:
                result, class_label = analyze_image(image)
                classes = ['NO_DR', 'Moderate', 'Moderate', 'Proliferate', 'Severe']
                class_name = classes[class_label]
                preventive_measures = generate_preventive_measures(class_label)
                flash(f'Analysis result: {result}, Class: {class_name}', 'success')
                return render_template('output.html', result=result, class_label=class_name, preventive_measures=preventive_measures)
        else:
            flash('No file uploaded.', 'error')
    return render_template('upload.html', form=form)

@app.route('/nephropathy-form', methods=['GET', 'POST'])
@login_required
def nephro_form():
    form = NephropathyForm()
    if form.validate_on_submit():
        input_data = {
            'Age': form.age.data,
            'Diabetes duration (y)': form.diabetes_duration.data,
            'Height(cm)': form.height.data,
            'Weight(kg)': form.weight.data,
            'BMI (kg/m2)': form.bmi.data,
            'SBP (mmHg)': form.sbp.data,
            'DBP (mmHg)': form.dbp.data,
            'HbA1c (%)': form.hba1c.data,
            'FBG (mmol/L)': form.fbg.data,
            'TG（mmoll）': form.tg.data,
            'C-peptide (ng/ml）': form.c_peptide.data,
            'TC（mmoll）': form.tc.data,
            'HDLC（mmoll）': form.hdlc.data,
            'LDLC（mmoll）': form.ldlc.data,
            'Diabetic retinopathy (DR)': form.retinopathy.data,
            'Insulin': form.insulin.data,
            'Metformin': form.metformin.data,
            'Lipid lowering drugs': form.lipid_lowering_drugs.data
        }
        # Debugging
        print("Form Data:", input_data)
        result = make_prediction(input_data)
        result_text = "Diabetic nephropathy detected" if result == 1 else "No diabetic nephropathy detected"
        flash(result_text, 'success')
        # Render a result template
        return render_template('nephro_output.html', result=result_text)
    return render_template('form.html', form=form)


@app.route('/predict-nephropathy', methods=['POST'])
@login_required
def predict_nephropathy():
    input_data = {
        'Age': float(request.form['age']),
        'Diabetes duration (y)': float(request.form['diabetes_duration']),
        'Height(cm)': float(request.form['height']),
        'Weight(kg)': float(request.form['weight']),
        'BMI (kg/m2)': float(request.form['bmi']),
        'SBP (mmHg)': float(request.form['sbp']),
        'DBP (mmHg)': float(request.form['dbp']),
        'HbA1c (%)': float(request.form['hba1c']),
        'FBG (mmol/L)': float(request.form['fbg']),
        'TG（mmoll）': float(request.form['tg']),
        'C-peptide (ng/ml）': float(request.form['c_peptide']),
        'TC（mmoll）': float(request.form['tc']),
        'HDLC（mmoll）': float(request.form['hdlc']),
        'LDLC（mmoll）': float(request.form['ldlc']),
        'Diabetic retinopathy (DR)': request.form['retinopathy'],
        'Insulin': request.form['insulin'],
        'Metformin': request.form['metformin'],
        'Lipid lowering drugs': request.form['lipid_lowering_drugs']
    }
    # Debugging
    print("Form Data:", input_data)
    result = make_prediction(input_data)
    result_text = "Diabetic nephropathy detected" if result == 1 else "No diabetic nephropathy detected"
    return f"<h3>{result_text}</h3>"

if __name__ == '__main__':
    app.run(debug=True)
