🩺 Integrated Diabetic Complication Prediction System

📌 Overview

Diabetes is a growing global health concern, often leading to severe complications such as diabetic retinopathy, nephropathy, and cardiovascular diseases. Early detection and prevention are vital for improving patient outcomes and reducing long-term risks.

This project introduces an integrated machine learning-based web platform for the prediction and management of major diabetic complications using a combination of deep learning and ensemble models.


🚀 Features

🔍 Diabetic Retinopathy Detection

Uses Convolutional Neural Networks (CNNs) to analyze retinal images

Classifies retinopathy into five stages

Offers diagnostic insights and preventive recommendations

💧 Diabetic Nephropathy Prediction

Utilizes an ensemble learning model on structured clinical data

Inputs: blood glucose, kidney function, blood pressure, medication history

Outputs a risk score and personalized preventive strategies

❤️ Cardiovascular Disease Risk Assessment

Employs a regression-based ensemble model

Evaluates cholesterol, BP, BMI, age, diabetes history

Provides heart disease risk scores and lifestyle recommendations

👤 Secure User Management

User authentication using Flask-Login

Personalized dashboards for patients and professionals

📊 Health Trend Monitoring

Track prediction history and monitor risk evolution over time

🧠 Technologies Used

Frontend: HTML, CSS, JavaScript (Bootstrap)

Backend: Python, Flask

Machine Learning & Deep Learning:

CNN (for image classification)

Random Forest, Gradient Boosting, Logistic Regression (for tabular prediction)

Voting Classifier (ensemble methods)

Database: MySQL (via SQLAlchemy)

Model Persistence: joblib / pickle

