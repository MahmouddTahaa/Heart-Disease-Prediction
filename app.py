import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import tensorflow as tf
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPClassifier

st.write(f"""
# Heart Disease Prediction App
""")

st.write('This app predicts if a patient has a **heart disease**!')

st.image('./images/heart.jpg')

st.sidebar.header('Input Features')

clfs = ['Descision Tree', 'Random Forest', 'KNN', 'Genitic Algorithm', 'Diffrential Evolution Algorithm', 'Artificial Neural Network', 'MLP (using PCA)']

classifier = st.sidebar.selectbox("### Select a Classifier", clfs)

uploaded_file = st.sidebar.file_uploader("Upload your input CSV file", type=["csv"])

if uploaded_file is not None:
    input_df = pd.read_csv(uploaded_file)
else:
    def input_features():
        smoking = st.sidebar.radio('Smoking', ['Yes', 'No'])
        alcohol = st.sidebar.radio('Alcohol Drinking', ['Yes', 'No'])
        stroke = st.sidebar.radio('Stroke', ['Yes', 'No'])
        diabetic = st.sidebar.radio('Diabetic', ['Yes', 'No'])
        physical_activity= st.sidebar.radio('Physically Active', ['Yes', 'No'])
        diffWalking = st.sidebar.radio('Difficulty Walking', ['Yes', 'No'])
        asthma= st.sidebar.radio('Asthma', ['Yes', 'No'])
        kidney_disease = st.sidebar.radio('Kidney Disease', ['Yes', 'No'])
        skin_cancer = st.sidebar.radio('Skin Cancer', ['Yes', 'No'])
        sex = st.sidebar.radio('Sex', ['Male', 'Female'])
        race = st.sidebar.selectbox('Race', ('White', 'Black', 'Asian', 'American Indian/Alaskan Native', 'Hispanic', 'Other'))
        general_health = st.sidebar.selectbox('General Health', ('Excellent', 'Very Good', 'Good', 'Fair', 'Poor'))
        age_category = st.sidebar.selectbox('Age Category', ('18-24', '25-29', '30-34', '35-39', '40-44', '45-49', '50-54', '55-59', '60-64', '65-69', '70-74', '75-79', '80 or older'))
        bmi = st.sidebar.slider('Body Mass Index', 15.0, 50.0, 29.4, 0.1)
        physical_health = st.sidebar.slider('Physical Health', 0, 30, 18, 1)
        mental_health = st.sidebar.slider('Mental Health', 0, 30, 18, 1)
        sleep_time = st.sidebar.slider('Sleep Time', 3, 23, 1)
        
        

        data = {'BMI': bmi,
                'Smoking': smoking,
                'AlcoholDrinking': alcohol,
                'Stroke': stroke,
                'PhysicalHealth': physical_health,
                'MentalHealth': mental_health,
                'DiffWalking': diffWalking,
                'Sex': sex,
                'AgeCategory': age_category,
                'Race': race,
                'Diabetic': diabetic,
                'PhysicalActivity': physical_activity,
                'GenHealth': general_health,
                'SleepTime': sleep_time,
                'Asthma': asthma,
                'KidneyDisease': kidney_disease,
                'SkinCancer': skin_cancer}
        
        features = pd.DataFrame(data, index=[0])
        
        return features
    
    input_df = input_features()
    

raw_data = pd.read_csv('./datasets/heart_2020_cleaned.csv')
classes = raw_data.drop(columns=['HeartDisease'])
df = pd.concat([input_df, classes], axis=0)

cols = ["Smoking","AlcoholDrinking","Stroke","DiffWalking","Sex","AgeCategory","Race","Diabetic","PhysicalActivity","GenHealth","Asthma","KidneyDisease","SkinCancer"]

inp = df[:1]
st.subheader('Input features:')
st.write(inp)

le = LabelEncoder()

for col in cols:
    df[col] = le.fit_transform(df[col])

    
df = df[:1]

pca = PCA(n_components=0.95, random_state=42)

def pred():
    if classifier == 'Descision Tree':
        load_clf = pickle.load(open('./models/dt.pkl', 'rb'))
        prediction = load_clf.predict(df)
    elif classifier == 'Random Forest':
        load_clf = pickle.load(open('./models/rf.pkl', 'rb'))
        prediction = load_clf.predict(df)
    elif classifier == 'KNN':
        load_clf = pickle.load(open('./models/knn.pkl', 'rb'))
        prediction = load_clf.predict(df)
    elif classifier == 'Artificial Neural Network':
        load_clf = tf.keras.models.load_model('./models/nn.h5')
        prediction = load_clf.predict(df)
    elif classifier == 'Genitic Algorithm':
        st.subheader('Best features:')
        best_features = ['Smoking', 'Stroke', 'PhysicalHealth', 'MentalHealth', 'DiffWalking',
                        'Sex', 'AgeCategory', 'Race', 'Diabetic', 'PhysicalActivity',
                        'GenHealth', 'KidneyDisease', 'SkinCancer']
        st.write(inp[best_features])
        load_clf = pickle.load(open('./models/genetic.pkl', 'rb'))
        prediction = load_clf.predict(df.loc[:, best_features])
    elif classifier == 'Diffrential Evolution Algorithm':
        st.subheader('Best features:')
        best_features = ['BMI', 'AlcoholDrinking', 'Stroke', 'PhysicalHealth', 'MentalHealth',
                        'DiffWalking', 'Sex', 'AgeCategory', 'Race', 'Diabetic',
                        'PhysicalActivity', 'GenHealth', 'SleepTime', 'KidneyDisease']
        st.write(inp[best_features])
        load_clf = pickle.load(open('./models/evolution.pkl', 'rb'))
        prediction = load_clf.predict(df.loc[:, best_features])
    elif classifier == 'MLP (using PCA)':
        data = pd.read_csv('./datasets/preprocessed_data.csv')
        X = data.drop('HeartDisease', axis=1)
        Y = data['HeartDisease']
        x_train, x_test, y_train, y_test = train_test_split(X, Y, train_size=0.8, random_state=42)
        load_clf = pickle.load(open('./models/mlp.pkl', 'rb'))
        x_train_pca = pca.fit_transform(x_train)
        df_trans = pca.transform(df)
        prediction = load_clf.predict(df_trans)
    else:
        return "Undefined!"
    
    return load_clf, prediction


load_clf, prediction = pred()

st.subheader("Result:")
if prediction == 1:
    st.error("The Patiant Has a Heart Disease")
elif prediction == 0:
    st.success("The Patient Doesn't Have a Heart Disease")
elif type(prediction) == np.ndarray:
    if prediction[0][0] >= 0.2591321:
        st.error("The Patiant Has a Heart Disease")
    elif prediction[0][0] < 0.2591321:
        st.success("The Patient Doesn't Have a Heart Disease")
    

st.subheader("Calculated Accuracies:")

if classifier in clfs[:3]:
    data = pd.read_csv('./datasets/preprocessed_data.csv')
    X = data.drop('HeartDisease', axis=1)
    Y = data['HeartDisease']
    x_train, x_test, y_train, y_test = train_test_split(X, Y, train_size=0.8, random_state=42)
    st.write(f"##### *Training Accuracy:* %{round(load_clf.score(x_train, y_train)*100, 2)}")
    st.write(f"##### *Test Accuracy:* %{round(load_clf.score(x_test, y_test)*100, 2)}")
  
elif classifier == clfs[3]:
    data = pd.read_csv('./datasets/preprocessed_data.csv')
    best_features = ['Smoking', 'Stroke', 'PhysicalHealth', 'MentalHealth', 'DiffWalking',
                    'Sex', 'AgeCategory', 'Race', 'Diabetic', 'PhysicalActivity',
                    'GenHealth', 'KidneyDisease', 'SkinCancer']
    X = data.loc[:, best_features]
    Y = data['HeartDisease']
    x_train, x_test, y_train, y_test = train_test_split(X, Y, train_size=0.8, random_state=42)
    st.write(f"##### *Training Accuracy:* %{round(load_clf.score(x_train, y_train)*100, 2)}")
    st.write(f"##### *Test Accuracy:* %{round(load_clf.score(x_test, y_test)*100, 2)}")

elif classifier == clfs[4]:
    data = pd.read_csv('./datasets/preprocessed_data.csv')
    best_features = ['BMI', 'AlcoholDrinking', 'Stroke', 'PhysicalHealth', 'MentalHealth',
                    'DiffWalking', 'Sex', 'AgeCategory', 'Race', 'Diabetic',
                    'PhysicalActivity', 'GenHealth', 'SleepTime', 'KidneyDisease']
    X = data.loc[:, best_features]
    Y = data['HeartDisease']
    x_train, x_test, y_train, y_test = train_test_split(X, Y, train_size=0.8, random_state=42)
    st.write(f"##### *Training Accuracy:* %{round(load_clf.score(x_train, y_train)*100, 2)}")
    st.write(f"##### *Test Accuracy:* %{round(load_clf.score(x_test, y_test)*100, 2)}")

elif classifier == clfs[-1]:
    data = pd.read_csv('./datasets/preprocessed_data.csv')
    X = data.drop('HeartDisease', axis=1)
    Y = data['HeartDisease']
    x_train, x_test, y_train, y_test = train_test_split(X, Y, train_size=0.8, random_state=42)
    load_clf.fit(x_train, y_train)
    st.write(f"##### *Training Accuracy:* %{round(load_clf.score(x_train, y_train)*100, 2)}")
    st.write(f"##### *Test Accuracy:* %{round(load_clf.score(x_test, y_test)*100, 2)}")
  
  
else:
    data = pd.read_csv('./datasets/preprocessed_data.csv')
    X = data.drop('HeartDisease', axis=1)
    Y = data['HeartDisease']
    x_train, x_test, y_train, y_test = train_test_split(X, Y, train_size=0.8, random_state=42)
    test_loss, test_accuracy = load_clf.evaluate(x_test, y_test)
    train_loss, train_accuracy = load_clf.evaluate(x_train, y_train)
  
    st.write(f"##### Training Accuracy: %{round(train_accuracy*100, 2)}")
    st.write(f"##### Test Accuracy: %{round(test_accuracy*100, 2)}")
