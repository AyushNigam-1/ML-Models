import streamlit as st
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler,LabelEncoder,OneHotEncoder
import pandas as pd
import pickle

model = tf.keras.models.load_model('model.h5')

with open('label_encoder.pkl','rb') as f:
    label_encoder = pickle.load(f)

with open('onehot_encoder_geo.pkl','rb') as f:
    onehot_encoder_geo = pickle.load(f)

with open('scaler.pkl','rb') as f:
    scaler = pickle.load(f)


geography = st.selectbox('Geography',onehot_encoder_geo.categories_[0])
gender = st.selectbox('Gender',label_encoder.classes_)
age = st.slider('Age',18,92)
balance =st.number_input('Balance')
credit_score = st.number_input('Credit Score')
estimated_salary = st.number_input('Estimated Salary')
tenure = st.slider('Tenure',0,10)
num_of_products = st.slider('Number of Products',1,4)
has_cr_card = st.selectbox('Has Credit Card',[0,1])
is_active_member = st.selectbox('Is Active Member',[0,1])


input_data = pd.DataFrame({
    'CreditScore':[credit_score],
    # 'Geography':[label_encoder.transform([gender])[0]],
    'Gender':[label_encoder.transform([gender])[0]],
    'Age':[age],
    'Tenure':[tenure],
    'Balance':[balance],
    'NumOfProducts':[num_of_products],
    'HasCrCard':[has_cr_card],
    'IsActiveMember':[is_active_member],
    'EstimatedSalary':[estimated_salary]
})

geo_encoded = onehot_encoder_geo.transform([[geography]]).toarray()
geo_encoded_df = pd.DataFrame(geo_encoded,columns=onehot_encoder_geo.get_feature_names_out(['Geography']))

input_data = pd.concat([input_data.reset_index(drop=True),geo_encoded_df],axis=1)

input_scaled = scaler.transform(input_data)
# input_scaled

predicition = model.predict(input_scaled)
# predicition

prediction_proba = predicition[0][0]

if prediction_proba > 0.5:
  st.write('It"s a churner')
else:
  st.write('It"s not a churner')
