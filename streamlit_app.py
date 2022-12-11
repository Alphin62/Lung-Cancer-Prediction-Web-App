# -*- coding: utf-8 -*-
"""
Created on Fri Dec  9 02:23:32 2022

@author: Lucifer
"""

import numpy as np
import pickle
import streamlit as st


st.title(""" Lung Cancer Detection""")
st.image("image.jpg")
st.write("""
         ## About
         
         Lung cancer is caused by the habit of SMOKING. Lung cancers usually are grouped into two main types called small cell and non-small cell. Symptoms includes wheezing, cough (with blood), shortness of breath, swallowing difficulty, chest pain. Different people have different symptoms for lung cancer. Most people with lung cancer don’t have symptoms until the cancer is advanced. Treatments vary but may include surgery, chemotherapy, radiation therapy, targeted drug therapy and immunotherapy.
             
         **This Streamlit App utilizes a Machine Learning API in order to detect lung cancer in patients based on the following criteria: age, gender, blood pressure, smoke, coughing, allergies, fatigue etc.**
             
         **By Alphin Gnanaraj I**
         """)

loaded_model = pickle.load(open("trained_model.sav","rb"))

# creating a function for prediction

def lung_cancer_prediction(input_data):
       
    # change the i/p data to np array
    input_data_as_np_array = np.asarray(input_data)

    # reshape the array as we are predicting for one instance
    input_data_reshaped = input_data_as_np_array.reshape(1,-1)

    prediction = loaded_model.predict(input_data_reshaped)
    
    print(prediction)

    if (prediction[0] == 0):
        return 'Chance of having lung cancer is low. Kindly be aware'
    else:
        return 'Very high chance of having lung cancer. Need immediate checkup !!!!!'


# creating a side bar for user input features   
      
st.sidebar.header('Patients input features')
    

# creating the user input features 
    
gender = st.sidebar.number_input("GENDER: Enter 1 for Male and 0 for Female", min_value=0, max_value=1)
age = st.sidebar.slider("AGE: Enter your Age", min_value=1, max_value=100)
smoking = st.sidebar.number_input("SMOKING: Enter 1 if you smoke or 0 if you don't smoke", min_value=0, max_value=1)
yellow_finger = st.sidebar.number_input("YELLOW FINGERS: Enter 1 if you have yellow fingers or 0 if you don't", min_value=0, max_value=1)
anxiety = st.sidebar.number_input("ANXIETY: Enter 1 if you have anxiety and 0 if you don't", min_value=0, max_value=1)
peer = st.sidebar.number_input("PEER PRESSURE: Enter 1 if you feel you suffer from peer pressure or 0 if you don't", min_value=0, max_value=1)
chronic = st.sidebar.number_input("CHRONIC DISEASE: Enter 1 if you suffer from a chronic disease or O if you don't", min_value=0, max_value=1)
fatigue = st.sidebar.number_input("FATIGUE: Enter 1 if you have fatigue or 0 if you don't", min_value=0, max_value=1)
allergy = st.sidebar.number_input("ALLERGY: Enter 1 if you have some sort of allergy or 0 if you don't", min_value=0, max_value=1)
wheezing = st.sidebar.number_input("WHEEZING: Enter 1 if you wheeze or 0 if you don't", min_value=0, max_value=1)
alcohol =  st.sidebar.number_input("ALCOHOL CONSUMPTION: Enter 1 if you consume alcohol or 0 if you don't", min_value=0, max_value=1)
coughing = st.sidebar.number_input("COUGHING: Enter 1 if you cough a lot or 0 if you don't", min_value=0, max_value=1)
breath = st.sidebar.number_input("SHORTNESS OF BREATH: Enter 1 if you suffer from shortness of breath or 0 if you don't", min_value=0, max_value=1)
swallow =  st.sidebar.number_input("SWALLOWING DIFFICULTY: Enter 1 if you have difficulty swallowing or 0 if you don't", min_value=0, max_value=1)
chest =  st.sidebar.number_input("CHEST PAIN: Enter 1 if you have chest pain or 0 if you don't", min_value=0, max_value=1)


# code for prediction

diagnosis = ''

# create button

if st.button('Detection Result'):
    diagnosis = lung_cancer_prediction([gender,age,smoking,yellow_finger,anxiety,peer,chronic,fatigue,allergy,wheezing,alcohol,coughing,breath,swallow,chest]) 
    
    st.success(diagnosis)
              
