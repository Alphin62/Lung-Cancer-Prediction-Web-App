# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import pickle
import warnings
warnings.filterwarnings('ignore')

loaded_model = pickle.load(open('C:/Users/Lucifer.LAPTOP-19ER5RAR/Desktop/Internship/Assignment3/Deploy ML model/trained_model.sav','rb'))

input_data = (0,59,0,0,0,0,0,0,1,1,0,0,0,0,1)

# change the i/p data to np array
input_data_as_np_array = np.asarray(input_data)

# reshape the array as we are predicting for one instance
input_data_reshaped = input_data_as_np_array.reshape(1,-1)

prediction = loaded_model.predict(input_data_reshaped)
print(prediction)

if (prediction[0] == 0):
    print('Chance of having lung cancer is low. Kindly be aware')
else:
    print('Very high chance of having lung cancer. Need immediate checkup !!!!!')