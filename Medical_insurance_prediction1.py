# -*- coding: utf-8 -*-
"""
Created on Tue Feb 21 22:31:25 2023

@author: saite
"""

import numpy as np 
import streamlit as st
import pickle

loaded_model=pickle.load(open(r"Trained_model_Medical","rb"))

def medical_insurance(input_data):
    data_np_array=np.asarray(input_data)
    data_reshape=data_np_array.reshape(1,-1)
    prediction=loaded_model.predict(data_reshape)
    print(prediction)
    return prediction

def main():
    st.title("Medical cost prediction App")
    
#getting the inputs from the user
Age=st.text_input("Age")
sex=st.text_input("Sex(1=male,0=female)")
bmi=st.text_input("BMI")
children=st.text_input("No of children")
smoker=st.text_input("Smoker or non-smoker(0=non-smoker,1=smoker)")

#code
cost=""
if st.button("Test Result: "):
    cost=medical_insurance([Age,sex,bmi,children,smoker])
    st.success(cost)

if __name__=="__main__":
    main()
