from email.policy import EmailPolicy
import pandas as pd
import streamlit as st
import numpy as np
import matplotlib as plt
import seaborn as sns
import os
from PIL import Image
import pickle
import joblib

def main() :
    df = pd.read_csv('data/stroke.csv')

    add_selectbox = st.sidebar.subheader('뇌졸증 예측 프로그램')

  
    with st.sidebar :
        gender = st.radio('성별을 선택하세요',  ('남자', '여자') )
        if gender == '남자' :
            gender = 1
        elif gender =='여자' :
            gender = 0


        age = st.slider('나이를 입력하세요.',1,100, 29)
        

        col1, col2 = st.columns(2)

        with col1 : hyper_tension = st.checkbox('고혈압',value= False)
        if hyper_tension ==True :
            hyper_tension = 1
        elif hyper_tension == False :
            hyper_tension = 0
            


        with col2 : heart_disease = st.checkbox('심장 질환',value = False)
        if heart_disease == True :
            heart_disease = 1
        elif heart_disease == False :
            heart_disease = 0
       
        

        ever_married = st.radio('결혼 유무',('O','X'))
        if ever_married == 'O' : 
            ever_married = 1
        elif ever_married == 'X' :
            ever_married = 0


        avg_glucose_level = st.number_input('''혈당 수치를 입력해주세요.''')
        if avg_glucose_level == 0 :
            avg_glucose_level = df['avg_glucose_level'].mean()
        st.text('본인의 혈당수치를 모를때에는 비워두세요.')
        

        bmi = st.number_input('BMI 수치를 입력해주세요',15,35,22)


        smoked_status = st.selectbox('흡연 유무', ['없음','흡연 중'])
        if smoked_status == '없음' :
            smoked_status = 0
        else :
            smoked_status = 1

            
    with st.sidebar :
        
        st.text('17.3 Copyright @ 2022 \nDelta Dental of NJ and CT, Inc.')
               
  

    classifier = joblib.load('data/classifier1.pkl')
    scaler_M = joblib.load('data/scaler_M.pkl')
   

    new_data = np.array([gender,age,hyper_tension,heart_disease,ever_married,avg_glucose_level,bmi,smoked_status])
    new_data = new_data.reshape(1,8)
    X = scaler_M.transform(new_data)
    
    
   
    
    y_pred = classifier.predict(X)
    if y_pred == 0 :
        st.subheader('뇌졸증 아닙니다')
    else :
        st.subheader('뇌졸증 위험입니다')
  

if __name__ == '__main__' :
    main()