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

    add_selectbox = st.sidebar.selectbox(
        'how would',
        ('test','test2','test3')
    )

    with st.sidebar :
        col1, col2 = st.columns(2)
        with col1 : st.button('button test 1')
        with col2 : st.button('button test 2')


    with st.sidebar :
        add_radio = st.radio(
            '성별을 선택하세요',
            ('남자', '여자')
        )
        age = st.slider('slider test',10,100)

        st.checkbox('test checkbox',value= False)

    with st.sidebar :
        
        st.text('''17.3 Copyright @ 2022
Delta Dental of NJ and CT, Inc.''')
        
    st.title('뇌졸증 예측')
    
    st.dataframe(df)
    

    classifier = joblib.load('data/classifier.pkl')
    encoder = joblib.load('data/encoder_label.pkl')

    st.subheader('와우,와우')
    new_data = st.text_input('입력하세요')
    new_data = np.array([new_data])
   ## new_data = np.array([스트림릿에서 입력받는거])
    X_new = encoder.transform(new_data)
    X_new = X_new.toarray()
    y_pred = classifier.predict(X_new)


## 'gender',	'age'	,'hypertension'	,'heart_disease',	'ever_married',	'avg_glucose_level',	'bmi'	,'smoking_status_0',	'smoking_status_1',	'smoking_status_2',	'smoking_status_3'
if __name__ == '__main__' :
    main()