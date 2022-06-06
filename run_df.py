import pandas as pd
import streamlit as st
import numpy as np
import matplotlib as plt
import seaborn as sns
import os
from PIL import Image

import joblib

def run_df() :
    df = pd.read_csv('data/stroke.csv')
    df.isna().sum()
    df.fillna(df['bmi'].mean(),inplace = True)
    df.drop(['id','work_type','Residence_type'],axis=1,inplace=True)
    X = df.iloc[:,  :-1]
    y = df['stroke']
    df = df[df['gender'] !='Other']
    
    classifier = joblib.load('data/classifier.pkl')
    encoder = joblib.load('data/encoder_label.pkl')

    st.subheader('와우')
    new_data = st.text_input()
   ## new_data = np.array([스트림릿에서 입력받는거])
    X_new = encoder.transform(new_data)
    X_new = X_new.toarray()
    y_pred = classifier.predict(X_new)