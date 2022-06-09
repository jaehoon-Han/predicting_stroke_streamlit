import pandas as pd
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import joblib
from run_df import run_df

#차트 한글 깨짐 현상
import matplotlib.font_manager as fm
plt.rcParams['font.family'] = 'Malgun Gothic'
st.set_page_config(layout="wide")

base="dark"
primaryColor="purple"

def main() :
    df = pd.read_csv('data/stroke.csv')
    

    add_selectbox = st.sidebar.title('Prediction ')

  
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
            st.write('----------')
        elif hyper_tension == False :
            hyper_tension = 0
            st.write('----------')
        
        with col1 : ever_married = st.radio('결혼 유무',('O','X'))
        if ever_married == 'O' : 
            ever_married = 1
        elif ever_married == 'X' :
            ever_married = 0
            


        with col2 : heart_disease = st.checkbox('심장 질환',value = False)
        if heart_disease == True :
            heart_disease = 1
        elif heart_disease == False :
            heart_disease = 0

        with col2 : smoked_status = st.radio('흡연 유무', ['없음','흡연 중'])
        if smoked_status == '없음' :
            smoked_status = 0
        else :
            smoked_status = 1

       
        

        

       
        work_type = st.selectbox('근무 유형',['무직','전업주부','자영업','프리랜서','공무원'])

        avg_glucose_level = st.number_input('''혈당 수치를 입력해주세요.''')
        if st.checkbox('혈당수치를 모를때에는 체크',value=False)==True :
            avg_glucose_level = (df['avg_glucose_level'].mean())
        
        

        bmi = st.number_input('BMI 수치를 입력해주세요',15.0,35.0,22.0)
        with st.expander("BMI 측정"):
            Height = st.slider('키',130,200,175)
            Weight = st.slider('몸무게',30,100,65)
            st.write(round((Weight)/(Height/100)**2,1))
           
        


       

        

    classifier = joblib.load('data/classifier1.pkl')
    scaler_M = joblib.load('data/scaler_M.pkl')
   

    new_data = np.array([gender,age,hyper_tension,heart_disease,ever_married,avg_glucose_level,bmi,smoked_status])
    new_data = new_data.reshape(1,8)
    X = scaler_M.transform(new_data)
       
    y_pred = classifier.predict(X)
    with st.sidebar :
        st.write('----------')
        if st.button('예측 실행') :
            if y_pred == 0 :
                st.subheader('뇌졸증 안전 범위입니다.')
                
            else :
                st.subheader('뇌졸증 위험 범위입니다')

        st.write('')
        st.write('')
        st.write('')
        st.write('''17.3 License
        Data files © Original Authors   👨‍⚕FEDESORIANO 
       ''')
  ######### 본문 내용 #######
    
    st.title('EDA + Visualization ')
    st.image('https://healthjournal.uconn.edu/wp-content/uploads/sites/1391/2017/10/featured_brain.jpg')
    with st.container() :
        st.write('세계보건기구(WHO)에 따르면 뇌졸중은 전 세계 사망 원인 2위이며, 전체 사망의 약 11%를 차지한다.\n이 데이터는 성별, 나이, 다양한 질병 및 흡연 상태와 같은 변수를 기반으로 환자가 뇌졸중에 걸릴 가능성을 예측하는 데 사용됩니다.\n데이터의 각 행은 환자에 대한 관련 정보를 제공하며 뇌졸중을 예측하기 위한 총 11개의 임상 기능이 있습니다.')
        col3, col4, col5= st.columns(3)
        with col3 :
            st.write('👨‍⚕   5110 patient')
        with col4 :
            st.write('📊   9 Comparison target')
        with col5 :
            st.write('📌   94.5 accuracy score')
    
    

    run_df()
        
    
        
if __name__ == '__main__' :
    main()