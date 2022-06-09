### ○ 프로젝트 개요

- **프로젝트 주제**
    
    뇌졸증 환자의 표본을 분석하여, 예측하는 인공지능을 만들고 적용하는 프로젝트
    
- **프로젝트 개요**
    - 프로젝트 목표
        
        데이터의 가공 및 분석과 그에 맞는 인공지능을 적용하기
        
    - 구현 내용
        - EDA
            - Jupyter Notebook을 이용하여 데이터 특성 분석
        - Data Processing
            - 문자와 숫자값을 preprocessing 하여 인공지능에 활용할수 있게끔 가공
        - Classifier 모델을 사용하여 인공지능 생성
            - 모델 중 정확도가 가장 높게 평가된 모델을 사용( GaussianNB )
        - 데이터의 시각화
        
    - 사전 구축된 대규모 데이터를 이용해 인공지능 활용
        

- **활용 장비 및 재료**(개발 환경, 협업 tool 등)
    - 서버환경 : Python 3.7.13
    - 개발툴 : vscode, jupyter notebook
    - 협업툴 : Git, putty, streamlit
    - 
-  사용된 라이브러리 
    - pandas
    - streamlit
    - numpy
    - matplolib.pyplot
    - seaborn
    - joblib
    - warning
    - sklearn



### ○ 프로젝트 수행 절차 및 방법

- **프로젝트 개발 Process**
    
    개발 과정을 아래와 같이 크게 3가지 파트로 분류함.
    
    - EDA : Jupyter Notebook을 이용하여 데이터 특성 및 이상치 분석
    - Data Processing : 모델 학습에 유용한 형태로 데이터를 처리
    - Modeling : 모델을 구현하고 성능 향상을 위해 Parameter Tunning 및 GridSearchCV 사용
   


### ○ 프로젝트 수행 결과

1. **EDA**
    - 데이터셋 구성


        ![Untitled (1)](https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2Fts8ml%2FbtrEgwzaOTD%2FuUkm3Pxy72NmKiYrVTwgo0%2Fimg.png)

  
2. **Machine Learning**
    - Classifier 예측


        ![Untitled (2)](https://blog.kakaocdn.net/dn/cFjYma/btrEiIlB6eh/wPdwJ24oU1NYaVM0jyYXy1/img.png)

       
  


 
    

1. **데이터 처리**
    - 전처리
        - 각 컬럼별 NaN 값 확인 후 평균값으로 대체



        - Label Encoder를 통해 문자열 데이터를 숫자로 변환
        - MinMax Scaler를 사용해 숫자의 범위를 조정
              
        - 성능 비교
        
            - 다양한 모델 중 성능이 가장 뛰어난 MultinomialNB 모델을 사용
 ![Untitled (1)](https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FLbxxg%2FbtrD8jIvcAW%2FGR19JUsdKfK5Ctvsr0Yky1%2Fimg.png)

### Connect URL
http://ec2-3-35-49-86.ap-northeast-2.compute.amazonaws.com:8503/

### DataSet
Stroke Prediction Dataset

https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset/code
