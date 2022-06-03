import pandas as pd
import streamlit as st
import numpy as np
import matplotlib as plt
import seaborn as sns
import os
from PIL import Image



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
            'test text',
            ('radio test1', 'radio test2')
        )
        st.slider('slider test',10,100)

        st.checkbox('test checkbox',value= False)

    with st.sidebar :
        st.text('''17.3 Copyright @ 2022
Delta Dental of NJ and CT, Inc.''')
        
        

if __name__ == '__main__' :
    main()