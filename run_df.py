import warnings 
warnings.filterwarnings('ignore')

import streamlit as st

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#Common model helpers
from sklearn.preprocessing import (StandardScaler,
                                   LabelEncoder,
                                   OneHotEncoder)




def run_df() :
    
    df = pd.read_csv('data/stroke.csv')
    fig = plt.figure(figsize=(20,20))
    gs = fig.add_gridspec(3,4)
    gs.update(wspace=0.3, hspace=0.15)
    ax0 = fig.add_subplot(gs[0,0])
    ax1 = fig.add_subplot(gs[0,1])
    ax2 = fig.add_subplot(gs[0,2])
    ax3 = fig.add_subplot(gs[1,1])
    ax4 = fig.add_subplot(gs[1,2])
    ax5 = fig.add_subplot(gs[-2,0])
    ax6 = fig.add_subplot(gs[1,3])
    ax7 = fig.add_subplot(gs[0,3])

    background_color = "#1b1d21"
    COLOR = 'white'
    plt.rcParams['xtick.color'] = COLOR
    plt.rcParams['ytick.color'] = COLOR

    fig.patch.set_facecolor(background_color) 
    ax0.set_facecolor(background_color) 
    ax1.set_facecolor(background_color) 
    ax2.set_facecolor(background_color) 
    ax3.set_facecolor(background_color) 
    ax4.set_facecolor(background_color) 
    ax5.set_facecolor(background_color) 
    ax6.set_facecolor(background_color) 
    ax7.set_facecolor(background_color) 

    # Title of the plot
    ax0.text(0.5,0.5,"Counting Categorical features \nand checking stroke\n among various categories\n___________",
            horizontalalignment = 'center',
            verticalalignment = 'center',
            fontsize = 16,
            fontweight='bold',
            fontfamily='serif',
            color='#ffffff')

    ax0.set_xticklabels([])
    ax0.set_yticklabels([])
    plt.rcParams['xtick.color'] = COLOR
    plt.rcParams['ytick.color'] = COLOR
    ax0.tick_params(left=False, bottom=False)

    ###
    '','', '', '', '', 'Residence_type', 'smoking_status'

    ax1.grid(color='#ffffff', linestyle=':', axis='y', zorder=0,  dashes=(1,5))
    sns.countplot(ax=ax1, data=df, x='gender',palette = 'Reds', hue='stroke',edgecolor='black')
   
    ax1.set_xlabel("gender",fontsize=14, fontweight='bold', fontfamily='serif', color="#ffffff")
    ax1.set_ylabel("",fontsize=14, fontweight='bold', fontfamily='serif', color="#ffffff")

    ax2.grid(color='#ffffff', linestyle=':', axis='y', zorder=0,  dashes=(1,5))
    sns.countplot(ax=ax2, data=df, x='hypertension',palette = 'Reds', hue='stroke',edgecolor='black')
    ax2.set_xlabel("hypertension",fontsize=14, fontweight='bold', fontfamily='serif', color="#ffffff")
    ax2.set_ylabel("",fontsize=14, fontweight='bold', fontfamily='serif', color="#ffffff")

    ax3.grid(color='#ffffff', linestyle=':', axis='y', zorder=0,  dashes=(1,5))
    sns.countplot(ax=ax3, data=df, x='heart_disease',palette = 'Reds',hue='stroke',edgecolor='black')
    ax3.set_xlabel("heart_disease",fontsize=14, fontweight='bold', fontfamily='serif', color="#ffffff")
    ax3.set_ylabel("",fontsize=14, fontweight='bold', fontfamily='serif', color="#ffffff")

    ax4.grid(color='#ffffff', linestyle=':', axis='y', zorder=0,  dashes=(1,5))
    sns.countplot(ax=ax4, data=df, x='ever_married',palette = 'Reds',hue='stroke',edgecolor='black')
    ax4.set_xlabel("ever_married",fontsize=14, fontweight='bold', fontfamily='serif', color="#ffffff")
    ax4.set_ylabel("",fontsize=14, fontweight='bold', fontfamily='serif', color="#ffffff")

    ax5.grid(color='#ffffff', linestyle=':', axis='y', zorder=0,  dashes=(1,5))
    sns.countplot(ax=ax5, data=df, x='work_type',palette = 'Reds',hue='stroke',edgecolor='black')
    ax5.set_xlabel("work_type",fontsize=14, fontweight='bold', fontfamily='serif', color="#ffffff")
    ax5.set_ylabel("",fontsize=14, fontweight='bold', fontfamily='serif', color="#ffffff")

    ax6.grid(color='#ffffff', linestyle=':', axis='y', zorder=0,  dashes=(1,5))
    sns.countplot(ax=ax6, data=df, x='Residence_type',palette = 'Reds',hue='stroke',edgecolor='black')
    ax6.set_xlabel("Residence_type",fontsize=14, fontweight='bold', fontfamily='serif', color="#ffffff")
    ax6.set_ylabel("",fontsize=14, fontweight='bold', fontfamily='serif', color="#ffffff")

    ax7.grid(color='#ffffff', linestyle=':', axis='y', zorder=0,  dashes=(1,5))
    sns.countplot(ax=ax7, data=df, x='smoking_status',palette = 'Reds',hue='stroke',edgecolor='black')
    ax7.set_xlabel("smoking_status",fontsize=14, fontweight='bold', fontfamily='serif', color="#ffffff")
    ax7.set_ylabel("",fontsize=14, fontweight='bold', fontfamily='serif', color="#ffffff")

    ax0.spines["top"].set_visible(False)
    ax0.spines["left"].set_visible(False)
    ax0.spines["bottom"].set_visible(False)
    ax0.spines["right"].set_visible(False)

    ax1.spines["top"].set_visible(False)
    ax1.spines["left"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    ax2.spines["top"].set_visible(False)
    ax2.spines["left"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    ax3.spines["top"].set_visible(False)
    ax3.spines["left"].set_visible(False)
    ax3.spines["right"].set_visible(False)

    ax4.spines["top"].set_visible(False)
    ax4.spines["left"].set_visible(False)
    ax4.spines["right"].set_visible(False)

    ax5.spines["top"].set_visible(False)
    ax5.spines["left"].set_visible(False)
    ax5.spines["right"].set_visible(False)

    ax6.spines["top"].set_visible(False)
    ax6.spines["left"].set_visible(False)
    ax6.spines["right"].set_visible(False)

    ax7.spines["top"].set_visible(False)
    ax7.spines["left"].set_visible(False)
    ax7.spines["right"].set_visible(False)
    st.pyplot(fig)

    with st.expander('????????????? ????????????'):
            st.write('gender = ??????')
            st.write('hypertension = ????????? [ 0 = ?????? ??????, 1 = ????????? ]')
            st.write('smoking status = ?????? ??????')
            st.write('work type = ?????? ??????')
            st.write('heart disease = ?????? ?????? [ 0 = ??????, 1 = ??????????????? ?????????, ?????? ????????? ???????????? ?????? ]')
            st.write('ever married = ?????? ??????')
            st.write('Residence type = ?????? ??????')
            
    #st.write('''EDA??? ?????????:
        #?????? ?????? ???????????? ????????? ????????? ??? ?????? ?????? ??????, ????????? ?????? ?????? ????????? ?????? ?????? ??????. \n
        #?????? ????????? ?????? ????????? ???????????? ??? ?????? ???????????? ??? ?????? ?????? ????????? ????????? ??????. \n
        #????????? ?????????????????? ????????? ?????? ????????? ?????? ??????. \n
        #????????? ?????? ?????? ????????? ?????? ????????? ??????.????????? ????????? ????????? ????????? ????????????. \n
        #???????????? ????????? ?????? ?????? ??????????????? ???????????? \n 
        #????????? ???????????? ?????? ?????? ??????????????? ????????? ??????. \n 
        #???????????? ????????? ????????? ????????? ?????? ????????? ???????????? ?????????, ????????? ???????????? ???????????? ????????? ????????? ??? ????????? ?????? ???????????? 60-80??? ????????? ???????????? ?????? ????????????. \n
        #????????? ????????? avg_glucose_level??? 120?????? ????????????. \n 
        #?????? ????????? ????????? ????????? ???????????? ?????? ????????? ??? ?????????, ????????? ????????? ???????????? ????????? ???????????? ??? ????????? ????????? ?????????.''')

######################### 2?????? ?????? ###########################
    with st.expander('???????????????? ????????? ???????????? ??????') :
        fig2 = plt.figure(figsize=(18,18))
        gs = fig2.add_gridspec(5,2)
        gs.update(wspace=0.5, hspace=0.5)
        ax0 = fig2.add_subplot(gs[0,0])
        ax1 = fig2.add_subplot(gs[0,1])
        ax2 = fig2.add_subplot(gs[1,0])
        ax3 = fig2.add_subplot(gs[1,1])
        ax4 = fig2.add_subplot(gs[2,0])
        ax5 = fig2.add_subplot(gs[2,1])

        background_color = "#1b1d21"
        fig2.patch.set_facecolor(background_color) 
        ax0.set_facecolor(background_color) 
        ax1.set_facecolor(background_color) 
        ax2.set_facecolor(background_color)
        ax3.set_facecolor(background_color)
        ax4.set_facecolor(background_color)
        ax5.set_facecolor(background_color) 

        # Age title
        ax0.text(0.5,0.5,"Distribution of age\naccording to\n target variable\n___________",
                horizontalalignment = 'center',
                verticalalignment = 'center',
                fontsize = 18,
                fontweight='bold',
                fontfamily='serif',
                color='#ffffff')
        ax0.spines["bottom"].set_visible(False)
        ax0.set_xticklabels([])
        ax0.set_yticklabels([])
        ax0.tick_params(left=False, bottom=False)

        # Age
        ax1.grid(color='#000000', linestyle=':', axis='y', zorder=0,  dashes=(1,5))
        sns.kdeplot(ax=ax1, data=df, x='age',hue="stroke", fill=True,palette="Reds", alpha=.5, linewidth=0)
        ax1.set_xlabel("")
        ax1.set_ylabel("")

        # bmi title
        ax2.text(0.5,0.5,"Distribution of bmi\naccording to\n target variable\n___________",
                horizontalalignment = 'center',
                verticalalignment = 'center',
                fontsize = 18,
                fontweight='bold',
                fontfamily='serif',
                color='#ffffff')
        ax2.spines["bottom"].set_visible(False)
        ax2.set_xticklabels([])
        ax2.set_yticklabels([])
        ax2.tick_params(left=False, bottom=False)

        # bmi
        ax3.grid(color='#000000', linestyle=':', axis='y', zorder=0,  dashes=(1,5))
        sns.kdeplot(ax=ax3, data=df, x='bmi',hue="stroke", fill=True,palette="Reds", alpha=.5, linewidth=0)
        ax3.set_xlabel("")
        ax3.set_ylabel("")

        # avg_glucose_level title
        ax4.text(0.5,0.5,"Distribution of avg_glucose_level\naccording to\n target variable\n___________",
                horizontalalignment = 'center',
                verticalalignment = 'center',
                fontsize = 18,
                fontweight='bold',
                fontfamily='serif',
                color='#ffffff')
        ax4.spines["bottom"].set_visible(False)
        ax4.set_xticklabels([])
        ax4.set_yticklabels([])
        ax4.tick_params(left=False, bottom=False)

        # avg_glucose_level
        ax5.grid(color='#000000', linestyle=':', axis='y', zorder=0,  dashes=(1,5))
        sns.kdeplot(ax=ax5, data=df, x='avg_glucose_level',hue="stroke", fill=True,palette="Reds", alpha=.5, linewidth=0)
        ax5.set_xlabel("")
        ax5.set_ylabel("")



        for i in ["top","left","right"]:
            ax0.spines[i].set_visible(False)
            ax1.spines[i].set_visible(False)
            ax2.spines[i].set_visible(False)
            ax3.spines[i].set_visible(False)
            ax4.spines[i].set_visible(False)
            ax5.spines[i].set_visible(False)
        st.pyplot(fig2)


    with st.expander('????Correlation Heatmap') :
        fig3 = plt.figure(figsize=(35,25))
        gs = fig3.add_gridspec(3,4)
        gs.update(wspace=0.3, hspace=0.15)
        ax0 = fig3.add_subplot(gs[0,0])
        ax1 = fig3.add_subplot(gs[0,1])

        background_color = "#1b1d21"

        fig3.patch.set_facecolor(background_color) 
        ax0.set_facecolor(background_color) 
        ax1.set_facecolor(background_color) 

        # Title of the plot
        ax0.text(0.5,0.5,"Correlation Heatmap\n___________",
                horizontalalignment = 'center',
                verticalalignment = 'center',
                fontsize = 16,
                fontweight='bold',
                fontfamily='serif',
                color='#ffffff')

        ax0.set_xticklabels([])
        ax0.set_yticklabels([])
        ax0.tick_params(left=False, bottom=False)

        ax1.grid(color='#000000', linestyle=':', axis='y', zorder=0,  dashes=(1,5))
        sns.heatmap(df.corr(), vmin=-1, vmax=1, annot=True, cmap='BrBG')

        ax0.spines["top"].set_visible(False)
        ax0.spines["left"].set_visible(False)
        ax0.spines["bottom"].set_visible(False)
        ax0.spines["right"].set_visible(False)

        ax1.spines["top"].set_visible(False)
        ax1.spines["left"].set_visible(False)
        ax1.spines["right"].set_visible(False)
        st.pyplot(fig3)

