import requests
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import plotly.express as px
import numpy as np

plt.style.use('fivethirtyeight')
sns.set_style('darkgrid')
#Title display
html_temp = """
<div style="background-color: black; padding:10px; border-radius:10px">
<h1 style="color: white; text-align:center">Dashboard Scoring Credit</h1>
</div>
<p style="font-size: 20px; font-weight: bold; text-align:center">Prédictions de scoring client et comparaison à l'ensemble des clients </p>
"""
st.markdown(html_temp, unsafe_allow_html=True)
@st.cache
#Chargement des données
def load_data():
    data_test = pd.read_csv('data/data_test.csv', index_col='SK_ID_CURR', encoding='utf-8') #
    data_train = pd.read_csv('data/data_train.csv', index_col='SK_ID_CURR', encoding='utf-8')
    X_test = pd.read_csv('data/X_test.csv',index_col='SK_ID_CURR', encoding='utf-8')
    description = pd.read_csv("data/HomeCredit_columns_description.csv",
                              usecols=['Row', 'Description'], index_col=0, encoding='unicode_escape')
    target = data_train.iloc[:, -1:]
    return data_test,data_train,X_test,target, description
#### Chargement des donnés et du modèle de prédiction
data_test,data_train ,X_test,target, description = load_data()
id_client = data_test.index.values

#Loading selectbox ==> choisir l'identifiant
chk_id = st.sidebar.selectbox("Identifiant du client", id_client)

st.header("**Décision - crédit**")
if st.checkbox("Prédiction"):
    response=requests.post("http://127.0.0.1:5000/credit/" + str(chk_id))
    decision=response.text
    #for i in decision:
     #   st.write(i)
    if '1' in decision:
        st.write('Crédit refusé')
    else:
        st.write('Crédit accordé')

else:
    st.markdown("<i>…</i>", unsafe_allow_html=True)