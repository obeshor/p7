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

@st.cache
def load_infos_gen(data):
    lst_infos = [data.shape[0],
                    round(data["AMT_INCOME_TOTAL"].mean(), 2),
                    round(data["AMT_CREDIT"].mean(), 2)]

    nb_credits = lst_infos[0] #Nombre de clients
    rev_moy = lst_infos[1] #Revenus moyen
    credits_moy = lst_infos[2] # montant moyen de crédit

    targets = data.TARGET.value_counts()
    return nb_credits, rev_moy, credits_moy, targets

@st.cache
def chargement_ligne_data(id, df):
    return df[df.index==int(id)].drop(['Unnamed: 0'], axis=1)
#### Chargement des donnés et du modèle de prédiction
data_test,data_train ,X_test,target, description = load_data()
id_client = data_test.index.values

#Loading selectbox ==> choisir l'identifiant
chk_id = st.sidebar.selectbox("Identifiant du client", id_client)
#Loading general info ==> Calculs de quelques informations générales
nb_credits, rev_moy, credits_moy, targets = load_infos_gen(data_train)

st.sidebar.markdown("<u>**Nombre totale de client :**</u>", unsafe_allow_html=True)
st.sidebar.text(nb_credits)

# Average income
st.sidebar.markdown("<u>**Revenu moyen (USD) :**</u>", unsafe_allow_html=True)
st.sidebar.text(rev_moy)

# AMT CREDIT
st.sidebar.markdown("<u>**Montant de crédit moyen (USD) :**</u>", unsafe_allow_html=True)
st.sidebar.text(credits_moy)
# PieChart
st.sidebar.markdown("<u>**Répartition des clients:**</u>", unsafe_allow_html=True)
fig, ax = plt.subplots(figsize=(5, 5))
plt.pie(targets, explode=[0, 0.1], labels=['Sans difficultés de paiement', 'Difficultés de paiement'] ,colors=['red','green'],autopct='%1.1f%%', startangle=90)
st.sidebar.pyplot(fig)

#######################################
# Page d'accueil
#######################################
#Identifiant et données du client
st.write("Identifiant du client :", chk_id)
df_client = chargement_ligne_data(chk_id, data_test)
st.write(df_client)

#Prédiction
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