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

def load_model():
    pickle_in = open('data/LR_pkl','rb')
    model = pickle.load(pickle_in)
    return model
@st.cache(allow_output_mutation=True)
def clusters():
    pickle_in = open('data/clustering','rb')
    model = pickle.load(pickle_in)
    return (model)

@st.cache
def load_kmeans(df, id, mdl):
    index = df[df.index == int(id)].index.values
    index = index[0]
    data_client = pd.DataFrame(df.loc[df.index, :])
    df_neighbors = pd.DataFrame(mdl.fit_predict(data_client), index=data_client.index)
    df_neighbors = pd.concat([df_neighbors, data_test.drop(['Unnamed: 0'],axis=1)], axis=1)
    return df_neighbors.iloc[:,1:].sample(5)

def identite_client(data, id):
    data_client = data[data.index == int(id)]
    return data_client
@st.cache
def load_age_population(data):
    data_age = round((data["DAYS_BIRTH"] / 365), 2)
    return data_age

@st.cache
def load_income_population(data):
    df_income = pd.DataFrame(data["AMT_INCOME_TOTAL"])
    df_income = df_income.loc[df_income['AMT_INCOME_TOTAL'] < 200000, :]
    return df_income

#### Chargement des donnés et du modèle de prédiction
data_test,data_train ,X_test,target, description = load_data()
id_client = data_test.index.values
clf = load_model()

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
# description variables
if st.checkbox("Besoin d'aide - description des variables ?"):
    features = description.index.to_list()
    feature = st.selectbox('Feature checklist…', features)
    st.table(description.loc[description.index == feature][:1])

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
#Interpretation - Lime
explainer = pickle.load(open('data/Lime_LR_pkl', 'rb'))
if st.checkbox("Identifiant du client  {:.0f} -  feature importance ?".format(chk_id)):
    x_test=X_test.loc[chk_id]
    explainer_client=explainer.explain_instance(np.array(x_test), clf.predict_proba, num_features=len(X_test.columns))
    fig = explainer_client.as_pyplot_figure()
    st.pyplot(fig)

else:
    st.markdown("<i>…</i>", unsafe_allow_html=True)

chk_voisins = st.checkbox("Clients avec un profil similaire  ?")
if chk_voisins:
    knn = clusters() #modele de clustering
    st.markdown("<u>la liste de 5 clients similaires :</u>", unsafe_allow_html=True)
    st.dataframe(load_kmeans(X_test, chk_id, knn))
    st.markdown("<i>Target 1 = Client avec difficultés de paiment</i>", unsafe_allow_html=True)
else:
    st.markdown("<i>…</i>", unsafe_allow_html=True)
#quelques informations générales
st.header("**Informations du client**")

if st.checkbox("Détails"):
    infos_client = identite_client(data_test, chk_id) #Identité
    st.write(infos_client)
    st.write("**Genre :**",infos_client["CODE_GENDER"].values[0]) #Genre
    st.write("**Age:** {:.0f} ans".format(abs(int(infos_client["DAYS_BIRTH"] / 365))))
    st.write("**Situation familliale :**", infos_client["NAME_FAMILY_STATUS"].values[0])
    st.write("**Nombre d'enfants:** {:.0f}".format(infos_client["CNT_CHILDREN"].values[0]))

# Age distribution plot
    data_age = load_age_population(data_test)
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.histplot(abs(data_age), edgecolor='k', color="red", bins=20)
    ax.axvline(abs(int(infos_client["DAYS_BIRTH"].values / 365)), color="green", linestyle='--')
    ax.set(title='Age des clients', xlabel='Age (Années)', ylabel='')
    st.pyplot(fig)

    st.subheader("*Revenus (USD)*")
    st.write("**Revenus total du client :** {:.0f}".format(infos_client["AMT_INCOME_TOTAL"].values[0]))
    st.write("**Montant du crédit du client :** {:.0f}".format(infos_client["AMT_CREDIT"].values[0]))
    st.write("**Annuités:** {:.0f}".format(infos_client["AMT_ANNUITY"].values[0]))
    st.write("**Montant des biens pour crédit de consommation:** {:.0f}".format(infos_client["AMT_GOODS_PRICE"].values[0]))

    # Income distribution plot
    data_income = load_income_population(data_test)
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.histplot(data_income["AMT_INCOME_TOTAL"], edgecolor='k', color="red", bins=10)
    ax.axvline(int(infos_client["AMT_INCOME_TOTAL"].values[0]), color="green", linestyle='--')
    ax.set(title='Revenu du client', xlabel='Revenu (USD)', ylabel='')
    st.pyplot(fig)

else:
    st.markdown("<i>…</i>", unsafe_allow_html=True)