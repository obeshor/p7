from flask import Flask

import pandas as pd
import pickle

app = Flask(__name__)
X_test = pd.read_csv('data/X_test.csv',index_col='SK_ID_CURR', encoding='utf-8') # Données pour la prédiction
pickle_in = open('data/LR_pkl', 'rb') # importation du modèle
clf = pickle.load(pickle_in)

@app.get('/')
def index():
    return("lancement application - Scoring")


#lancement de l'application
if __name__ == "__main__":
    app.run(debug=True)