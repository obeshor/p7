from flask import Flask
import gunicorn
import pandas as pd
import pickle

app = Flask(__name__)
X_test = pd.read_csv('data/X_test.csv',index_col='SK_ID_CURR', encoding='utf-8') # Données pour la prédiction
pickle_in = open('data/LR_pkl', 'rb') # importation du modèle
clf = pickle.load(pickle_in)

@app.get('/')
def index():
    return("lancement application - Scoring")

@app.post('/credit/<id_client>')
def credit(id_client):
    pred=clf.predict(X_test[X_test.index == id_client])
    dict_final = {

       'pred': str(pred),
    }
    return(dict_final)
    #return(pred)
#lancement de l'application
if __name__ == "__main__":
    app.run(debug=True)