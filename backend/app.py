from flask import Flask, request, jsonify
import joblib
import pandas as pd
import pandas as pd
from flask_cors import CORS
app = Flask(__name__)
CORS(app)  # Autorise CORS
# Charger le modèle sauvegardé
model_path = 'best_model.pkl'
best_model = joblib.load(model_path)

# Dictionnaire pour l'encodage des valeurs
encoding_dict = {
    "Parental_Involvement": {"Low": 0, "Medium": 1, "High": 2},
    "Access_to_Resources": {"Low": 0, "Medium": 1, "High": 2},
    "Extracurricular_Activities": {"No": 0, "Yes": 1},
    "Motivation_Level": {"Low": 0, "Medium": 1, "High": 2},
    "Internet_Access": {"No": 0, "Yes": 1},
    "Family_Income": {"Low": 0, "Medium": 1, "High": 2},
    "Teacher_Quality": {"Low": 0, "Medium": 1, "High": 2},
    "School_Type": {"Public": 0, "Private": 1},
    "Peer_Influence": {"Negative": 0, "Neutral": 1, "Positive": 2},
    "Learning_Disabilities": {"No": 0, "Yes": 1},
    "Parental_Education_Level": {"High School": 0, "College": 1, "Postgraduate": 2},
    "Distance_from_Home": {"Far": 0, "Moderate": 1, "Near": 2},
    "Gender": {"Male": 0, "Female": 1}
}

@app.route('/predict', methods=['POST'])
def predict():
    # Récupérer les données de la requête
    data = request.get_json()

    # Encoder les valeurs catégorielles
    for key in encoding_dict:
        if key in data:
            data[key] = encoding_dict[key].get(data[key], -1)  # -1 pour les valeurs inconnues

    # Convertir les données en DataFrame
    X = pd.DataFrame([data])

    # Prédire le score d'examen
    y_pred = best_model.predict(X)

    # Retourner le résultat
    return jsonify({'predicted_score': y_pred[0]})

if __name__ == '__main__':
    app.run(debug=True)
