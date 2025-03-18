from flask import Flask, request, jsonify
import joblib
import numpy as np
import xgboost as xgb
from pydantic import ValidationError
from model import PredictionInput
from flask_swagger_ui import get_swaggerui_blueprint

app = Flask(__name__)

# Charger le modèle correctement avec Booster.load_model()
model = xgb.Booster()
model.load_model("xgb_model.json")

# Configurer Swagger UI
SWAGGER_URL = "/docs"  # URL où Swagger UI sera accessible
API_URL = "/static/swagger.json"  # Fichier de définition OpenAPI
swagger_ui_blueprint = get_swaggerui_blueprint(SWAGGER_URL, API_URL)
app.register_blueprint(swagger_ui_blueprint, url_prefix=SWAGGER_URL)

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Récupérer et valider les données JSON
        data = request.get_json()
        validated_data = PredictionInput(**data)

        # Définir les noms des features (ceux utilisés lors de l'entraînement du modèle)
        feature_names = [
            "age", "location", "race:AfricanAmerican", "race:Asian", "race:Caucasian",
            "race:Hispanic", "race:Other", "hypertension", "heart_disease", "smoking_history",
            "bmi", "hbA1c_level", "blood_glucose_level", "Female", "Male"
        ]

        # Supposons que validated_data est une instance de PredictionInput
        features = np.array([[
            validated_data.age,
            validated_data.location,
            validated_data.race_african_american,
            validated_data.race_asian,
            validated_data.race_caucasian,
            validated_data.race_hispanic,
            validated_data.race_other,
            validated_data.hypertension,
            validated_data.heart_disease,
            validated_data.smoking_history,
            validated_data.bmi,
            validated_data.hbA1c_level,
            validated_data.blood_glucose_level,
            validated_data.Female,
            validated_data.Male
        ]])

        # Créer DMatrix avec les noms de features
        dmatrix = xgb.DMatrix(features, feature_names=feature_names)

        # Faire la prédiction
        prediction = model.predict(dmatrix)[0]

        # Retourner le résultat
        return jsonify({"prediction": float(prediction)})

    except ValidationError as e:
        return jsonify({"error": e.errors()}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)