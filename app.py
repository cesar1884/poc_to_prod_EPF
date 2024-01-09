# Path: predict/predict/app.py
import os
import json
import logging
import pandas as pd
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
from flask import Flask, request

import run

app = Flask(__name__)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# create a route to predict
@app.route("/predict", methods=["POST"])
def predict():
    # get the data from the request
    model_path = "/Users/delaygues/Desktop/5A/Poc To Prod/poc-to-prod-capstone/train/data/artefacts/2024-01-09-11-31-10"
    data = request.get_json(force=True)
    data_df = pd.DataFrame(data["text"], columns=["text"])

    # load the model
    model = run.TextPredictionModel.from_artefacts(model_path)
    # predict the data
    predictions = model.predict(data_df["text"])
    # create a response
    response = {"predictions": predictions}
    # return the response
    return json.dumps(response)

# Curl example:
# curl -X POST -H "Content-Type: application/json" -d '{"model_path": "artefacts", "text": ["How to create a list in Python?", "How to convert a pandas dataframe to a numpy array?"]}' http://localhost:5000/predict

# create a route to health check
@app.route("/health", methods=["GET"])
def health():
    return "ok"

# create a hello world route
@app.route("/", methods=["GET"])
def hello():
    return "Hello World!"

# Create a page to test the model with any text
@app.route("/test", methods=["GET"])
def test():
    return """
    <html>
    <body>
        <form id="predictForm">
            <label for="text">Text:</label><br>
            <input type="text" id="text" name="text" value="How to create a list in python?"><br>
            <input type="hidden" id="model_path" name="model_path" value="artefacts">
            <input type="submit" value="Submit">
        </form>

        <script>
            document.getElementById('predictForm').onsubmit = function(event) {
                event.preventDefault();

                // Récupération des valeurs du formulaire
                var textValue = document.getElementById('text').value;
                var modelPathValue = document.getElementById('model_path').value;

                // Création de l'objet JSON à envoyer
                var jsonData = {
                    model_path: modelPathValue,
                    text: [textValue]
                };

                // Création de la requête AJAX
                var xhr = new XMLHttpRequest();
                xhr.open("POST", "/predict", true);
                xhr.setRequestHeader("Content-Type", "application/json");
                xhr.onreadystatechange = function() {
                    if (xhr.readyState == 4 && xhr.status == 200) {
                        // Traitement de la réponse ici
                        console.log(xhr.responseText);
                    }
                };

                // Envoi des données JSON
                xhr.send(JSON.stringify(jsonData));
            };
        </script>
    </body>
    </html>

    """

if __name__ == '__main__':
    app.run(debug=True)
