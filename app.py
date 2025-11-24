from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np

app = Flask(__name__)
CORS(app)  # Permite llamadas desde cualquier frontend

# Cargar modelo entrenado
modelo = joblib.load("mi_modelo.pkl")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()

        # Convertimos todos los campos a float
        nivel = float(data.get('nivel', 0))
        nota = float(data.get('nota', 0))
        situacion = float(data.get('situacion', 0))
        motivacion = float(data.get('motivacion', 0))
        responsabilidad = float(data.get('responsabilidad', 0))

        # Crear array para el modelo
        X = np.array([[nivel, nota, situacion, motivacion, responsabilidad]])
        pred = modelo.predict(X)

        return jsonify({'prediccion': float(pred[0])})

    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == "__main__":
    app.run(debug=True)







