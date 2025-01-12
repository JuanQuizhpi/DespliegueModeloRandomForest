from flask import Flask, request, jsonify 
import pickle
import numpy as np
import pandas as pd
import firebase_admin
from firebase_admin import credentials, firestore


# Inicializar la aplicacion Flask
app = Flask(__name__)

#Configurar las credenciales a partir del archivo JSON
cred = credentials.Certificate('fir-app-2bfcb-firebase-adminsdk-66op5-592e0b8a2b.json')  # Reemplaza con la ruta de tu archivo
firebase_admin.initialize_app(cred)

# Conectar a Firestore
db = firestore.client()

#Cargamos el modelo y el pipeline desde los archivos .pickle
with open('./model/modeloRF.pickle', 'rb') as model_file:
    model = pickle.load(model_file)
with open('./model/pipePreprocesadores.pickle', 'rb') as pipeline_file:
    pipeline = pickle.load(pipeline_file)

@app.route('/')
def home():
    return "Api para la prediccion con modelo Random Forest Funcionando Correctamente"

@app.route('/predict' , methods= ['POST'])
def predict():
    try:
        #Obtener datos JSON enviados por el cliente
        data = request.json

        #Validar los datos
        if not data:
            return jsonify({"error":"No se enviaron datos."}),400
        
        #Transforamar los datos usando el pipeline
        #input_data = [data] # Verificamos que sea una lista
        input_data = pd.DataFrame([data])
        transformed_data = pipeline.transform(input_data)
        
        #Hacer la prediccion con el modelo
        prediction = model.predict(transformed_data)
        prediction_proba = model.predict_proba(transformed_data)
        
        # Preparar los datos para guardar en Firebase
        prediction_result = {
            "input_data": data,  # Los datos originales enviados por el usuario
            "prediction": int(prediction[0]),  # Predicción (0 o 1)
            "probability": prediction_proba[0].tolist()  # Probabilidades de cada clase
        }
        
        # Guardar en Firestore (colección "predictions")
        db.collection("predictions").add(prediction_result)
        
        #Devolvemos la respuesta en formato JSON
        return jsonify({
            #"prediction": int(prediction[0]),
            #"probability": prediction_proba[0].tolist()
            "prediction": prediction_result["prediction"],
            "probability": prediction_result["probability"]
        })
    except Exception as e:
        return jsonify({"error":str(e)}),500
    
@app.route('/history',methods=['GET'])
def get_history():
    try:
        #Obtener las predicciones desde firebase
        predictions_ref = db.collection("predictions")
        docs = predictions_ref.stream()
        
        #Formatear las predicciones en una lista
        history=[]
        for doc in docs:
            record = doc.to_dict()
            record["id"]= doc.id 
            history.append(record)
        
        #Retornar el historial como JSON
        return jsonify({"history":history}),200
    except Exception as e:
        return jsonify({"error": str(e)}),500
    
@app.route('/history/<id>', methods=['GET'])
def get_prediction_by_id(id):
    try:
        prediction_ref= db.collection("predictions").document(id)
        doc = prediction_ref.get()
        
        #Verificar si el documento existe
        if doc.exists:
            prediction = doc.to_dict()
            prediction["id"]= doc.id
            return jsonify(prediction),200
        else:
            return jsonify({"error":f"No se encontro una prediccion con ID: {id}"}),404
    except Exception as e:
        return jsonify({"error": str(e)}),500
#Corremos la aplicacion
if __name__=='__main__':
    app.run(debug=True)