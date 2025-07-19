from flask import Flask, request, render_template, jsonify
import pickle
import numpy as np
import os

app = Flask(__name__)

try:
    with open('rf_model.pkl', 'rb') as f:
        modelo = pickle.load(f)
        print("Modelo cargado")
except FileNotFoundError:
    print("No se encontró el modelo")
    modelo = None


@app.route('/', methods=['GET'])
def home():
    """Mostrar formulario"""
    return render_template('index.html', 
                         titulo='Mi App ML',
                         encabezado='Predictor de Diabetes',
                         mensaje='Ingresa los datos para hacer una predicción')



@app.route('/prediccion', methods=['POST'])
def prediccion():
    """Procesar datos y hacer la predicción"""
    if modelo == None:
        return "Modelo no cargado", 500
    try:
        Pregnancies = request.form.get('Pregnancies', type=int)
        Glucose = request.form.get('Glucose', type=int)
        BloodPressure = request.form.get('BloodPressure', type=int)
        SkinThickness = request.form.get('SkinThickness', type=int)
        Insulin = request.form.get('Insulin', type=float)
        BMI = request.form.get('BMI', type=float)
        DiabetesPedigreeFunction = request.form.get('DiabetesPedigreeFunction', type=float)
        Age = request.form.get('Age', type=int)
        # Una vez obtenidos se crea una lista para que el modelo pueda procesar los datos de forma correcta
        datos = [Pregnancies, Glucose, BloodPressure, SkinThickness, 
                Insulin, BMI, DiabetesPedigreeFunction, Age]

        datos_entrada = np.array([datos])

        if None in datos:
            return "Datos no disponibles, vuelva a cargar", 400
        # Sigue la predicción
        prediccion = modelo.predict(datos_entrada)[0]
        probabilidad = modelo.predict_proba(datos_entrada)[0].max()

        resultado_texto = "Diabetes" if prediccion == 1 else "No diabetes"

        return render_template('resultado.html',
                               prediccion = resultado_texto,
                               probabilidad = round(probabilidad * 100, 2),
                               datos_entrada = datos
                               )

    except (ValueError, KeyError) as e:
        return f'Error en los datos: {str(e)}', 400
    except Exception as e:
        return f'Error inesperado: {str(e)}', 500
        



if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)


