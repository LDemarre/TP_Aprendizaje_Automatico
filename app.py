import streamlit as st
import joblib
import pandas as pd
import numpy as np
from custom_transformers import OutliersImputer, CustomEncoder, CustomStandardScaler, ModelTransformer, ClassificationNeuralNetworkTensorFlow, RegressionNeuralNetworkTensorFlow
import keras.backend.tensorflow_backend as tb
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
import warnings
warnings.simplefilter('ignore')
tb._SYMBOLIC_SCOPE.value = True

# Extraemos los archivos (pipelines y csv)
pipeline_cla = joblib.load('archivos/pipeline_cla.joblib')
pipeline_reg = joblib.load('archivos/pipeline_reg.joblib')
data = pd.read_csv('archivos/data_pipe.csv')

st.sidebar.header('Parámetros puestos por el usuario')

# Función para filtrar los tipos de columnas
def filter_columns_types(X):
    numerical_feature = [feature for feature in X.columns if X[feature].dtypes != 'O']
    categorical_feature = [feature for feature in X.columns if feature not in numerical_feature]
    discrete_feature = [feature for feature in numerical_feature if len(X[feature].unique()) < 25]
    continuous_feature = [feature for feature in numerical_feature if feature not in discrete_feature]
    return numerical_feature, categorical_feature, discrete_feature, continuous_feature
numerical_feature, categorical_feature, discrete_feature, continuous_feature = filter_columns_types(data)

def user_input_parameters(data, categorical_feature, continuous_feature, discrete_feature):
    selected_values = {}  # Diccionario para almacenar los valores seleccionados por el usuario

    # Creamos una lista de opciones de los modelos que puede elegir el usuario
    options = ['Modelo de Regresión', 'Modelo de Clasificación']

    # Lo mostramos en streamlit
    model = st.sidebar.selectbox('¿Qué modelo te gustaría usar?', options)

    # Si es categórica, creamos un selectbox
    for feature in categorical_feature:
        unique_values = data[feature].dropna().unique()
        value = st.sidebar.selectbox(f'Selecciona "{feature}"', unique_values)
        selected_values[feature] = value

    # Si es continua, creamos un slider
    for feature in continuous_feature + discrete_feature:
        min_val = data[feature].min()
        max_val = data[feature].max()
        step_val = (max_val - min_val) / 100  # ajusta el paso según tus necesidades
        
        value = st.sidebar.slider(f'Selecciona "{feature}"', min_val, max_val, data[feature].mean(), step=step_val)
        selected_values[feature] = value
    
    selected_values_df = pd.DataFrame(selected_values, index=[0])
    return model, selected_values_df

# Guardamos los datos seleccionados por el usuario
model, selected_values_df = user_input_parameters(data, categorical_feature, continuous_feature, discrete_feature)

st.title('Predicción de lluvia en Australia')
st.subheader(f'Modelo elegido por el usuario: {model}')
st.subheader('Parámaetros elegidos:')
st.write(selected_values_df)

if st.button('Predecir'):
    if model == 'Modelo de Clasificación': 
        prediction = pipeline_cla.predict(selected_values_df)[0][0]

        if prediction == 0: prediction = 'No va a llover.'
        else: prediction = 'Va a llover.'
        st.success(prediction)
    else:
        prediction = pipeline_cla.predict(selected_values_df)[0][0]
        st.success(f'Cantidad de lluvia que va a caer: {prediction}mm')