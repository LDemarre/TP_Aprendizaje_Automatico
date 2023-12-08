# Librerías
import streamlit as st
import joblib
import pandas as pd
import numpy as np
from custom_transformers import OutliersImputer, CustomEncoder, CustomStandardScaler, ModelTransformer, ClassificationNeuralNetworkTensorFlow, RegressionNeuralNetworkTensorFlow
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
import warnings
warnings.simplefilter('ignore')

# Extraemos los archivos (pipelines y csv)
pipeline_cla = joblib.load('Archivos/pipeline_cla.joblib')
pipeline_reg = joblib.load('Archivos/pipeline_reg.joblib')
data = pd.read_csv('Archivos/data_pipe.csv')

# Ponelo un título a la barra lateral
st.sidebar.header('Parámetros puestos por el usuario')

# Función para filtrar los tipos de columnas
def filter_columns_types(X):
    numerical_feature = [feature for feature in X.columns if X[feature].dtypes != 'O']
    categorical_feature = [feature for feature in X.columns if feature not in numerical_feature]
    discrete_feature = [feature for feature in numerical_feature if len(X[feature].unique()) < 25]
    continuous_feature = [feature for feature in numerical_feature if feature not in discrete_feature]
    return numerical_feature, categorical_feature, discrete_feature, continuous_feature

# Filtramos las columnas por tipo
numerical_feature, categorical_feature, discrete_feature, continuous_feature = filter_columns_types(data)

# Función para ir guardando los paramétros ingresados por el usuario
def user_input_parameters(data, categorical_feature, continuous_feature, discrete_feature):
    # Diccionario para almacenar los valores seleccionados por el usuario
    selected_values = {}

    # Creamos una lista de opciones de los modelos que puede elegir el usuario
    options = ['Modelo de Regresión', 'Modelo de Clasificación']

    # Lo mostramos en streamlit
    model = st.sidebar.selectbox('¿Qué modelo te gustaría usar?', options)

    # Si es categórica, creamos un selectbox
    for feature in categorical_feature:
        unique_values = data[feature].dropna().unique()

        # Valor elegido por el usuario
        value = st.sidebar.selectbox(f'Selecciona "{feature}"', unique_values)
        selected_values[feature] = value

    # Si es continua, creamos un slider
    for feature in continuous_feature + discrete_feature:
        min_val = data[feature].min()
        max_val = data[feature].max()
        step_val = (max_val - min_val) / 100
        
        # Valor elegido por el usuario
        value = st.sidebar.slider(f'Selecciona "{feature}"', min_val, max_val, data[feature].mean(), step=step_val)
        selected_values[feature] = value
    
    # Convertimos todos los valores en un dataset
    selected_values_df = pd.DataFrame(selected_values, index=[0]) 
    return model, selected_values_df

# Guardamos los datos seleccionados por el usuario
model, selected_values_df = user_input_parameters(data, categorical_feature, continuous_feature, discrete_feature)

# Ponemos títulos y subtítulos al entorno
st.title('Predicción de lluvia en Australia')
st.subheader(f'Modelo elegido por el usuario: {model}')
st.subheader('Parámaetros elegidos:')

# Mostramos en forma de Dataset los datos elegidos por el usuario
st.write(selected_values_df)

# Detectamos el modelo elegido y predecimos con dicho modelo
if st.button('Predecir'):
    if model == 'Modelo de Clasificación': 
        prediction = pipeline_cla.predict(selected_values_df)[0][0]

        if prediction == 0: prediction = 'No va a llover.'
        else: prediction = 'Va a llover.'

        # Mostramos la predicción en pantalla
        st.success(prediction)
    else:
        prediction = pipeline_reg.predict(selected_values_df)[0][0]

        # Mostramos la predicción en pantalla
        st.success(f'Cantidad de lluvia que va a caer: {prediction:.2f}mm')