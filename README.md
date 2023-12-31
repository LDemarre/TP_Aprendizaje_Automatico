# Trabajo Práctico - Aprendizaje Automático
### Integrante: Lucas Demarré
### Año: 2023

## Intrucciones para ejecutar el Trabajo Práctico.

Este repositorio contiene dos carpetas y tres archivos, vamos a ir uno por viendo explicando que hace cada uno. Es importante tener en cuenta
que uno de esos tres archivos es **requirements.txt**, para poder ejecutar tanto el *app.py* como el archivo con todo el procesamiento, entrenamiento
y evaluación de los modelos *Trabajo_Práctico_Aprendizaje_Automático*, es necesario instalar todas esas librerías dentro de su entorno sistema local/virtual.
Ahora vamos a ver cada carpeta y lo que contiene:

### **Archivos**: 
Esta carpeta contiene los archivos exportados para poder utilizar el script *app.py*, es decir, poner en marcha la puesta en producción. Los archivos son:
* **data_pipe**: es una tabla exportada desde el archivo general, es para poder sacar de forma automático los tipos de cada columna.
* **pipeline_cla.pkl**: es el pipeline que transforma los datos y predice usando el modelo elegido para **Clasificación**.
* **pipeline_reg.pkl**: es el pipeline que transforma los datos y predice usando el modelo elegido para **Regresión**.

### **Procesamiento de Datos**: 
Esta carpeta contiene los archivos que se utilizaron para todo lo relacionado con el preprocesamiento de datos, entrenamiento y evaluación de modelos. Los archivos son:
* **Consignas.pdf**: es el PDF que contiene cada consigna que se tuvo que hacer en el Trabajo Práctico.
* **Trabajo_Práctico_Aprendizaje_Automático.ipynb**: es el Trabajo Práctico como tal, esto contiene, en gran medida, todo lo responsable de que la puesta en producción exista y se pueda ejecutar.
* **weatherAUS.csv**: es el dataset que vamos a utilizar para entrenar y evaluar nuestros modelos.

### **Archivos sueltos**
* **app.py**: contiene todo lo relacionado con la puesta en producción de los modelos de aprendizaje automático elegidos para cada problema.
* **customs_transformers.py**: contiene las clases personalizadas que utiliza app.py para transformar los datos y luego entrenar los modelos.
* **requirements.txt**: es lo que se tiene que utlizar para instalar las librerías necesarias para ejecutar el Trabajo Práctico.

Para poder correr el entorno encargado de la puesta en producción, es necesario correr dentro del mismo directorio que se encuentra el **app.py** el comando `streamlit run app.py`.
Esto le abrirá automáticamente un entorno local donde puede modificar los datos que quiera ingresar y predecir con dichos datos.