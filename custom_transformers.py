from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, ClassifierMixin
from sklearn.preprocessing import StandardScaler
import pandas as pd
import category_encoders as ce
import numpy as np
from keras.layers import Dense
from keras.models import Sequential
from keras.optimizers import Adam
from keras.regularizers import l1, l2, l1_l2
import time

# Lista de características continuas
continuous_feature = ['MinTemp', 'MaxTemp', 'Rainfall', 'Evaporation', 'Sunshine', 'WindGustSpeed', 'WindSpeed9am', 'WindSpeed3pm', 'Humidity9am', 'Humidity3pm', 'Pressure9am', 'Pressure3pm', 'Temp9am', 'Temp3pm']

# Lista de características discretas
discrete_feature = ['Cloud9am', 'Cloud3pm']

# Lista de características categóricas
categorical_feature = ['Location', 'WindGustDir', 'WindDir9am', 'WindDir3pm', 'RainToday']
numerical_feature = continuous_feature + discrete_feature

# Clase de Outliers (Datos Atípicos)
class OutliersImputer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        X = pd.DataFrame(X, columns=numerical_feature + categorical_feature)
        self.bridge_dict = {}

        for feature in continuous_feature:
            IQR = X[feature].quantile(0.75) - X[feature].quantile(0.25)
            lower_bridge = X[feature].quantile(0.25) - (IQR * 1.5)
            upper_bridge = X[feature].quantile(0.75) + (IQR * 1.5)

            self.bridge_dict[feature] = [lower_bridge, upper_bridge]

        return self

    def transform(self, X):
        X = pd.DataFrame(X, columns=numerical_feature + categorical_feature)

        for feature in continuous_feature:
            lower_bridge, upper_bridge = self.bridge_dict[feature]

            X.loc[X[feature] >= upper_bridge, feature] = upper_bridge
            X.loc[X[feature] <= lower_bridge, feature] = lower_bridge
        return X

# Clase de Encoder (Codificación)
class CustomEncoder(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        # Convertimos las columnas categóricas restantes a numéricos ('Date' y 'Location' no tienen valores faltantes y en 'RainToday' y 'RainTomorrow' vamos a usar otro método)
        windgustdir = {'NNW':0, 'NW':1, 'WNW':2, 'N':3, 'W':4, 'WSW':5, 'NNE':6, 'S':7, 'SSW':8, 'SW':9, 'SSE':10, 'NE':11, 'SE':12, 'ESE':13, 'ENE':14, 'E':15}
        winddir9am = {'NNW':0, 'N':1, 'NW':2, 'NNE':3, 'WNW':4, 'W':5, 'WSW':6, 'SW':7, 'SSW':8, 'NE':9, 'S':10, 'SSE':11, 'ENE':12, 'SE':13, 'ESE':14, 'E':15}
        winddir3pm = {'NW':0, 'NNW':1, 'N':2, 'WNW':3, 'W':4, 'NNE':5, 'WSW':6, 'SSW':7, 'S':8, 'SW':9, 'SE':10, 'NE':11, 'SSE':12, 'ENE':13, 'E':14, 'ESE':15}
        location = { 'Sydney': 1, 'SydneyAirport': 2, 'MelbourneAirport': 3, 'Melbourne': 4, 'Canberra': 5}
        
        # Lista de columnas categóricas que necesitan ser mapeadas
        categorical_columns = ['WindGustDir', 'WindDir9am', 'WindDir3pm', 'Location', 'RainToday']

        # Definimos las funciones de mapeo para cada columna
        mapping_functions = {
            'WindGustDir': windgustdir,
            'WindDir9am': winddir9am,
            'WindDir3pm': winddir3pm,
            'Location': location
        }

        # Mapeamos las columnas categóricas
        for column in categorical_columns:
            if column == 'RainToday':
                X[column] = X[column].map({'Yes': 1, 'No': 0})
            else:
                X[column] = X[column].map(mapping_functions[column])

        return X

    def fit_transform(self, X, y=None):
        self.fit(X)
        return self.transform(X)


# Clase de Estandarización
class CustomStandardScaler(BaseEstimator, TransformerMixin):
    def __init__(self, discarded_columns=categorical_feature):
        self.scaler = StandardScaler()
        self.discarded_columns = discarded_columns

    def fit(self, X, y=None):
        X = pd.DataFrame(X, columns=numerical_feature + categorical_feature)

        self.numeric_data = X.drop(self.discarded_columns, axis=1)
        self.scaler.fit(self.numeric_data)
        return self

    def transform(self, X):
        X = pd.DataFrame(X, columns=numerical_feature + categorical_feature)
        
        standarized_data = self.scaler.transform(X.drop(self.discarded_columns, axis=1))
        standarized_df = pd.DataFrame(standarized_data, columns=self.numeric_data.columns)

        missing_columns = X[self.discarded_columns].reset_index(drop=True)
        result = pd.concat([missing_columns, standarized_df], axis=1)

        return result

    def fit_transform(self, X, y=None):
        self.fit(X)
        return self.transform(X)

# Clase para no fitear el modelo y solo predecir con el
class ModelTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, model):
        self.model = model

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X

    def predict(self, X):
        return self.model.predict(X)

'''  Clase personalizada para la creación de las Redes Neuronales para Regresión '''
class RegressionNeuralNetworkTensorFlow(BaseEstimator, RegressorMixin):
    def __init__(self, lr=1, epochs=1, batch_size=1, verbose=0, study=None):
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.verbose = verbose
        self.study = study
        self.model = self.build_model()

    def build_model(self):
        best_params_nn = self.study.best_params
        sublist = []
        layers = []

        for key, value in best_params_nn.items():
            if key == 'learning_rate': lr = value
            elif key == 'regularizer': regularizer = value
            elif key == 'reg_strength': reg_strength = value
            elif key != 'num_layers': sublist.append([key, value])

            if len(sublist) == 2:
                layers.append(sublist)
                sublist = []
        
        if regularizer == 'l1': regularizer = l1(reg_strength)
        elif regularizer == 'l2': regularizer = l2(reg_strength)
        elif regularizer == 'l1_l2': regularizer = l1_l2(l1=reg_strength, l2=reg_strength)
        else: regularizer = None

        model = Sequential()
        for neurons, activation in layers:
            layer_name = neurons[0]
            num_neurons = neurons[1]
            activation_name = activation[1]

            if layer_name == 'num_neurons_ent': model.add(Dense(num_neurons, input_dim=71, activation=activation_name, kernel_regularizer=regularizer))
            else: model.add(Dense(num_neurons, activation=activation_name, kernel_regularizer=regularizer))
        model.add(Dense(1, activation='linear'))
        
        optimizer = Adam(learning_rate=lr)
        model.compile(loss='huber_loss', optimizer=optimizer)
        return model

    def fit(self, X, y, X_test=None, y_test=None):
        start = time.time()
        
        X = np.array(X)
        y = np.array(y)

        self.model.fit(X, y, epochs=self.epochs, verbose=self.verbose, batch_size=self.batch_size)
        return self

    def predict(self, X):
        X = np.array(X)
        predictions = self.model.predict(X)
        return predictions
    
'''  Clase personalizada para la creación de las Redes Neuronales para Clasificación Binaria '''
class ClassificationNeuralNetworkTensorFlow(BaseEstimator, ClassifierMixin):
    def __init__(self, lr=1, epochs=1, batch_size=1, verbose=0, study=None):
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.verbose = verbose
        self.study = study
        self.classes_ = [0, 1]

    def build_model(self):
        best_params_nn = self.study.best_params
        sublist = []
        layers = []

        for key, value in best_params_nn.items():
            if key == 'learning_rate': lr = value
            elif key == 'regularizer': regularizer = value
            elif key == 'reg_strength': reg_strength = value
            elif key != 'num_layers': sublist.append([key, value])

            if len(sublist) == 2:
                layers.append(sublist)
                sublist = []

        if regularizer == 'l1': regularizer = l1(reg_strength)
        elif regularizer == 'l2': regularizer = l2(reg_strength)
        elif regularizer == 'l1_l2': regularizer = l1_l2(l1=reg_strength, l2=reg_strength)
        else: regularizer = None
            
        model = Sequential()
        for neurons, activation in layers:
            layer_name = neurons[0]
            num_neurons = neurons[1]
            activation_name = activation[1]

            if layer_name == 'num_neurons_ent': model.add(Dense(num_neurons, input_dim=71, activation=activation_name, kernel_regularizer=regularizer))
            else: model.add(Dense(num_neurons, activation=activation_name, kernel_regularizer=regularizer))
        model.add(Dense(1, activation='sigmoid'))
        
        optimizer = Adam(learning_rate=lr)
        model.compile(loss='binary_crossentropy', optimizer=optimizer)
        return model

    def fit(self, X, y, X_test=None, y_test=None):
        start = time.time()
        
        X = np.array(X)
        y = np.array(y)
        
        self.model = self.build_model()
        self.model.fit(X, y, epochs=self.epochs, verbose=self.verbose, batch_size=self.batch_size)
        return self

    def predict(self, X):        
        X = np.array(X)
        predictions = (self.model.predict(X) > 0.5).astype(int)
        return predictions