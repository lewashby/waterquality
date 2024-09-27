import pandas as pd
import numpy as np
import math
import random
import matplotlib.pyplot as plt
import keras_tuner as kt
from sklearn.metrics import mean_squared_error

from .DataHandler import Reader, DataPreparation
from .Model import Model
from .Utils import plot_single_prediction, plot_prediction_sequence, plot_predictions
from .ModelBuilder import RNNHyperModel

class Pipeline:

    def __init__(self, sensor_dir_file, weather_dir_file):
        self.sensor_file = sensor_dir_file
        self.weather_file = weather_dir_file

    def load_files(self):
        # load sensors data
        reader = Reader(self.sensor_file)
        reader.parse_data()
        self.sensor_data =  reader.data
        
        # load weather data
        weather_data = pd.read_excel(self.weather_file, sheet_name="Dati meteorologici")        
        weather_data["Data - ora"] = weather_data["Data - ora"].dt.tz_localize("UTC").dt.tz_convert("Europe/Rome")
        weather_data.rename(columns={"Data - ora": "DateTime"}, inplace=True)
        self.weather_data = weather_data

    def slice_data(self):
        self.sensor_data = self.sensor_data[self.sensor_data['DateTime'] > '2022-07-01 00:00:00']

    def drop_columns(self):
        # delete unnecesary columns
        sensors_unnecesary_parameters = ["Time (Fract", " Sec)", "Site Name", "Date (MM/DD/YYYY)", "Time (HH:mm:ss)"]
        sensors_wrong_measures = ["NO3 -N mg/L", "NO3 -N mV", "NH4+ -N mg/L", "NH4+ -N mV"]
        sensors_duplicated_measures = ["fDOM RFU", "pH mV"]
        # do some analysis with parameters with too many missing values
        sensors_too_many_missing_values = ["NH3 mg/L", "ODO % sat", "ODO % local", "ODO mg/L"]
        
        # if used the following line, check handle_missing_values funtion for ORP mV and pH columns
        sensors_unused = ['Cond µS/cm', 'nLF Cond µS/cm', 
                          'ORP mV', 'Battery V', 'Cable Pwr V', 
                          'Wiper Position volt', 'Sal psu', 'SpCond µS/cm', 
                          'TDS mg/L', 'pH', 'Turbidity FNU']
        
        columns_to_delete = sensors_unnecesary_parameters + sensors_wrong_measures + sensors_duplicated_measures + sensors_too_many_missing_values + sensors_unused
        self.sensor_data.drop(labels=columns_to_delete, axis=1, inplace=True)

        weather_columns_to_delete = ["Data", "Ora"]
        self.weather_data.drop(labels=weather_columns_to_delete, axis=1, inplace=True)

    def handle_missing_values(self):
        # fill null values
        ph_mean = self.sensor_data[self.sensor_data['pH'] < 8.5]['pH'].mean()
        self.sensor_data['ORP mV'] = self.sensor_data['ORP mV'].fillna(self.sensor_data['ORP mV'].mean())
        self.sensor_data['pH'] = self.sensor_data['pH'].fillna(ph_mean)
        
        # pH values greater or equal than 10 could be a device error
        # let's fill pH values greater or equal than 10 with the mean of values less than 8.5
        self.sensor_data.loc[self.sensor_data['pH'] >= 10, 'pH'] = ph_mean

        # weather values to numeric
        weather_data_columns = self.weather_data.columns
        self.weather_data[weather_data_columns[3:]] = self.weather_data[weather_data_columns[3:]].apply(pd.to_numeric, errors='coerce')
        # fill all columns missing values with the column mean
        for c in weather_data_columns[3:]:
          self.weather_data[c] = self.weather_data[c].fillna(self.weather_data[c].mean())
        

    def merge_data_sources(self):
        merged = pd.merge_ordered(self.sensor_data, self.weather_data, fill_method="ffill", on="DateTime")
        self.merged_data = merged
        self.merged_data.set_index("DateTime", inplace = True)

    def align_start_date(self):
        s_start_date = self.sensor_data.min().iloc[0]
        w_start_date = self.weather_data.min().iloc[0]

        min_date = max(s_start_date, w_start_date)
        self.merged_data = self.merged_data[self.merged_data.index > min_date]

    def add_doc_formula_values(self):
        fDOM_QSU = self.sensor_data["fDOM QSU"]
        temp = self.sensor_data["Temp °C"]
        turbidity_FNU = self.sensor_data["Turbidity FNU"]
        turbidity_soglia = 700
        DOC_max = 12
        turbidity_max = turbidity_FNU.max()
        fDOMcorrmax = (DOC_max - 0.8) / 0.054
        fDOMT = fDOM_QSU / (1 - 0.01*(temp - 25))
        fDOMcorr = fDOMT / np.exp(-0.004 * turbidity_FNU)
        fDOMcorrSoglia = fDOMT / np.exp(-0.004 * turbidity_soglia)
        formula = fDOMcorrSoglia + (fDOMcorrmax - fDOMcorrSoglia) / (turbidity_max - turbidity_soglia) * (turbidity_FNU - turbidity_soglia)
        fDOMcorr_greater_soglia = np.where(turbidity_FNU < turbidity_soglia, fDOMcorr, formula)
        DOC_formula = (0.054 * fDOMcorr_greater_soglia) + 0.8
        
        self.sensor_data["DOC formula"] = DOC_formula

    def create_dataset(self, timesteps_in=24, timesteps_out=1):
        train_X, train_y, test_X, test_y = DataPreparation.create_dataset(
            self.merged_data, 
            "DOC formula", 
            (len(self.merged_data) // 3) * 2, 
            timesteps_in=timesteps_in,
            timesteps_out=timesteps_out
        )
        self.scaler = DataPreparation.scaler

        self.train_X = train_X
        self.train_y = train_y
        
        half_test = len(test_y) // 2
        self.validation_X = test_X[:half_test,:]
        self.validation_y = test_y[:half_test]

        self.test_X = test_X[half_test:,:]
        self.test_y = test_y[half_test:]

    def create_model(self, output=1):
        input_shape = (self.train_X.shape[1], self.train_X.shape[2])
        self.model = Model(input_shape, output=output)

    def train_model(self, epochs, verbose=1, plot=False):
        half_test = len(self.test_y) // 2
        self.history = self.model.fit(
            self.train_X,
            self.train_y,
            epochs=epochs,
            validation_data=(self.validation_X, self.validation_y),
            verbose=verbose
        )
        if plot:
            plt.plot(self.history.history['loss'], label='train')
            plt.plot(self.history.history['val_loss'], label='validation')
            plt.legend()
            plt.show()

    def evaluate_model(self, plot=False, save_plot_path=None):
        yhat = self.model.predict(self.test_X)
        yhat = DataPreparation.inverse_transform_y(yhat, self.test_X.shape[2])
        real_y = DataPreparation.inverse_transform_y(self.test_y, self.test_X.shape[2])

        rmse = math.sqrt(mean_squared_error(real_y, yhat))
        print('Test RMSE: %.3f' % rmse)
        if plot:
            plot_predictions(real_y, yhat, save_plot_path)

    def save_model(self, model, path):
        model.save(path)

    def load_trained_model(self, path):
        self.model = Model(path=path)

    def predict_single_value(self, X, y=None, plot=False, save_plot_path=None):
        #given past values predict next target column value
        #plot if required
        X_values = DataPreparation.transform_x(X)
        yhat = self.model.predict(X_values)
        yhat = DataPreparation.inverse_transform_y(yhat, X_values.shape[1])
        yhat = yhat[0]

        if y is not None:
            real_Y = y[:,0]
            rmse = math.sqrt(mean_squared_error(real_Y, yhat))
            print('RMSE: %.3f' % rmse)

        if plot==True:
            past_Y = X[:,0]
            real_Y = y[:,0]
            plot_single_prediction(yhat, past_Y, real_Y, save_plot_path)
        return yhat

    def predict_sequence_values(self, X, y=None, plot=False, save_plot_path=None):
        #given past values predict target column nexts sequence of n values
        #plot if required
        X_values = DataPreparation.transform_x(X)
        yhat = self.model.predict(X_values)
        yhat = DataPreparation.inverse_transform_y(yhat, X_values.shape[1])
        yhat = yhat[0,:]

        if y is not None:
            real_Y = y[:,0]
            rmse = math.sqrt(mean_squared_error(real_Y, yhat))
            print('RMSE: %.3f' % rmse)

        if plot==True:
            past_Y = X[:,0]
            real_Y = y[:,0] if y is not None else None
            plot_prediction_sequence(yhat, past_Y, real_Y, save_plot_path)
        return yhat

    def pick_random_sample(self):
        timesteps_in = self.model.model.input_shape[1]
        timesteps_out = self.model.model.output_shape[1]

        max_value = len(self.merged_data) -timesteps_in + timesteps_out
        min_value = len(self.merged_data)//3*2
        random_index = random.randint(min_value, max_value)
        random_sample = self.merged_data.iloc[random_index:random_index+timesteps_in].values
        random_sample_real_output = self.merged_data.iloc[random_index+timesteps_in:random_index+timesteps_in+timesteps_out].values

        return random_sample, random_sample_real_output


    def hyperparameter_tuning(self, epochs=20, batch_size=72, max_trials=5, executions_per_trial=3):            
        tuner = kt.RandomSearch(
            hypermodel = RNNHyperModel((self.train_X.shape[1], self.train_X.shape[2]), self.train_y.shape[1]),
            objective='mse',
            max_trials=max_trials,
            executions_per_trial=executions_per_trial
        )
        tuner.search(
            x=self.train_X,
            y=self.train_y,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(self.validation_X, self.validation_y),
        )
        models = tuner.get_best_models(num_models=1)
        best_model = models[0]
        self.best_model = best_model

    