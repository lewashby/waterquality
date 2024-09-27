import pandas as pd
import numpy as np
import re
import csv
from sklearn.preprocessing import MinMaxScaler

from .Utils import series_to_supervised

class DataLoader:

    def __init__(self, dir, file_type="xlsx"):
        if file_type == "xlsx":
            self.data = pd.read_excel(dir)
        else:
            self.data = pd.read_csv(dir)
    
    def slice_data(self, column, start_value, end_value):
        self.data = self.data[(self.data[column] > start_value) & (self.data[column] < end_value)]
    
    def handle_na(self):
        self.data = self.data.dropna()

class DataPreparation:

    scaler = None
    
    @staticmethod
    def create_dataset(data, target_column, train_size, timesteps_in=1, timesteps_out=1, normalize=True, feature_range=(0,1)):
        columns = data.columns
        n_columns = len(columns)

        target_feature = data.pop(target_column)
        data.insert(0, target_column, target_feature)
        
        target_column_index = 0
        n_features = timesteps_in * n_columns

        values = data.values
        if normalize == True:
            DataPreparation.scaler = MinMaxScaler(feature_range=feature_range)
            values = DataPreparation.scaler.fit_transform(values)
        
        reframed = series_to_supervised(values, n_in=timesteps_in, n_out=timesteps_out)
        start = timesteps_in * n_columns
        end = start + n_columns
        remove_indexes = list(range(start, end))
        remove_indexes.remove(start + target_column_index)
        # leave only target_column variable for predictions => var(t), var(t+1), var(t+2), ..., var(t+timesteps_out)
        # if timesteps_out==1 then there is only one future/present prediction
        # otherwise if timesteps_out>1 we want to guess n timesteps of the target variable in the future
        remove_indexes = [i+n_columns*j for i in remove_indexes for j in range(timesteps_out)]
        reframed.drop(reframed.columns[remove_indexes], axis=1, inplace=True)

        values = reframed.values
        train = values[:train_size, :]
        test = values[train_size:, :]
        train_X, train_y = train[:, :n_features], train[:, -timesteps_out:]
        test_X, test_y = test[:, :n_features], test[:, -timesteps_out:]

        train_X = train_X.reshape((train_X.shape[0], timesteps_in, n_columns))
        test_X = test_X.reshape((test_X.shape[0], timesteps_in, n_columns))

        return train_X, train_y, test_X, test_y
        
    @staticmethod
    def inverse_transform_y(y, n_features):
        fill = np.zeros((y.shape[0], y.shape[1], n_features))    
        fill[:,:,0] = y
        real = []
        for i in fill:
            real.append(DataPreparation.scaler.inverse_transform(i))
        real = np.array(real)
        return real[:,:,0]

    @staticmethod
    def transform_x(X):
        if DataPreparation.scaler is not None:
            return DataPreparation.scaler.transform(X)
        else:
            print("Not scaler has been fit")


class Reader:

  def __init__(self, dir, encoding="ISO-8859-1"):
    data = []
    header = None
    measures = []
    with open(dir, 'r', encoding=encoding) as file:
      csvreader = csv.reader(file, dialect=csv.excel)
      for row in csvreader:
        values = ','.join(row)
        values = values.split(';')
        if values[0] == 'Date (MM/DD/YYYY)' and header == None:
          header = [x for x in values if x != '']
        elif re.match(r"\d{1,2}\/\d{1,2}\/\d{2,4}", values[0]):
          measures.append([x.replace(',', '.') for x in values if x != ''])
        elif len(measures) > 0:
          df = pd.DataFrame(data=measures, columns=header)
          df.set_index(header[0])
          data.append(df)
          header = None
          measures = []
    self.data = pd.concat(data, axis=0)
    self.data.reset_index(drop=True, inplace=True)

  def parse_data(self):
    self.data.insert(2, "DateTime", self.data[[self.data.columns[0], self.data.columns[1]]].agg(' '.join, axis=1))
    columns = self.data.columns
    self.data[columns[6:]] = self.data[columns[6:]].apply(pd.to_numeric, errors='coerce')
    self.data[columns[0]] = pd.to_datetime(self.data[columns[0]], dayfirst=True, format='%d/%m/%Y')
    self.data[columns[2]] = pd.to_datetime(self.data[columns[2]], dayfirst=True)
    datetime_utc = pd.to_datetime(self.data[columns[2]], dayfirst=True)
    datetime_utc = datetime_utc.dt.tz_localize("UTC").dt.tz_convert("Europe/Rome")
    self.data[columns[2]] = datetime_utc
    self.data[columns[3:5]] = self.data[columns[3:5]].apply(pd.to_numeric, errors='coerce')
    self.data[columns[5]] = self.data[columns[5]].astype('string')

