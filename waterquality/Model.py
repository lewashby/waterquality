from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Input, Dropout
from numpy import expand_dims, zeros, concatenate, array

class Model:

    def __init__(self, input_shape, output=1, loss='mean_squared_error', optimizer='adam', metrics=None):
        model = Sequential()
        model.add(Input(shape=(input_shape[0], input_shape[1])))
        model.add(LSTM(128, return_sequences=True, dropout=0.2))
        model.add(LSTM(64, return_sequences=True, dropout=0.2))
        model.add(LSTM(32))
        model.add(Dense(10))
        model.add(Dense(output))
        model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
        self.model = model

    def fit(self, train_X, train_Y, validation_data, epochs=20, batch_size=72, verbose=0, shuffle=False):
        history = self.model.fit(
            train_X, train_Y, 
            batch_size=batch_size,
            epochs=epochs, 
            validation_data=validation_data, 
            verbose=verbose, 
            shuffle=shuffle
        )
        return history


    def predict(self, X):
        samples = 1
        output = self.model.output_shape[1]
        if X.ndim == 2:
            timestamps = X.shape[0]
            n_features = X.shape[1]
            instance_x = expand_dims(X, axis=0)
        else:
            samples = X.shape[0]
            timestamps = X.shape[1]
            n_features = X.shape[2]
            instance_x = X
        
        predictions = self.model.predict(instance_x)
        return predictions
    

    @staticmethod
    def build_a_model(hp):
        print(hp)
        model = Sequential()
        model.add(LSTM(hp.Int('input_unit',min_value=32,max_value=128,step=32),return_sequences=True, input_shape=(24,11)))
        for i in range(hp.Int('n_layers', 1, 5)):
            model.add(LSTM(hp.Int(f'lstm_{i}_units',min_value=32,max_value=128,step=32),return_sequences=True))
        model.add(LSTM(hp.Int(f'lstm_{6}_units',min_value=32,max_value=128,step=32),return_sequences=False))
        model.add(Dropout(hp.Float('Dropout_rate',min_value=0,max_value=0.5,step=0.1)))
        model.add(Dense(6))
        model.add(Dropout(hp.Float('Dropout_rate',min_value=0,max_value=0.5,step=0.1)))
        model.add(Dense(1))
        model.compile(loss='mean_squared_error', optimizer='adam',metrics = ['mse'])
        return model

