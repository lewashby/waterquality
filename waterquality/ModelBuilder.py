from kerastuner import HyperModel
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Input, Dropout


class RNNHyperModel(HyperModel):
    def __init__(self, input_shape, output):
        self.input_shape = input_shape
        self.output = output

    def build(self, hp):
        model = Sequential()
        model.add(LSTM(hp.Int('input_unit',min_value=32,max_value=128,step=32),return_sequences=True, input_shape=self.input_shape))
        for i in range(hp.Int('n_layers', 1, 5)):
            model.add(LSTM(hp.Int(f'lstm_{i}_units',min_value=32,max_value=128,step=32),return_sequences=True))
        model.add(LSTM(hp.Int(f'lstm_{6}_units',min_value=32,max_value=128,step=32),return_sequences=False))
        model.add(Dropout(hp.Float('Dropout_rate',min_value=0,max_value=0.5,step=0.1)))
        model.add(Dense(6))
        model.add(Dropout(hp.Float('Dropout_rate',min_value=0,max_value=0.5,step=0.1)))
        model.add(Dense(self.output))
        model.compile(loss='mean_squared_error', optimizer='adam',metrics = ['mse'])
        return model