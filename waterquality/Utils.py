import matplotlib.pyplot as plt
import plotly.express as px
import pandas as pd
from pandas import DataFrame
from plotly.subplots import make_subplots

# convert series to supervised learning
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
    # put it all together
    agg = pd.concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg
    

def boxplot(data, columns, elements_by_row=3, height=1000, width=700):
    rows = len(columns) // elements_by_row if len(columns) % elements_by_row == 0 else (len(columns) // elements_by_row) + 1    
    fig = make_subplots(rows = rows, cols = elements_by_row, start_cell = "top-left", subplot_titles=columns)
    
    for i, c in enumerate(columns):
        r_number, c_number = (i//elements_by_row)+1, (i%elements_by_row)+1
        fig.add_trace(px.box(data, y=c).data[0], row=r_number, col=c_number)
    
    fig.update_layout(height=height, width=width, title_text="BoxPlots")
    fig.show()

def plot_predictions(y, yhat, save_path=None):
    if y.shape[1] == 1:
        plt.plot(yhat, label='predictions')
        plt.plot(y, label='real values')
        plt.legend()
        if save_path is not None: 
            plt.savefig(save_path, bbox_inches='tight')
        plt.show()
    else:
        print("Plotting sequence prediction for the whole dataset is not available")

def plot_single_prediction(yhat, history=None, y=None, save_path=None):
    if history is not None:
        timesteps_in = history.shape[0]
        plt.plot(range(timesteps_in), history, color='blue', label='past values')
    plt.plot(timesteps_in+1, yhat, label='prediction', markersize=10, marker=".", color="red")
    if y is not None:
        plt.plot(timesteps_in+1, y, label='real value', markersize=10, marker=".", color="green", alpha=0.5)
    plt.legend()
    if save_path is not None: 
        plt.savefig(save_path, bbox_inches='tight')
    plt.show()

def plot_prediction_sequence(yhat, history=None, y=None, save_path=None):
    timesteps_in = history.shape[0]
    timesteps_out = yhat.shape[0]
    if history is not None:
        plt.plot(range(timesteps_in), history, color='blue', label='past values')
    plt.plot(range(timesteps_in, timesteps_in+timesteps_out), yhat, color='red', label='predictions')
    if y is not None:
        plt.plot(range(timesteps_in, timesteps_in+timesteps_out), y, color='orange', label='real future values')
    plt.legend()
    if save_path is not None: 
        plt.savefig(save_path, bbox_inches='tight')
    plt.show()
    