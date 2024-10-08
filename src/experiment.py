'''
Created on 21 Jun 2020

@author: Nur Raudzah Binti Abdullah
'''
# import all packages
import pandas as pd
pd.set_option("display.max_column",100)
import numpy as np
from datetime import timedelta, datetime
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
import xgboost as xgb
import matplotlib.pyplot as plt
import plotly.graph_objects as go

#read the data
df = pd.read_csv('C:/data.csv')

df.head()

# import packages 
import tensorflow as tf
from tensorflow.keras import layers
from sklearn.preprocessing import MinMaxScaler

# simple data cleaning
X = df.select_dtypes('number').drop(['API'], axis=1)
y = np.log1p(df['API'])

# Need to scale the features for neural networks, otherwise the training doesn't converge.
scalar = MinMaxScaler()
scalar.fit(X)
df = scalar.transform(X)
print(X)

# split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# This returns a multi-layer-perceptron model in Keras.
def get_keras_model(num_hidden_layers, 
                    num_neurons_per_layer, 
                    dropout_rate, 
                    activation):
    # create the MLP model.
    # define the layers.
    inputs = tf.keras.Input(shape=(X_train.shape[1],))  # input layer.
    x = layers.Dropout(dropout_rate)(inputs) # dropout on the weights.
    # Add the hidden layers.
    for i in range(num_hidden_layers):
        x = layers.Dense(num_neurons_per_layer, 
                         activation=activation)(x)
        x = layers.Dropout(dropout_rate)(x)
    
    # output layer.
    outputs = layers.Dense(1, activation='linear')(x)
    
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model
    
# This function takes in the hyperparameters and returns a score (Cross validation).
def keras_mlp_cv_score(parameterization, weight=None):
    
    model = get_keras_model(parameterization.get('num_hidden_layers'),
                            parameterization.get('neurons_per_layer'),
                            parameterization.get('dropout_rate'),
                            parameterization.get('activation'))
    
    opt = parameterization.get('optimizer')
    opt = opt.lower()
    
    learning_rate = parameterization.get('learning_rate')
    
    if opt == 'adam':
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    elif opt == 'rms':
        optimizer = tf.keras.optimizers.RMSprop(learning_rate=learning_rate)
    else:
        optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate)
    
    NUM_EPOCHS = 50
    
    # Specify the training configuration.
    model.compile(optimizer=optimizer,
                  loss=tf.keras.losses.MeanSquaredError(),
                  metrics=['mse'])

    data = X_train
    labels = y_train.values
    
    # fit the model using a 20% validation set.
    res = model.fit(data, labels, epochs=NUM_EPOCHS, batch_size=parameterization.get('batch_size'),
                    validation_split=0.2)
    
    # look at the last 10 epochs. Get the mean and standard deviation of the validation score.
    last10_scores = np.array(res.history['val_loss'][-10:])
    mean = last10_scores.mean()
    sem = last10_scores.std()
    
    # If the model didn't converge then set a high loss.
    if np.isnan(mean):
        return 9999.0, 0.0
    
    return mean, sem
# Define the search space.
parameters=[
    {
        "name": "learning_rate",
        "type": "range",
        "bounds": [0.0001, 0.5],
        "log_scale": True,
    },
    {
        "name": "dropout_rate",
        "type": "range",
        "bounds": [0.01, 0.5],
        "log_scale": True,
    },
    {
        "name": "num_hidden_layers",
        "type": "range",
        "bounds": [1, 10],
        "value_type": "int"
    },
    {
        "name": "neurons_per_layer",
        "type": "range",
        "bounds": [1, 300],
        "value_type": "int"
    },
    {
        "name": "batch_size",
        "type": "choice",
        "values": [8, 16, 32, 64, 128, 256],
    },
    
    {
        "name": "activation",
        "type": "choice",
        "values": ['tanh', 'relu'],
    },
    {
        "name": "optimizer",
        "type": "choice",
        "values": ['adam', 'rms', 'sgd'],
    },
]
# import more packages
from ax.service.ax_client import AxClient
from ax.utils.notebook.plotting import render, init_notebook_plotting

init_notebook_plotting()

ax_client = AxClient()

# create the experiment.
ax_client.create_experiment(
    name="keras_experiment",
    parameters=parameters,
    objective_name='keras_cv',
    minimize=True)

def evaluate(parameters):
    return {"keras_cv": keras_mlp_cv_score(parameters)}

#print all result
for i in range(25):
    parameters, trial_index = ax_client.get_next_trial()
    print(ax_client.complete_trial(trial_index=trial_index, raw_data=evaluate(parameters)))
    
# look at all the trials.
print(ax_client.get_trials_data_frame().sort_values('trial_index'))
best_parameters, values = ax_client.get_best_parameters()

# the best set of parameters.
for k in best_parameters.items():
    print(k)

print()

# the best score achieved.
means, covariances = values
print(means)
#render(ax_client.get_optimization_trace()) # Objective_optimum is optional.

# Cannot do contour plot because it doesn't use a GP model.
# train the model on the full training set and test.
keras_model = get_keras_model(best_parameters['num_hidden_layers'], 
                              best_parameters['neurons_per_layer'], 
                              best_parameters['dropout_rate'],
                              best_parameters['activation'])
print(keras_model.summary())

opt = best_parameters['optimizer']
opt = opt.lower()

learning_rate = best_parameters['learning_rate']

if opt == 'adam':
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
elif opt == 'rms':
    optimizer = tf.keras.optimizers.RMSprop(learning_rate=learning_rate)
else:
    optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate)

NUM_EPOCHS = 50

# Specify the training configuration.
keras_model.compile(optimizer=optimizer,
              loss=tf.keras.losses.MeanSquaredError(),
              metrics=['mse'])

data = X_train
labels = y_train.values
history = keras_model.fit(data, labels, epochs=NUM_EPOCHS, batch_size=best_parameters['batch_size'],validation_split=0.2)
#Visualize the training and validation loss using loss graph
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Loss')
plt.ylabel('Loss value')
plt.xlabel('No. epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()
# Plot history: MSE
plt.plot(history.history['mse'], label='MSE (testing data)')
plt.plot(history.history['val_mse'], label='MSE (validation data)')
plt.title('MSE')
plt.ylabel('MSE value')
plt.xlabel('No. epoch')
plt.legend(loc="upper left")
plt.show()
# Use the model to predict the test values.
test_pred = keras_model.predict(X_test)
print("MSE:",mean_squared_error(y_test.values, test_pred))

# save results to json file.
ax_client.save_to_json_file()

