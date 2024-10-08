'''
Created on 16 Jun 2020

@author: Nur Raudzah Binti Abdullah
'''
#Import required libraries && necessary modules
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
#Sklearn specific
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
#Keras specific
from keras.models import Sequential
from keras.layers import Dense,BatchNormalization, Dropout
from keras.callbacks import EarlyStopping, ModelCheckpoint
import tensorflow_docs as tfdocs
import tensorflow_docs.plots
from keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
#Import csv file and load data using Pandas
df = pd.read_csv('C:/data.csv') 
print(df.shape)
df.describe()

#Changing pandas dataframe to numpy array and separate features(X) and targets(Y)
X = df.iloc[:,0:5].values
y = df.iloc[:,5].values

scalar = MinMaxScaler()
scalar.fit(X)
df = scalar.transform(X)
print(X)
#Split data into training and testing data (9:10)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
print(X_train.shape); print(X_test.shape)

#Building the Deep Learning Regression Model
#Define model
model = Sequential()
model.add(Dense(5, input_dim = 5, activation = 'relu'))
model.add(Dropout(0.10187039549226673))
model.add(Dense(179, activation = 'relu'))
model.add(Dropout(0.10187039549226673))
model.add(Dense(179, activation = 'relu'))
model.add(Dropout(0.10187039549226673))
model.add(Dense(179, activation = 'relu'))
model.add(Dropout(0.10187039549226673))
model.add(Dense(179, activation = 'relu'))
model.add(Dense(1))

print(model.summary()) #Print model Summary

#Hyperparameter configuration and fitting the data
#Configure the model and start training
# Compile model
opt = Adam(lr=0.0034290622790032403) 
model.compile(loss='mse', optimizer=opt, metrics= ['mse','mae'])

# Set callback functions to early stop training and save the best model so far
callbacks = [EarlyStopping(monitor='val_loss', patience=2),
             ModelCheckpoint(filepath='best15_model.h5', monitor='val_loss', save_best_only=True)]
#Train model
history = model.fit(X_train, y_train, epochs=1000, batch_size=16, callbacks=callbacks,  verbose=1, validation_data=(X_test,y_test))

#print(history.history.keys())
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
# Plot history: MAE
plt.plot(history.history['mae'], label='MAE (testing data)')
plt.plot(history.history['val_mae'], label='MAE (validation data)')
plt.title('MAE')
plt.ylabel('MAE value')
plt.xlabel('No. epoch')
plt.legend(loc="upper left")
plt.show()

#Evaluate model with test dataset
loss, mae, mse = model.evaluate(X_test, y_test, verbose=1)
print("Mean Squared Error: {:5.2f} ".format(mse))
model.save("'best14_model.h5'")
#Make prediction on new data
#Load New Data
real_df = pd.read_csv('C:/Prediction.csv')
scalar = MinMaxScaler()
scalar.fit(real_df)
df = scalar.transform(real_df)
ynew = model.predict(df)
print(ynew)
