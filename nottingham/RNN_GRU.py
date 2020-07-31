"""
Nottingham Data
GRU RNN
"""
import json
import random
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GRU
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard, ReduceLROnPlateau

with open('location-history.json', 'r') as f:
    location = json.load(f)

locationData = pd.DataFrame.from_dict(location["locations"], orient='columns')

# index 766600 is location data from 23rd of sept 2019
locationData = locationData.drop(locationData.index[0:766600])

locationData['timestampMs'] = pd.to_datetime(locationData['timestampMs'], unit='ms')

del locationData['velocity']
del locationData['accuracy']
del locationData['heading']
del locationData['altitude']
del locationData['verticalAccuracy']
del locationData['activity']

locationData = locationData.set_index('timestampMs')

# shift steps is the amount of time steps into the future we want to predict
# depending on the data and how spaced out each record is timewise is important to consider
shiftSteps = 493 # average daily records

targetNames = ['latitudeE7', 'longitudeE7']

locationDataTargets = locationData[targetNames].shift(-shiftSteps)

xData = locationData.values[0:-shiftSteps] # inputs signals

yData = locationDataTargets.values[:-shiftSteps] # output signals (target data) shiftSteps into the future

print("x shape:", xData.shape)
print("y shape:", yData.shape)

numData = len(xData) # amount of data from the input signls

trainSplit = 0.9 # the split of training to test sets

numTrain = int(trainSplit * numData) # number of observations in training set

xTrain = xData[0:numTrain] # input training set
xTest = xData[numTrain:] # input test set

yTrain = yData[0:numTrain] # output training set
yTest = yData[numTrain:] # output test set

print("xTrain shape:", xTrain.shape)
print("xTest shape:", xTest.shape)
print("yTrain shape:", yTrain.shape)
print("yTest shape:", yTest.shape)

numSignalsX = xData.shape[1] # number of input signals

numSignalsY = yData.shape[1] # number of output signals

xScaler = MinMaxScaler()

xTrainScaled = xScaler.fit_transform(xTrain) # this scales the min max values between 0 and 1

xTestScaled = xScaler.transform(xTest) # this scales the min max values between 0 and 1

yScaler = MinMaxScaler()

yTrainScaled = yScaler.fit_transform(yTrain) # this scales the min max values between 0 and 1

yTestScaled = yScaler.transform(yTest) # this scales the min max values between 0 and 1

def batchGen(batchSize, sequenceLength):
    """
    creates random sequence batches of sequenceLength from the data for training
    
    """
    
    while True:
        # create an array for input batch
        xShape = (batchSize, sequenceLength, numSignalsX)
        xBatch = np.zeros(shape=xShape, dtype=np.float16)

        # create an array for output batch
        yShape = (batchSize, sequenceLength, numSignalsY)
        yBatch = np.zeros(shape=yShape, dtype=np.float16)

        # batch filled with random sequences
        for i in range(batchSize):
            # finds a rondom index number to select from training data
            index = np.random.randint(numTrain - sequenceLength)
            
            # gets training data of sequenceLength from the random index
            xBatch[i] = xTrainScaled[index:index+sequenceLength]
            yBatch[i] = yTrainScaled[index:index+sequenceLength]
        
        yield (xBatch, yBatch)

# 128 random sequences of sequenceLength from the data
batchSize = 128 # batch is dependent on GPU and RAM

# avg daily records times by 4
sequenceLength = 493 * 4 # amount of time steps the random batch will be

generator = batchGen(batchSize=batchSize, sequenceLength=sequenceLength)

validationData = (np.expand_dims(xTestScaled, axis=0), np.expand_dims(yTestScaled, axis=0)) # used to check loss after each epoch

model = Sequential() # creating the model

model.add(GRU(units=256, return_sequences=True, input_shape=(None, numSignalsX,))) # adding 256 output signals

model.add(Dense(numSignalsY, activation='sigmoid')) # dense layer to condense output signals down to 2

warmUpSteps = 50

def lossMSE(yTrue, yPred):
    """
    find the mean square error but ignoring the warmUpSteps
    
    yTrue is the true output
    yPred is the predicted output

    """
    
    # slices out warmUpSteps
    yTrueSlice = yTrue[:, warmUpSteps:, :] # [batchSize, sequenceLength, numSignalsY]
    yPredSlice = yPred[:, warmUpSteps:, :] # [batchSize, sequenceLength, numSignalsY]


    # calculates means squared error for every value in the tensor
    loss = tf.losses.mean_squared_error(yTrueSlice, yPredSlice)
    lossMean = tf.reduce_mean(loss)

    return lossMean

optimizer = RMSprop(lr=1e-3) # this is the optimiser with first learning rate = 0.001

model.compile(loss=lossMSE, optimizer=optimizer) # this compiles the model ready for training

pathCheckpoint = 'checkpoint.keras'

callbackCheckpoint = ModelCheckpoint(filepath=pathCheckpoint, monitor='val_loss', verbose=1, save_weights_only=True, save_best_only=True) # this callback writes checkpoints while training

callbackEarlyStopping = EarlyStopping(monitor='val_loss', patience=5, verbose=1) # callback for stopping the optimization when performance worsens on the validation-set

callbackReduceLR = ReduceLROnPlateau(monitor='val_loss', factor=0.1, min_lr=1e-4, patience=0, verbose=1) # this callback reduces the learning-rate if the validation-loss has not improved since the last epoch

callbacks = [callbackEarlyStopping, callbackCheckpoint, callbackReduceLR]

model.fit_generator(generator=generator, epochs=30, steps_per_epoch=100, validation_data=validationData, callbacks=callbacks) # this now trains our network

try:
    model.load_weights(pathCheckpoint)
except Exception as error:
    print("error loading checkpoint")
    print(error)
    
result = model.evaluate(x=np.expand_dims(xTestScaled, axis=0), y=np.expand_dims(yTestScaled, axis=0))

print("validation loss:", result)

def pltComp(startIndex=0, length=1000, train=True, graph=0):
    """
    plots the true and predicted values to a graph
    
    startIndex is the selected index from the data
    length is amount of data to show
    train is a boolean value which decides which data to show (training or test)
    graph allows for unique graph .png files
    
    """
    
    # use training or test data
    if train:
        x = xTrainScaled
        yTrue = yTrain
    else:
        x = xTestScaled
        yTrue = yTest
    
    endIndex = startIndex + length
    
    x = x[startIndex:endIndex]
    yTrue = yTrue[startIndex:endIndex]
    
    x = np.expand_dims(x, axis=0)

    yPred = model.predict(x)
    
    yPredRescaled = yScaler.inverse_transform(yPred[0])
    
    for signal in range(len(targetNames)):
        signalPred = yPredRescaled[:, signal]
        
        signalTrue = yTrue[:, signal]

        plt.figure(figsize=(15,5))
        
        plt.plot(signalTrue, label='true')
        plt.plot(signalPred, label='pred')
        
        # plots a grey area for the warmUpSteps
        greyWarmUp = plt.axvspan(0, warmUpSteps, facecolor='black', alpha=0.15)
        
        # plots data and saves a figure to a .png
        plt.ylabel(targetNames[signal])
        plt.legend()
        plt.savefig(str(graph) + str(signal) + '.png')
        plt.show()
        
def longLatComp(startIndex=0, length=1000, train=True):
    """
    get different long and lats to compare to prediction
    
    startIndex is the selected index from the data
    length is amount of data to show
    train is a boolean value which decides which data to show (training or test)
    
    """
    # use training or test data
    if train:
        x = xTrainScaled
        yTrue = yTrain
    else:
        x = xTestScaled
        yTrue = yTest
    
    endIndex = startIndex + length
    
    x = x[startIndex:endIndex]
    yTrue = yTrue[startIndex:endIndex]
    
    x = np.expand_dims(x, axis=0)

    yPred = model.predict(x)
    
    yPredRescaled = yScaler.inverse_transform(yPred[0])

    for x in range(5):
        idx = random.randint(startIndex, endIndex)
        
        latPredRandom = yPredRescaled[idx, 0]
        longPredRandom = yPredRescaled[idx, 1]
    
        longPredRandom = longPredRandom / 10**7   
        latPredRandom = latPredRandom / 10**7
        
        print ("random longitude prediction",x + 1,":",longPredRandom)
        print ("random latitude prediction",x + 1,":",latPredRandom)
        print ("\n")
        
        latTrueRandom = yTrue[idx, 0]
        longTrueRandom = yTrue[idx, 1]
        
        longTrueRandom = longTrueRandom / 10**7   
        latTrueRandom = latTrueRandom / 10**7
        
        print ("random longitude true",x + 1,":",longTrueRandom)
        print ("random latitude true",x + 1,":",latTrueRandom)
        print ("\n")
    
        
pltComp(startIndex=0, length=84243, train=True, graph=1)
longLatComp(startIndex=0, length=84243, train=True)

pltComp(startIndex=0, length=9361, train=False, graph=2)
longLatComp(startIndex=0, length=9361, train=False)