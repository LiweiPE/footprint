# USAGE
# python mixed_training.py --dataset Houses_Dataset

# import the necessary packages
from keras import backend as K
from adabound import AdaBound
from pyimagesearch_2 import datasets
from pyimagesearch_2.SEResNeXt import SEResNeXt
from pyimagesearch_2 import models
from sklearn.model_selection import train_test_split
from keras.layers.core import Dense
from keras.models import Model
from keras.optimizers import Adam
from keras.optimizers import SGD
from keras.layers import concatenate
import numpy as np
import argparse
import datetime
import pandas as pd
import json
import matplotlib.pyplot as plt

import cv2
from sklearn.preprocessing import MinMaxScaler


import locale
import os
ep=10
bs=16

def root_mean_squared_error(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true)))

def create_submission(prediction,testY):
    now = datetime.datetime.now()
    sub_file = 'submission_'+str(now.strftime("%Y-%m-%d-%H-%M"))+'_epoch_'+str(ep)+'_batchsize_'+str(bs)+'.csv'
    print ('Creating submission: ', sub_file)
    pd.DataFrame({'Data': testY*maxHeight,'Height': prediction*maxHeight}).to_csv(sub_file, index=False)

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", type=str, required=True,
                help="path to input dataset of house images")
args = vars(ap.parse_args())

# construct the path to the input .txt file that contains information
# on each house in the dataset and then load the dataset
print("[INFO] loading footprint attributes...")
# inputPath = os.path.sep.join([args["dataset"], "HousesInfo.txt"])
df = datasets.load_footprint_attributes("FootInfo2.txt")
print(df)
# load the house images and then scale the pixel intensities to the
# range [0, 1]
print("[INFO] loading footprint images...")
(imagesL,imagesR) = datasets.load_footprint_images(df, args["dataset"])
# imagesL = datasets.load_footprint_images(df, args["dataset"])

imagesR = imagesR / 255.0
imagesL = imagesL / 255.0

# partition the data into training and testing splits using 75% of
# the data for training and the remaining 25% for testing
print("[INFO] processing data...")
# split = train_test_split(df, images, test_size=0.20, random_state=42)
print(imagesR.shape)
print(imagesL.shape)

images = np.concatenate([imagesR, imagesL], axis = 1)

print(images.shape)

split = train_test_split(df, images, test_size=0.1, random_state=42)
(trainAttrX, testAttrX, trainImagesX, testImagesX) = split
trainImagesXR, trainImagesXL = trainImagesX[:, :96, ...], trainImagesX[:, 96:, ...]
testImagesXR, testImagesXL = testImagesX[:, :96, ...], testImagesX[:, 96:, ...]
# trainImagesXR, trainImagesXL = trainImagesX[:, :64, ...], trainImagesX[:,64:, ...]
# testImagesXR, testImagesXL = testImagesX[:, :64, ...], testImagesX[:,64:, ...]
"""
split = train_test_split(df, imagesL, test_size=0.20, random_state=42)
(trainAttrXL, testAttrXL, trainImagesXL, testImagesXL) = split

splitR = train_test_split(df, imagesR, test_size=0.20, random_state=42)
(trainAttrXR, testAttrXR, trainImagesXR, testImagesXR) = splitR

trainAttrX=trainAttrXR
testAttrX=testAttrXR
"""


# trainAttrX=pd.concat([trainAttrXL, trainAttrXR])
# testAttrX=pd.concat([testAttrXL, testAttrXR])

# print(testAttrXL)
# print("###################################################")
# testAttrXL.append(testAttrXR)
# print(testAttrXL)
# print("###################################################")
# print("###################################################")
# print(testAttrX)
# trainAttrX=np.concatenate((trainAttrXL,trainAttrXR))
# testAttrX=np.concatenate((testAttrXL,testAttrXR))

print(testAttrX["height"])
print(trainAttrX.shape)

print(testImagesXR.shape)
print(testImagesXL.shape)



# scale height parameter to the range [0, 1] (will lead to better
# training and convergence)
maxHeight = trainAttrX["height"].max()
trainY = trainAttrX["height"] / maxHeight
testY = testAttrX["height"] / maxHeight
# trainY = trainAttrX["height"]
# testY = testAttrX["height"]
# process the house attributes data by performing min-max scaling
# on continuous features, one-hot encoding on categorical features,
# and then finally concatenating them together
(trainAttrX, testAttrX, cs) = datasets.process_footprint_attributes(df, trainAttrX, testAttrX)
print(type(testAttrX))
print(testAttrX)
# create the MLP and CNN models
mlp = models.create_mlp(trainAttrX.shape[1], regress=True)
cnnL = models.create_cnn(64, 96, 1, regress=True)
cnnR = models.create_cnn(64, 96, 1, regress=True)
# mlp = models.create_mlp(trainAttrX.shape[1], regress=True)
# cnnL = SEResNeXt(64).model
# cnnR = SEResNeXt(64).model

# create the input to our final set of layers as the *output* of both
# the MLP and CNN
combinedCNN = concatenate([cnnL.output, cnnR.output])

combinedInput = concatenate([combinedCNN, mlp.output])
# combinedInput = concatenate([mlp.output, cnn.output])

# our final FC layer head will have two dense layers, the final one
# being our regression head
x = Dense(4, activation="relu")(combinedInput)
x = Dense(1, activation="linear")(x)

# our final model will accept categorical/numerical data on the MLP
# input and images on the CNN input, outputting a single value (the
# predicted price of the house)
model = Model(inputs=[cnnL.input, cnnR.input, mlp.input ], outputs=x)

# compile the model using mean absolute percentage error as our loss,
# implying that we seek to minimize the absolute percentage difference
# between our price *predictions* and the *actual prices*
# opt = Adam(lr=1e-3, decay=1e-3 / 200)

# opt = Adam(lr=1e-3, decay=1e-3 / 200,beta_1=0.9, beta_2=0.999,epsilon=None)
# opt = AdaBound(lr=1e-03,
#                 final_lr=0.1,
#                 gamma=1e-03,
#                 weight_decay=0.1,
#                 amsbound=False)
# model.load_weights('myModel.h5')

opt=SGD(lr=0.001, decay=1e-3/200, momentum=0.9, nesterov=True)
# model.compile(loss="mean_absolute_percentage_error", optimizer=opt)
model.compile(loss=root_mean_squared_error, optimizer=opt)

# train the model
print("[INFO] training model...")
hist=model.fit(
    [trainImagesXL,trainImagesXR,trainAttrX], trainY,
    validation_data=([testImagesXL,testImagesXR,testAttrX], testY),
    epochs=ep, batch_size=bs)
#200, 8


# summarize history for loss
plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# model.save_weights('myModel.h5')

# make predictions on the testing data
print("[INFO] predicting height...")
preds = model.predict([testImagesXL, testImagesXR,testAttrX])

# model.save('model2.h5')

json_file ='epoch_' + str(ep) + '_batchsize_' + str(bs) + '.json'

with open(json_file, 'w') as f:
    json.dump(hist.history, f)

# compute the difference between the *predicted* house prices and the
# *actual* house prices, then compute the percentage difference and
# the absolute percentage difference
diff = preds.flatten() - testY
percentDiff = (diff / testY) * 100
absPercentDiff = np.abs(percentDiff)

# compute the mean and standard deviation of the absolute percentage
# difference
mean = np.mean(absPercentDiff)
std = np.std(absPercentDiff)

# finally, show some statistics on our model
# locale.setlocale(locale.LC_ALL, "en_US.UTF-8")
# print("[INFO] avg. house price: {}, std house price: {}".format(
#     locale.currency(df["height"].mean(), grouping=True),
#     locale.currency(df["height"].std(), grouping=True)))
print("[INFO] mean: {:.2f}%, std: {:.2f}%".format(mean, std))
create_submission((preds.flatten()),testY)


##############################################################################################
model.load_weights('myModel.h5')

while  True:
    filenameL=input("Enter footprint Left: ")
    filenameR=input("Enter footprint Right: ")
    FPLL=input("Enter FPLL: ")
    FPBuL=input("Enter FPBuL: ")
    FPBdL=input("Enter FPBdL: ")
    FPLR=input("Enter FPLR: ")
    FPBuR=input("Enter FPBuR: ")
    FPBdR=input("Enter FPBdR: ")
    height=input("Enter real Stature: ")

    continuous = np.asarray([float(FPLL), float(FPBuL), float(FPBdL), float(FPLR), float(FPBuR), float(FPBdR)]).reshape([1, -1])

    # performin min-max scaling each continuous feature column to
    # the range [0, 1]
    # cs = MinMaxScaler()
    testX = cs.transform(continuous)

    filenamesL = args["dataset"] + '/' + filenameL
    filenamesR = args["dataset"] + '/' + filenameR

    imgL = cv2.imread(filenamesL, 0)
    imgL = cv2.resize(imgL, (64, 96)).reshape([1, 96, 64, 1])
    imgL=imgL/255

    imgR = cv2.imread(filenamesR, 0)
    imgR = cv2.resize(imgR, (64, 96)).reshape([1, 96, 64, 1])
    imgR = imgR / 255

    # result = model.predict_generator([imgL, imgR,continuous], 1, verbose=1)
    result = model.predict([imgL, imgR, continuous])
    print("Predicted value for stature is: ")
    print(result.flatten()*maxHeight)
    ################################################################################################
