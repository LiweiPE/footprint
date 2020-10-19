# USAGE
# python mixed_training.py --dataset Houses_Dataset

# import the necessary packages
from pyimagesearch import datasets
from pyimagesearch import models
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


import locale
import os

def create_submission(prediction,testY):
    now = datetime.datetime.now()
    sub_file = 'submission_'+'_'+str(now.strftime("%Y-%m-%d-%H-%M"))+'.csv'
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
df = datasets.load_footprint_attributes("FootInfo.txt")

# load the house images and then scale the pixel intensities to the
# range [0, 1]
print("[INFO] loading footprint images...")
images = datasets.load_footprint_images(df, args["dataset"])
images = images / 255.0

# partition the data into training and testing splits using 75% of
# the data for training and the remaining 25% for testing
print("[INFO] processing data...")
split = train_test_split(df, images, test_size=0.20, random_state=42)
(trainAttrX, testAttrX, trainImagesX, testImagesX) = split

# find the largest house price in the training set and use it to
# scale our house prices to the range [0, 1] (will lead to better
# training and convergence)
maxHeight = trainAttrX["height"].max()
trainY = trainAttrX["height"] / maxHeight
testY = testAttrX["height"] / maxHeight

print(testAttrX)
# trainY = trainAttrX["height"]
# testY = testAttrX["height"]
# process the house attributes data by performing min-max scaling
# on continuous features, one-hot encoding on categorical features,
# and then finally concatenating them together
(trainAttrX, testAttrX) = datasets.process_footprint_attributes(df, trainAttrX, testAttrX)

# create the MLP and CNN models
mlp = models.create_mlp(trainAttrX.shape[1], regress=False)
cnn = models.create_cnn(64, 64, 1, regress=False)

# create the input to our final set of layers as the *output* of both
# the MLP and CNN
combinedInput = concatenate([mlp.output, cnn.output])

# our final FC layer head will have two dense layers, the final one
# being our regression head
x = Dense(4, activation="relu")(combinedInput)
x = Dense(1, activation="linear")(x)

# our final model will accept categorical/numerical data on the MLP
# input and images on the CNN input, outputting a single value (the
# predicted price of the house)
model = Model(inputs=[mlp.input, cnn.input], outputs=x)

# compile the model using mean absolute percentage error as our loss,
# implying that we seek to minimize the absolute percentage difference
# between our price *predictions* and the *actual prices*
# opt = Adam(lr=1e-3, decay=1e-3 / 200)
opt = Adam(lr=1e-3, decay=1e-3 / 200,beta_1=0.9, beta_2=0.999,epsilon=None)
# opt=SGD(lr=0.01, decay=1e-3/200, momentum=0.9, nesterov=True)
# model.compile(loss="mean_absolute_percentage_error", optimizer=opt)
model.compile(loss="mean_squared_error", optimizer=opt)

# train the model
print("[INFO] training model...")
model.fit(
    [trainAttrX, trainImagesX], trainY,
    validation_data=([testAttrX, testImagesX], testY),
    epochs=300, batch_size=32)
#200, 8

# make predictions on the testing data
print("[INFO] predicting height...")
preds = model.predict([testAttrX, testImagesX])

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
