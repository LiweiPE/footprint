# import the necessary packages
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np
import glob
import cv2
import os


def load_footprint_attributes(inputPath):
    # initialize the list of column names in the CSV file and then
    # load it using Pandas

    cols = ["FPLL", "FPBuL", "FPBdL","FPLR", "FPBuR", "FPBdR","height"]

    df = pd.read_csv(inputPath, sep="\t", header=None, names=cols)

    return df


def process_footprint_attributes(df, train, test):
    continuous = ["FPLL", "FPBuL", "FPBdL","FPLR", "FPBuR", "FPBdR"]

    # performin min-max scaling each continuous feature column to
    # the range [0, 1]
    cs = MinMaxScaler()
    trainX = cs.fit_transform(train[continuous])
    testX = cs.transform(test[continuous])

    # one-hot encode the zip code categorical data (by definition of
    # one-hot encoing, all output features are now in the range [0, 1])
    # zipBinarizer = LabelBinarizer().fit(df["zipcode"])
    # trainCategorical = zipBinarizer.transform(train["zipcode"])
    # testCategorical = zipBinarizer.transform(test["zipcode"])

    # construct our training and testing data points by concatenating
    # the categorical features with the continuous features
    # trainX = np.hstack([trainCategorical, trainContinuous])
    # testX = np.hstack([testCategorical, testContinuous])

    # return the concatenated training and testing data
    return (trainX, testX, cs)


def load_footprint_images(df, inputPath):
    # initialize our images array (i.e., the house images themselves)
    images = []
    dataL= []
    dataR = []

    # img_dir = "C:\Users\Administrator\Desktop\keras-multi-input\database_fp"  # Enter Directory of all images
    # data_path = os.path.join(inputPath, '*g')
    # files = glob.glob(data_path)
    # data = []
    # for f1 in files:
    #     img = cv2.imread(f1,0)
    #     img = cv2.resize(img, (32, 32)).reshape([32, 32, 1])
    #     data.append(img)

    with open("filenamesL.txt") as fp:
        lines = fp.readlines()
        for l in lines:
            # print(l)
            # filename = l.strip().split()
            filename = l.strip()
            filenames = inputPath + '/' + filename
            #filenames=os.path.join(inputPath, filename)
            # housePaths = sorted(list(glob.glob(filenames)))
            try:
                img = cv2.imread(filenames,0)
                # print(img.shape)
            except Exception as e:
                print('Error [%s]: %s' % (e, filenames))
                continue
            img = cv2.resize(img, (64, 96)).reshape([96, 64, 1])
            # img = cv2.resize(img, (64, 64))
            dataL.append(img)

    with open("filenamesR.txt") as fp:
        lines = fp.readlines()
        for l in lines:
            # print(l)
            # filename = l.strip().split()
            filename = l.strip()
            filenames = inputPath + '/' + filename
            #filenames=os.path.join(inputPath, filename)
            # housePaths = sorted(list(glob.glob(filenames)))
            try:
                img = cv2.imread(filenames,0)
                # cv2.imwrite('test.jpg', img)
                # print(img.shape)
            except Exception as e:
                print('Error [%s]: %s' % (e, filenames))
                continue

            img = cv2.resize(img, (64, 96)).reshape([96, 64, 1])
            # print(img.shape)
            # cv2.imwrite('messigray.jpg', img)
            # img = cv2.resize(img, (64, 64))
            dataR.append(img)

    # return our set of images
    # return np.array(images)
    return (np.array(dataL),np.array(dataR))
