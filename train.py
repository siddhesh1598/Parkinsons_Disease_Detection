# import
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix
from imutils import paths
from utils import quantify_image

import pickle
import numpy as np
import argparse
import cv2
import os

# function to load training and testing images
def load_split(path):
	imagePaths = list(paths.list_images(path))
	data = []
	labels = []

	for imagePath in imagePaths:
		# extract label
		label = imagePath.split(os.path.sep)[-2]

		# load image, convert it to greyscale and
		# resize it to 200x200
		image = cv2.imread(imagePath)
		image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
		image = cv2.resize(image, (200, 200))

		# threshold the image in inverse manner
		image = cv2.threshold(image, 0, 255,
			cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

		# quantify the image
		features = quantify_image(image)

		# update the data and labels list
		data.append(features)
		labels.append(label)

	return (np.array(data), np.array(labels))

# construct argument parser
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,
	help="Path to the input dataset")
ap.add_argument("-t", "--trials", type=int, default=5, 
	help="#trials to run")
args = vars(ap.parse_args())

# define the training and testing paths
trainingPath = os.path.sep.join([args["dataset"],
	"training"])
testingPath = os.path.sep.join([args["dataset"],
	"testing"])

# load the training and testing data
print("[INFO] loading dataset")
(trainX, trainY) = load_split(trainingPath)
(testX, testY) = load_split(testingPath)

# encode the labels
le = LabelEncoder()
trainY = le.fit_transform(trainY)
testY = le.transform(testY)

# initialize trials dictionary
trials = {}

# loop over number of trials
for i in range(args["trials"]):
	# train
	print("[INFO] training model {} of {} ...".format(i+1,
		args["trials"]))
	model = RandomForestClassifier(n_estimators=100)
	model.fit(trainX, trainY)

	# make predictions
	predictions = model.predict(testX)
	metrics = {}

	# compute confusion metrics and derive
	# accuracy, sensitivity and specificity
	cm = confusion_matrix(testY, predictions).flatten()
	(tn, fp, fn, tp) = cm
	metrics["Accuracy"] = (tp + tn) / float(cm.sum())
	metrics["Sensitivity"] = tp / float(tp + fn)
	metrics["Specificity"] = tn / float(tn + fp)

	# loop over the metrics
	for (k, v) in metrics.items():
		# update trials dictionary
		l = trials.get(k, [])
		l.append(v)
		trials[k] = l

# loop over metrics
for metric in ("Accuracy", "Sensitivity", "Specificity"):

	values = trials[metric]
	mean = np.mean(values)
	std = np.std(values)

	print("{}: ".format(metric))
	print("\t Mean: {:.4f}".format(mean))
	print("\t Standard Deviation: {:.4f}".format(std))
	print()

# save the model and label encoder to disk
pickle.dump(model, open('model.pkl', 'wb'))
pickle.dump(le, open('label_encoder.pkl', 'wb'))
