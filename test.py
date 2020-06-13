# import
from imutils import paths
from utils import quantify_image

import cv2
import pickle
import os
import argparse

# construct argument parser
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--directory", type=str,
	default="test/", help="path to images(for testing)")
ap.add_argument("-m", "--model", required=True,
	help="path to model.pkl")
ap.add_argument("-l", "--label-encoder", required=True,
	help="path to label_encoder.pkl")
args = vars(ap.parse_args())

# load the model and label encoder
print("[INFO] loading model ...")
model = pickle.load(open(args["model"], 'rb'))
le = pickle.load(open(args["label_encoder"], 'rb'))

# load the images from the directory
imagePaths = list(paths.list_images(args["directory"]))

# loop through the images
for imagePath in imagePaths:

	# get filename
	fileName = imagePath.split(os.path.sep)[-1]

	# load the image
	image = cv2.imread(imagePath)
	output = image.copy()

	# preprocess the image
	image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	image = cv2.resize(image, (200, 200))
	image = cv2.threshold(image, 0, 255,
		cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

	# quantify the features
	features = quantify_image(image)

	# predict
	preds = model.predict([features])
	label = le.inverse_transform(preds)[0]

	# draw the label on the output image
	color = (0, 255, 0) if label == "healthy" else (0, 0, 255)
	cv2.putText(output, label, (3, 20), 
		cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

	cv2.imshow(fileName, output)
	cv2.imwrite(fileName, output)
	cv2.waitKey(0)

