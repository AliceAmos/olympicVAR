import math
import os
import pickle

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from imutils import paths
from sklearn import svm
from sklearn.ensemble import RandomForestRegressor
# from sklearn.preprocessing import LabelBinarizer
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import classification_report
from sklearn.metrics import classification_report, mean_squared_error
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelBinarizer
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# ap = argparse.ArgumentParser()																						# CLI to run the commands on the models
# ap.add_argument("-d", "--dataset", required=True,
#                     help="path to input dataset")
# ap.add_argument("-m", "--model", required=True,
#                     help="path to output serialized model")
# ap.add_argument("-l", "--label-bin", required=True,
#                     help="path to output label binarizer")
# ap.add_argument("-e", "--epochs", type=int, default=25,
#                     help="# of epochs to train our network for")
# ap.add_argument("-p", "--plot", type=str, default="plot.png",
#                     help="path to output loss/accuracy plot")
# args = vars(ap.parse_args())


LABELS = set(["10", "9", "8", "7", "6", "5", "4", "3", "2"])															# classes - no video with score 1

def train():
	print("[INFO] loading images...")
	#imagePaths = list(paths.list_images(args["dataset"]))
	imagePaths = list(paths.list_images('/Users/I555250/PycharmProjects/olympicVAR/data'))
	data = []
	labels = []
	# loop over the image paths
	for imagePath in imagePaths:
			# extract the class label from the filename
		label = imagePath.split(os.path.sep)[-2]
			# if the label of the current image is not part of of the labels
			# are interested in, then ignore the image
		if label not in LABELS:
			continue
			# load the image, convert it to RGB channel ordering, and resize
			# it to be a fixed 224x224 pixels, ignoring aspect ratio
		image = cv2.imread(imagePath)
		image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
		image = cv2.resize(image, (240, 240))
			# update the data and labels lists, respectively
		data.append(image)
		labels.append(label)

	data = np.array(data)
	labels = np.array(labels)
	# perform one-hot encoding on the labels
	lb = LabelBinarizer()
	labels = lb.fit_transform(labels)

	(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.25, stratify=labels, random_state=42)	# split train and test (-25%)

	trainAug = ImageDataGenerator(
		rotation_range=30,
		zoom_range=0.15,
		width_shift_range=0.2,
		height_shift_range=0.2,
		shear_range=0.15,
		horizontal_flip=True,
		fill_mode="nearest")

	valAug = ImageDataGenerator()

	mean = np.array([120, 115, 103], dtype="float32")
	trainAug.mean = mean
	valAug.mean = mean

	baseModel = ResNet50(weights="imagenet", include_top=False,input_tensor=Input(shape=(240, 240, 3)))					# the base net we're using

	headModel = baseModel.output																						# adding layers as the output of base model layer
	headModel = AveragePooling2D(pool_size=(7, 7))(headModel)
	headModel = Flatten(name="flatten")(headModel)
	headModel = Dense(512, activation="relu")(headModel)
	headModel = Dropout(0.5)(headModel)
	headModel = Dense(len(lb.classes_), activation="softmax")(headModel)

	model = Model(inputs=baseModel.input, outputs=headModel)															# build the layers on the base net

	for layer in baseModel.layers:
		layer.trainable = False

	print("[INFO] compiling model...")
	# opt = SGD(lr=1e-4, momentum=0.9, decay=1e-4 / args["epochs"])
	opt = SGD(lr=1e-4, momentum=0.9, decay=1e-4 / 15)
	# model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])
	model.compile(loss="mse", optimizer=opt, metrics=[tf.keras.metrics.CategoricalCrossentropy()])

	print("[INFO] training head...")
	# H = model.fit(
	# 	x=trainAug.flow(trainX, trainY, batch_size=32),
	# 	steps_per_epoch=len(trainX) // 32,
	# 	validation_data=valAug.flow(testX, testY),
	# 	validation_steps=len(testX) // 32,
	# 	epochs=args["epochs"])

	H = model.fit(
		x=trainAug.flow(trainX, trainY, batch_size=32),
		steps_per_epoch=len(trainX) // 32,
		validation_data=valAug.flow(testX, testY),
		validation_steps=len(testX) // 32,
		epochs=15)

	print("[INFO] evaluating network...")
	predictions = model.predict(x=testX.astype("float32"), batch_size=32)
	print(classification_report(testY.argmax(axis=1), predictions.argmax(axis=1), target_names=lb.classes_))

																														# plot the training netrics
	# N = args["epochs"]
	N = 15
	plt.style.use("ggplot")
	plt.figure()
	print(H.history)
	plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
	plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
	plt.plot(np.arange(0, N), H.history["categorical_crossentropy"], label="train_acc")
	plt.plot(np.arange(0, N), H.history["val_categorical_crossentropy"], label="val_acc")
	plt.title("Training Loss and Accuracy on Dataset")
	plt.xlabel("Epoch #")
	plt.ylabel("Loss/Accuracy")
	plt.legend(loc="lower left")
	plt.savefig("plot.png")

	print("[INFO] serializing network...")
	# model.save(args["model"], save_format="h5")
	model.save("model/extended.model", save_format="h5")
	# f = open(args["label_bin"], "wb")
	f = open("model/lb.pickle", "wb")
	f.write(pickle.dumps(lb))
	f.close()


def trainParts():

	parts = ['A', 'B', 'C']																								# 3 main parts to train on

	for part in parts:																									# for each part - train according to the arranged data

		print("[INFO] loading images for part ...")
		data_path = 'splited_data/part' + part
		imagePaths = list(paths.list_images(data_path))
		split_data = []
		labels = []
																														# loop over the image paths
		for imagePath in imagePaths:
																														# extract the class label
			label = imagePath.split(os.path.sep)[-2]
			if label not in LABELS:
				continue

			image = cv2.imread(imagePath)
			image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
			image = cv2.resize(image, (240, 240))
																														# update the data and labels lists, respectively
			split_data.append(image)
			labels.append(label)

		data = np.array(split_data)
		labels = np.array(labels)
		lb = LabelBinarizer()
		labels = lb.fit_transform(labels)

		(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.25, stratify=labels, random_state=42)

		trainAug = ImageDataGenerator(
			rotation_range=30,
			zoom_range=0.15,
			width_shift_range=0.2,
			height_shift_range=0.2,
			shear_range=0.15,
			horizontal_flip=True,
			fill_mode="nearest")

		valAug = ImageDataGenerator()
		mean = np.array([120.5, 115.7, 100.9], dtype="float32")
		trainAug.mean = mean
		valAug.mean = mean

		baseModel = ResNet50(weights="imagenet", include_top=False, input_tensor=Input(shape=(240, 240, 3)))

		headModel = baseModel.output
		headModel = AveragePooling2D(pool_size=(5, 5))(headModel)
		headModel = Flatten(name="flatten")(headModel)
		headModel = Dense(256, activation="relu")(headModel)
		headModel = Dropout(0.5)(headModel)
		headModel = Dense(len(lb.classes_), activation="softmax")(headModel)

		model = Model(inputs=baseModel.input, outputs=headModel)
		for layer in baseModel.layers:
			layer.trainable = False

		print("[INFO] compiling splitted model...")
		opt = SGD(learning_rate=1e-4, momentum=0.9, decay=1e-4 / 10)
		model.compile(loss="mse", optimizer=opt, metrics=[tf.keras.metrics.CategoricalCrossentropy()])

		print("[INFO] training splitted head...")

		H = model.fit(
			x=trainAug.flow(trainX, trainY, batch_size=32),
			steps_per_epoch=len(trainX) // 32,
			validation_data=valAug.flow(testX, testY),
			validation_steps=len(testX) // 32,
			epochs=10)

		print("[INFO] evaluating splitted network...")
		predictions = model.predict(x=testX.astype("float32"), batch_size=32)
		print(classification_report(testY.argmax(axis=1), predictions.argmax(axis=1), target_names=lb.classes_))
		# plot the training loss and accuracy

		N = 10
		plt.style.use("ggplot")
		plt.figure()
		print(H.history)
		plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
		plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
		plt.plot(np.arange(0, N), H.history["categorical_crossentropy"], label="train_acc")
		plt.plot(np.arange(0, N), H.history["val_categorical_crossentropy"], label="val_acc")
		plt.title("Training Loss and Accuracy on Dataset")
		plt.xlabel("Epoch #")
		plt.ylabel("Loss/Accuracy")
		plt.legend(loc="lower left")
		plt.savefig("plot_splited" + part +".png")

		print("[INFO] serializing network...")
		model.save("model/splited" + part + ".model", save_format="h5")

		f = open("model/lb.pickle", "wb")
		f.write(pickle.dumps(lb))
		f.close()

def trainMLmodels(data_path):

	data = pd.read_csv(data_path)
	X_data = data.iloc[:, 1:-1].values  																				# separate data to matrix X and vector y
	y = data.iloc[:, -1].values

	X_train, X_test, y_train, y_test = train_test_split(X_data, y, test_size=0.33, random_state=42)						# split to test and train
	scaler = StandardScaler()  																							# normalize values
	X_train = scaler.fit_transform(X_train)
	X_test = scaler.fit_transform(X_test)

	#####################
	####### SVM #########
	#####################

	s = svm.SVR()
	s.fit(X_train, y_train)
	y_pred_svm = s.predict(X_test)


	mse = mean_squared_error(y_pred_svm, y_test)
	print('The RMSE of SVM is:', round(math.sqrt(mse), 2))

	df_svm_scores = pd.DataFrame({'Actual': y_test.flatten(), 'Predicted': y_pred_svm})
	print(df_svm_scores)

	modelfilename = 'model/svm_model.sav'
	pickle.dump(s, open(modelfilename, 'wb'))

	#################
	# RANDOM FOREST #
	#################

	random_forest = RandomForestRegressor(max_depth=4, max_features='sqrt', n_estimators=200)							# after running cross-validation for best parameters
	random_forest.fit(X_train, y_train)
	y_pred_rf = random_forest.predict(X_test)

	mse_rf = mean_squared_error(y_pred_rf, y_test)
	print('The RMSE of RF is:', round(math.sqrt(mse_rf), 2))

	df_rf = pd.DataFrame({'Actual': y_test.flatten(), 'Predicted': y_pred_rf.flatten()})
	print(df_rf)

	model_file_name = 'model/random_forest_model.sav'
	pickle.dump(random_forest, open(model_file_name, 'wb'))

	random_forest_tuning = RandomForestRegressor(random_state=2)														# hypertuning to the model's parameters
	param_grid = {
		'n_estimators': [100, 200, 500],
		'max_features': ['auto', 'sqrt', 'log2'],
		'max_depth': [4, 5, 6, 7, 8],
		'criterion': ['mse', 'mae']
	}

	GSCV = GridSearchCV(estimator=random_forest_tuning, param_grid=param_grid, cv=5)									# GridSearch to find best parameters
	GSCV.fit(X_train, y_train)
	print(GSCV.best_params_)



