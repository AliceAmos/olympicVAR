import pickle
from collections import deque
import cv2
import numpy as np
from tensorflow.keras.models import load_model

FRAMES_PER_VID = 103
																														# CLI to run the commands on the models
# ap = argparse.ArgumentParser()
# ap.add_argument("-m", "--model", required=True,
# 	help="path to trained serialized model")
# ap.add_argument("-l", "--label-bin", required=True,
# 	help="path to  label binarizer")
# ap.add_argument("-i", "--input", required=True,
# 	help="path to our input video")
# ap.add_argument("-o", "--output", required=True,
# 	help="path to our output video")
# ap.add_argument("-s", "--size", type=int, default=128,
# 	help="size of queue for averaging")
# args = vars(ap.parse_args())


def predict(new_vid_path, model_path):																					# predict each frame of the video, according to the saved model passed as a parameter

	print("[INFO] loading model and label binarizer...")
	# model = load_model(args["model"])
	model = load_model(model_path)
	# lb = pickle.loads(open(args["label_bin"], "rb").read())
	lb = pickle.loads(open('/Users/I555250/PycharmProjects/olympicVAR/model/lb.pickle', "rb").read())
	Q = deque(maxlen=FRAMES_PER_VID)
	mean = np.array([120, 115, 103][::1], dtype="float32")

	# vs = cv2.VideoCapture(args["input"])
	vs = cv2.VideoCapture(new_vid_path)
	writer = None
	(W, H) = (None, None)
	frames_scores = list()

	while True:																											# go over frames
		(grabbed, frame) = vs.read()
		if not grabbed:																									# means end of frames/video
			break

		if W is None or H is None:																						# get dimensions of the frame
			(H, W) = frame.shape[:2]

		frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
		frame = cv2.resize(frame, (240, 240)).astype("float32")															# cut irrelevant parts
		frame -= mean

		preds = model.predict(np.expand_dims(frame, axis=0))[0]															# predict by the model
		Q.append(preds)
		results = np.array(Q).mean(axis=0)																				# consider previous predictions
		i = np.argmax(results)
		frames_scores.append(i)																							# append the final predictions for the current frames to predictions' list
		label = lb.classes_[i]

		# text = "score: {}".format(label)																				# in case we want to write the score of each frame on the actual video
		# cv2.putText(frame, text, (35, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.25, (0, 255, 0), 5)
		# if writer is None:
		# 	fourcc = cv2.VideoWriter_fourcc(*"MJPG")
		# 	writer = cv2.VideoWriter(new_vid_path, fourcc, 10, (W, H), True)
		# writer.write(output)

	return frames_scores
	print("[INFO] cleaning up...")
	writer.release()
	vs.release()



def predict_by_parts(vid_path):

	print("[INFO] loading models and label binarizer...")
	# model = load_model(args["model"])
	modelA = load_model('/Users/I555250/PycharmProjects/olympicVAR/model/splitedA.model')								# load all the designated models (trained on different parts of the video)
	modelB = load_model('/Users/I555250/PycharmProjects/olympicVAR/model/splitedB.model')
	modelC = load_model('/Users/I555250/PycharmProjects/olympicVAR/model/splitedC.model')
	# lb = pickle.loads(open(args["label_bin"], "rb").read())
	lb = pickle.loads(open('/Users/I555250/PycharmProjects/olympicVAR/model/lb.pickle', "rb").read())

	mean = np.array([120, 115, 104][::1], dtype="float32")
	Q = deque(maxlen=103)

	# vs = cv2.VideoCapture(args["input"])
	vs = cv2.VideoCapture(vid_path)

	(W, H) = (None, None)
	frames_scores = list()

	index = 0
	while True:
		(grabbed, frame) = vs.read()
		if not grabbed:
			break

		if W is None or H is None:
			(H, W) = frame.shape[:2]

		frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
		frame = cv2.resize(frame, (240, 240)).astype("float32")
		frame -= mean

		if index < 31:																									# get prediction to the frame according to its position, time in the video
			preds = modelA.predict(np.expand_dims(frame, axis=0))[0]
		elif index < 61:
			preds = modelB.predict(np.expand_dims(frame, axis=0))[0]
		else:
			preds = modelC.predict(np.expand_dims(frame, axis=0))[0]

		Q.append(preds)
		results = np.array(Q).mean(axis=0)
		i = np.argmax(results)
		frames_scores.append(i)
		label = lb.classes_[i]

	print(len(frames_scores))
	return frames_scores
	print("[INFO] cleaning up...")
	vs.release()


def most_frequent(preds):                                                                                               # get the most frequent number in a list - voting
	return max(set(preds), key=preds.count)

