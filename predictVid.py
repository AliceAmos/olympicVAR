from tensorflow.keras.models import load_model
from collections import deque
import numpy as np
import argparse
import pickle
import cv2

# construct the argument parser and parse the arguments
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

def predict():

	print("[INFO] loading model and label binarizer...")
	# model = load_model(args["model"])
	model = load_model('model/extended.model')
	# lb = pickle.loads(open(args["label_bin"], "rb").read())
	lb = pickle.loads(open('model/lb.pickle', "rb").read())
	# initialize the image mean for mean subtraction along with the
	# predictions queue
	mean = np.array([123.68, 116.779, 103.939][::1], dtype="float32")
	Q = deque(maxlen=103)


	# vs = cv2.VideoCapture(args["input"])
	vs = cv2.VideoCapture('new_vid_91.mp4')
	writer = None
	(W, H) = (None, None)
	# loop over frames from the video file stream
	while True:
		# read the next frame from the file
		(grabbed, frame) = vs.read()
		# if the frame was not grabbed, then we have reached the end
		# of the stream
		if not grabbed:
			break
		# if the frame dimensions are empty, grab them
		if W is None or H is None:
			(H, W) = frame.shape[:2]

		output = frame.copy()
		frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
		frame = cv2.resize(frame, (240, 240)).astype("float32")
		frame -= mean

		preds = model.predict(np.expand_dims(frame, axis=0))[0]
		Q.append(preds)
		# perform prediction averaging over the current history of
		# previous predictions
		results = np.array(Q).mean(axis=0)
		i = np.argmax(results)
		label = lb.classes_[i]

		text = "score: {}".format(label)
		cv2.putText(output, text, (35, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.25, (0, 255, 0), 5)
		# check if the video writer is None
		if writer is None:
			# initialize our video writer
			fourcc = cv2.VideoWriter_fourcc(*"MJPG")
			# writer = cv2.VideoWriter(args["output"], fourcc, 30, (W, H), True)
			writer = cv2.VideoWriter('output/scored_vid.avi', fourcc, 10, (W, H), True)
		# write the output frame to disk
		writer.write(output)
		# show the output image
		cv2.imshow("Output", output)
		key = cv2.waitKey(1) & 0xFF
		# if the `q` key was pressed, break from the loop
		if key == ord("q"):
			break
	# release the file pointers
	print("[INFO] cleaning up...")
	writer.release()
	vs.release()