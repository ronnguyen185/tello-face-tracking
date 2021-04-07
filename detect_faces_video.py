
# import the necessary packages
from __future__ import print_function
#from imutils.video import VideoStream
import numpy as np
import argparse
from imutils import paths
import imutils
import time
import cv2
import os 
import datetime
from threading import Thread


rtsp_stream_link = 'udp://@0.0.0.0:11111'

def videoRecorder():
    # create a VideoWrite object, recoring to ./video.avi
    height, width, _ = frame.shape
    video = cv2.VideoWriter(outvid_file, cv2.VideoWriter_fourcc(*'XVID'), 30.0, (width, height))

    while keepRecording:
        video.write(frame)
        #time.sleep(1 / 30)

    video.release()

# construct the argument parse and parse the arguments (to change confidence interval if wanted)
ap = argparse.ArgumentParser()
ap.add_argument("-c", "--confidence", type=float, default=0.5,help="minimum probability to filter weak detections")
args = vars(ap.parse_args())

path = os.getcwd()
MODEL_DIR = "model"
PROTO = "deploy.prototxt.txt"
CAFFE_MODEL = "res10_300x300_ssd_iter_140000.caffemodel"

model_path = os.path.join(path,MODEL_DIR)
proto_file = os.path.sep.join((model_path, PROTO))
caffe_model_file = os.path.sep.join((model_path, CAFFE_MODEL))

OUTVID_DIR = "output"
outvid_path = os.path.join(path,OUTVID_DIR)
ts = datetime.datetime.now()
outvid_name = "{}.avi".format(ts.strftime("%Y-%m-%d_%H-%M-%S"))
outvid_file = os.path.sep.join((outvid_path, outvid_name))

# load our serialized model from disk
print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe(proto_file,caffe_model_file)

# initialize the video stream and allow the camera sensor to warmup
keepRecording = True
print("[INFO] starting video stream...")

#cap = cv2.VideoCapture(rtsp_stream_link)
cap = cv2.VideoCapture(rtsp_stream_link)

# Check if camera opened successfully
if (cap.isOpened() == False): 
  print("Unable to read camera feed")
 # Define the codec and create VideoWriter object.The output is stored in 'outpy.avi' file.
out = cv2.VideoWriter(outvid_file,cv2.VideoWriter_fourcc(*'XVID'), 30, (int(cap.get(3)),int(cap.get(4))))
#out = cv2.VideoWriter(outvid_file,cv2.VideoWriter_fourcc(*'XVID'), 30, (960,720))
print(int(cap.get(3)),int(cap.get(4)))

time.sleep(2.0)


# loop over the frames from the video stream
while True:
	# grab the frame from the threaded video stream and resize it
	# to have a maximum width of 240 pixels
	ret, frame = cap.read()
	#frame = imutils.resize(frame,height=960, width=720)
 
	# grab the frame dimensions and convert it to a blob
	(h, w) = frame.shape[:2]
	blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,(300, 300), (104.0, 177.0, 123.0))
	cv2.circle(frame,(int(w/2), int(h/2)), 10, (0, 255, 255), 1)
	#print(blob);
	# pass the blob through the network and obtain the detections and
	# predictions
	net.setInput(blob)
	detections = net.forward()
	
	BstartX, BstartY, BendX, BendY, max_size, Bconfidence = [0, 0, 0, 0, 0, 0]
	

	# loop over the detections
	for i in range(0, detections.shape[2]):
		# extract the confidence (i.e., probability) associated with the
		# prediction
		confidence = detections[0, 0, i, 2]

		# filter out weak detections by ensuring the `confidence` is
		# greater than the minimum confidence
		if confidence < args["confidence"]:
			continue

		# compute the (x, y)-coordinates of the bounding box for the
		# object
		box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
		(startX, startY, endX, endY) = box.astype("int")
		
		#Find closest face
		test_area = (endX-startX)*(endY-startY)
		if test_area>=max_size:
			max_size = test_area
			BstartX = startX
			BstartY = startY
			BendX = endX
			BendY = endY
			Bconfidence = confidence
 	# draw the bounding box of the face along with the associated
	# probability
	text = "{:.2f}%".format(Bconfidence * 100)
	y = BstartY - 10 if BstartY - 10 > 10 else BstartY + 10
	cv2.rectangle(frame, (BstartX, BstartY), (BendX, BendY),(0, 0, 255), 2)
	cv2.circle(frame, (int(BstartX + (BendX-BstartX)/2), BstartY + int((BendY-BstartY)/2)), 10, (0, 255, 0), 1)
	area = (BendX - BstartX)*(BendY - BstartY)
	cv2.putText(frame, text, (BstartX, y),cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
	diff_x = w/2 - (BstartX + (BendX-BstartX)/2)
	diff_y = h/2 - (BstartY + (BendY-BstartY)/2)
	cv2.putText(frame,'%d, %d, %d' %(diff_x, diff_y, area), (0,20), cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,255),4,cv2.LINE_AA)

	# # show the output frame
	
	cv2.imshow('frame', frame)

	#frame_resize = imutils.resize(frame,height=640, width=480)
	out.write(frame)
	
	key = cv2.waitKey(1) & 0xFF
 
	#  if the `q` key was pressed, break from the loop
	if key == ord("q"):
		break

# do a bit of cleanup
#keepRecording = False
cap.release()
out.release()
#recorder.join()
cv2.destroyAllWindows()