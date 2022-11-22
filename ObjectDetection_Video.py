###Import modules
import numpy as np
import cv2
import time
import os
import imutils

###Defining some variables and parameters
CONFIDENCE = 0.5
score_threshold = 0.5
IoU_threshold = 0.5

###Neural Network
config_path = "cfg/yolov3.cfg"
###The YOLO net Weights file
weights_path = "weights/yolov3.weights"

###Load class labels (objects)
labels = open("data/coco.names").read().strip().split("\n")
###Generate colors for objects
colors = np.random.randint(0, 255, size=(len(labels), 3), dtype="uint8")

###Load the YOLO network
net = cv2.dnn.readNetFromDarknet(config_path,weights_path)


###Get layer names
ln = net.getLayerNames()
try:
    ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]
except IndexError:
    ###Incase getUnconnectedOutLayers() returns 1D array when CUDA isn't available
    ln = [ln[i - 1] for i in net.getUnconnectedOutLayers()]

#----------------Video Specific tasks-----------------------
###Load the video
path_name = "videos/tennis.mp4"
video = cv2.VideoCapture(path_name)
writer = None
(W,H) = (None, None)

file_name = os.path.basename(path_name)
filename, ext = file_name.split(".")

###try to determine total number of frames in the video
try:
    prop = cv2.cv.CV_CAP_PROP_FRAME_COUNT if imutils.is_cv2() \
        else cv2.CAP_PROP_FRAME_COUNT
    # def manual_count(handler):
    #     frames = 0
    #     while True:
    #         status, frame = handler.read()
    #         if not status:
    #             break
    #         frames += 1
    #     return frames
    #
    # prop = manual_count(video)
    totalFrame = int(video.get(prop))
    print("{} total frames in video".format(totalFrame))

except:
    print("Could not determine number of frames")
    totalFrame = -1

#----------------Frame Processing-----------------
###Loop over frames from video
while video.isOpened():
    ###Read next frame from the file
    (grabbed, frame) = video.read()
    ###if frame not grabbed, we have reached end of video
    if not grabbed:
        print("test")
        break

    ###if frame dimensions are empty -> grab them
    if W is None or H is None:
        (H, W) = frame.shape[:2]

    ###construct a blob from the frame and then perform a forward pass of the YOLO object detector
    blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416,416),swapRB=True, crop=False)
    net.setInput(blob)
    start = time.time()
    layerOutputs = net.forward(ln)
    end = time.time()

    ###Initialise lists
    font_scale = 1
    thickness = 2
    boxes, confidences, class_ids = [], [], []

    ###loop over each layer of outputs
    for output in layerOutputs:
        ###loop over each object detection
        for detection in output:
            ###extract the class_id(label) and confidence(probability)
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            ###Discard weak predictions
            if confidence > CONFIDENCE:
                ###scale bounding box coords back relative to size of image
                box = detection[:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")
                ###Use center coords to derive top and left corner of box
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))
                ###Update lists
                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                class_ids.append(class_id)

#-------------------Non-maximal suppression--------------------
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, score_threshold, IoU_threshold)
    ###Ensure at least one detection exists
    if len(idxs) > 0:
        ###loop over indexes being kept
        for i in idxs.flatten():
            ###extract bounding box coords
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])
            ###Draw bounding boxes and labels on frame
            color = [int(c) for c in colors[class_ids[i]]]
            cv2.rectangle(frame, (x, y), (x + w, y + h), color=color, thickness=thickness)
            text = f"{labels[class_ids[i]]}: {confidences[i]:.2f}"
            cv2.putText(frame, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=font_scale, color=(0, 0, 0), thickness=thickness)

    ###check if video wirter is None
    if writer is None:
        ###initialise the video writer
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(("output/" + filename + "_yolo3." + ext), fourcc, 30,(frame.shape[1], frame.shape[0]))

        if totalFrame > 0:
            elap = (end - start)
            print("[INFO] single frame took {:.4f} seconds".format(elap))
            print("[INFO] estimated total time to finish: {:.4f}".format(
                elap * totalFrame))
    ###write the output frame to disk
    writer.write(frame)

###Relsease the file pointers
writer.release()
video.release()