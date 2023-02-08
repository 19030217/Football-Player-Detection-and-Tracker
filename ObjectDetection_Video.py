###Import modules
import numpy as np
import cv2
import time
import os
import imutils
from CentroidTracker import CentroidTracker

###Defining some variables and parameters
CONFIDENCE = 0.5
score_threshold = 0.5
IoU_threshold = 0.2

###Neural Network
config_path = "cfg/yolov3.cfg"
###The YOLO net Weights file
weights_path = "weights/yolov3.weights"

###Load class labels (objects)
labels = open("data/coco.names").read().strip().split("\n")
###Generate colors for objects
colors = np.random.randint(0, 255, size=(len(labels), 3), dtype="uint8")

###Load the YOLO network
net = cv2.dnn.readNetFromDarknet(config_path, weights_path)

###Get layer names
ln = net.getLayerNames()
try:
    ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]
except IndexError:
    ###Incase getUnconnectedOutLayers() returns 1D array when CUDA isn't available
    ln = [ln[i - 1] for i in net.getUnconnectedOutLayers()]

# ----------------Video Specific tasks-----------------------
###Load the video
path_name = "videos/0a2d9b_9_test.mp4"
video = cv2.VideoCapture(path_name)
writer = None
(W, H) = (None, None)

# initialize our centroid tracker and frame dimensions
ct = CentroidTracker()

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
    print("[INFO] {} total frames in video".format(totalFrame))

except:
    print("Could not determine number of frames")
    totalFrame = -1
def match_features(bbox1, bbox2):
    # calculate the area of each bounding box
    threshold = 0.3
    print(f'bbox1 : {bbox1}')
    print(f'bbox2 : {bbox2}')
    area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
    area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])

    # calculate the coordinates of the intersection rectangle
    x1 = max(bbox1[0], bbox2[0])
    y1 = max(bbox1[1], bbox2[1])
    x2 = min(bbox1[2], bbox2[2])
    y2 = min(bbox1[3], bbox2[3])

    # calculate the area of the intersection rectangle
    if x1 < x2 and y1 < y2:
        intersection_area = (x2 - x1) * (y2 - y1)
    else:
        intersection_area = 0

    # calculate the area of union
    union_area = area1 + area2 - intersection_area

    # calculate the IoU
    iou = intersection_area / union_area
    print(f'IoU : {iou}')

    return iou > threshold

# ----------------Frame Processing-----------------
###Loop over frames from video
frameCount = -1
uniqueObjects = []
while video.isOpened():
    ###Read next frame from the file
    (grabbed, frame) = video.read()

    ###Get the frame count and display in console
    if grabbed:
        frameCount += 1
        print("--------------------")
        print(f'Frame number {frameCount}')
        print("--------------------")

    ###if frame not grabbed, we have reached end of video
    if not grabbed:
        break

    ###if frame dimensions are empty -> grab them
    if W is None or H is None:
        (H, W) = frame.shape[:2]

    ###construct a blob from the frame and then perform a forward pass of the YOLO object detector
    blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    start = time.time()
    layerOutputs = net.forward(ln)
    end = time.time()

    ###Initialise lists
    font_scale = 0.6
    thickness = 2
    boxes, confidences, class_ids, rects = [], [], [], []

    ###loop over each layer of outputs
    for output in layerOutputs:
        ###loop over each object detection
        for detection in output:
            ###extract the class_id(label) and confidence(probability)
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if labels[class_id] == 'person':
                ###Discard weak predictions
                if confidence > CONFIDENCE:
                    ###scale bounding box coords back relative to size of image
                    box = detection[:4] * np.array([W, H, W, H])
                    (centerX, centerY, width, height) = box.astype("int")
                    ###Use center coords to derive top and left corner of box
                    x = int(centerX - (width / 2))
                    y = int(centerY - (height / 2))
                    bbox = (x, y, x + width, y + height)
                    rects.append(bbox)
                    ###Update lists
                    boxes.append([x, y, int(width), int(height)])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        # objects = ct.update(rects)
        #
        # # loop over the tracked objects
        # for x, (objectID, centroid) in enumerate(objects.items()):
        #     if objectID > len(objects):
        #         objectID = x
        #     # draw both the ID of the object and the centroid of the
        #     # object on the output frame
        #     text = f"ID {objectID}"
        #     cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10),
        #                 cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        #     cv2.circle(frame, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)

    # -------------------Non-maximal suppression--------------------
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, score_threshold, IoU_threshold)
    ###Ensure at least one detection exists
    if len(idxs) > 0:
        ###loop over indexes being kept
        for i in idxs.flatten():
            ###extract bounding box coords
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])

            ###Get the average color of the kits
            colors = {
                "red": (0, 0, 255),
                "yellow": (0, 255, 255),
                "blue": (255, 0, 0),
                "green": (0, 255, 0),
                "black": (0, 0, 0),
                "white": (255, 255, 255),
            }
            ###Set the lower and upper bounds of some common color ranges
            colors_rng = {
                "red": (np.array([0, 50, 50]), np.array([10, 255, 255])),
                "yellow": (np.array([20, 50, 50]), np.array([30, 255, 255])),
                # "blue": (np.array([110, 50, 50]), np.array([130, 255, 255])),
                # "green": (np.array([50, 50, 50]), np.array([70, 255, 255])),
                "black": (np.array([0, 0, 0]), np.array([180, 255, 30])),
                "white": (np.array([0, 0, 200]), np.array([180, 20, 255])),
            }
            dominant_color = 'blue'
            max_count = 0
            ###Create a binary image where all pixels within the green color range are white
            for color, (lower, upper) in colors_rng.items():
                # Create a mask for the current color
                mask = cv2.inRange(frame, lower, upper)
                # Count the non-black pixels in the mask
                count = cv2.countNonZero(mask[y:y + h // 2, x:x + w])
                # Update the dominant color and count if necessary
                if count > max_count:
                    dominant_color = color
                    max_count = count

            kitColor = colors[dominant_color]

            ###Draw bounding boxes and labels on frame
            # color = [int(c) for c in colors[class_ids[i]]]
            cv2.ellipse(frame,
                        center=(int(x + (w / 2)), (y + h)),
                        axes=(int(w), int(0.35 * w)),
                        angle=0,
                        startAngle=-45,
                        endAngle=235,
                        color=(0, 255, 0),
                        thickness=thickness)
            # cv2.rectangle(frame, (x, y), (x + w, y + h), color=kitColor, thickness=thickness)
            text = f"{labels[class_ids[i]]}"
            cv2.putText(frame, text, (x, y + int(1.5 * h)), cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=font_scale, color=(0,0,0), thickness=thickness)

    ###check if video writer is None
    if writer is None:
        ###initialise the video writer
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(("output/" + filename + "_yolo3." + ext), fourcc, 30, (frame.shape[1], frame.shape[0]))

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
