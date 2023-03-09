###Import modules
import numpy as np
import cv2
import time
import os
import imutils

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
path_name = "videos/121364_9.mp4"
video = cv2.VideoCapture(path_name)
writer = None
(W, H) = (None, None)

file_name = os.path.basename(path_name)
filename, ext = file_name.split(".")

###try to determine total number of frames in the video
try:
    prop = cv2.cv.CV_CAP_PROP_FRAME_COUNT if imutils.is_cv2() \
        else cv2.CAP_PROP_FRAME_COUNT
    totalFrame = int(video.get(prop))
    print("[INFO] {} total frames in video".format(totalFrame))

except:
    print("Could not determine number of frames")
    totalFrame = -1


# -------------------Functions----------------------
# Function which compares to bbox, returns True if intersection > threshold
def match_features(bbox1, bbox2, threshold=0):
    # calculate the area of each bounding box
    # print(f'bbox1 : {bbox1}')
    # print(f'bbox2 : {bbox2}')
    area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
    area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])

    # calculate the coordinates of the intersection rectangle
    x1 = max(bbox1[0], bbox2[0])
    y1 = max(bbox1[1], bbox2[1])
    x2 = min(bbox1[2], bbox2[2])
    y2 = min(bbox1[3], bbox2[3])
    # print(f'x1: {x1}, x2: {x2}, y1: {y1}, y2: {y2}')

    # calculate the area of the intersection rectangle
    intersection_area = max(0, x2 - x1) * max(0, y2 - y1)

    # calculate the area of union
    union_area = area1 + area2 - intersection_area

    # calculate the IoU
    iou = intersection_area / union_area
    # print(f'IoU : {iou}')

    return iou > threshold


# Function which updates the tracker by comparing bbox using matchFeatures
def track(x, y, w, h, class_id):
    newBox = [x, y, x + w, y + h]
    matchedID = None
    for j, obj in enumerate(uniqueObjects):
        # time_since_seen = frameCount - obj['lastSeen']
        if match_features(newBox, obj['box']):
            matchedID = j
            break

    if matchedID is not None and frameCount > 0:
        obj = uniqueObjects[matchedID]
        obj['box'] = newBox
        obj['lastSeen'] = frameCount
        class_id = obj['class_id']
    else:
        obj = {'id': len(uniqueObjects),
               'box': newBox,
               'lastSeen': frameCount,
               'class_id': class_id}
        uniqueObjects.append(obj)

    # print(obj['id'])
    return obj


# Function to find the Region of Interest (ROI)
def roi(frame):
    # Convert the image to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Create a binary mask that covers only the pitch area
    pitch_mask = np.zeros_like(gray)
    pitch_contour = [
        np.array([[340, 350], [-1200, 750], [1920, 930], [1920, 350]])]  # Adjust this contour to cover the pitch area
    cv2.drawContours(pitch_mask, pitch_contour, -1, (255, 255, 255), -1)

    print(frame.shape)
    # Apply the mask to the original image to remove everything outside the white lines
    masked_img = cv2.bitwise_and(frame, frame, mask=pitch_mask)
    return masked_img


# Function to find the kit color
def find_color(x, y, w, h, frame):
    cx = int(x - (w / 2))
    cy = int(y - (h / 2))
    # roi = frame[y:y + h, x:x + w]
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    pixel_center_bbox = hsv_frame[cy, cx]
    rgb_pixel = frame[cy, cx]
    hue_value = pixel_center_bbox[0]
    print(f'Hue {hue_value}')
    print(f'HSV: {pixel_center_bbox}')
    print(f'RGB: {rgb_pixel}')

    if hue_value >= 41:
        color = (255, 0, 0)
    else:
        color = (0, 255, 0)

    return color


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

    # masked_img = roi(frame)
    masked_img = frame
    ###construct a blob from the frame and then perform a forward pass of the YOLO object detector
    blob = cv2.dnn.blobFromImage(masked_img, 1 / 255.0, (416, 416), swapRB=True, crop=False)
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
            ###Discard weak predictions
            if confidence > CONFIDENCE and labels[class_id] == 'person':
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

    # -------------------Non-maximal suppression--------------------
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, score_threshold, IoU_threshold)
    ###Ensure at least one detection exists
    if len(idxs) > 0:
        ###loop over indexes being kept
        for i in idxs.flatten():
            ###extract bounding box coords
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])
            bbox = (x, y, w, h)

            # --------------Color finder----------------
            kColor = find_color(x, y, w, h, frame)
            # print(kColor)
            # --------------- Tracking detections---------------------
            obj = track(x, y, w, h, class_id)
            print(obj['id'])

            # --------------- Birds Eye View ------------------
            src_bbox = np.array([[350, 360],
                                 [1920, 390],
                                 [1920, 910],
                                 [-1200, 750]])

            ###-------------Draw bounding boxes and labels on frame-----------
            cv2.ellipse(masked_img,
                        center=(int(x + (w / 2)), (y + h)),
                        axes=(int(w), int(0.35 * w)),
                        angle=0,
                        startAngle=-45,
                        endAngle=235,
                        color=(0, 255, 0),
                        thickness=thickness)
            if i < len(class_ids):
                text = f"{obj['id']}"
            cv2.putText(masked_img, text, (x, y + int(1.5 * h)), cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=font_scale, color=(20, 20, 20), thickness=thickness)

    ###check if video writer is None
    if writer is None:
        ###initialise the video writer
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(("output/" + filename + "_yolo3." + ext), fourcc, 30,
                                 (masked_img.shape[1], masked_img.shape[0]))

        if totalFrame > 0:
            elap = (end - start)
            print("[INFO] single frame took {:.4f} seconds".format(elap))
            print("[INFO] estimated total time to finish: {:.4f}".format(
                elap * totalFrame))
    ###write the output frame to disk
    writer.write(masked_img)
    print(f'uniqueObjects {uniqueObjects}')
###Relsease the file pointers
writer.release()
video.release()
