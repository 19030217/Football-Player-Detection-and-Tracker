###Importing required modules
import numpy as np
import cv2
import time
import os

###Defining some variables and parameters
CONFIDENCE = 0.5
score_threshold = 0.5
IoU_threshold = 0.5

###Neural Network config
config_path = "cfg/yolov3.cfg"
###The YOLO net Weights file
weights_path = "weights/yolov3.weights"

###Loading the class labels (objects)
labels = open("data/coco.names").read().strip().split("\n")
###Generating colors for class labels (objects)
colors = np.random.randint(0, 255, size=(len(labels), 3), dtype="uint8")

###Load the YOLO network
net = cv2.dnn.readNetFromDarknet(config_path, weights_path)

#-----------------Preparing the Image---------------------
###Loading an image
path_name = "images/table.jpg"
image = cv2.imread(path_name)
file_name = os.path.basename(path_name)
filename, ext = file_name.split(".")

###The image we loaded needs to be normalised, scaled and reshaped to be a suitable input for out Neural Network
h, w = image.shape[:2]
###Create 4D blob
blob = cv2.dnn.blobFromImage(image, 1/255.0, (416,416), swapRB=True, crop=False)

print("image.shape:", image.shape)
print("blob.shape:", blob.shape)

#-----------------Making Predictions----------------
###Set blob as input for NN
net.setInput(blob)
###Get layer names
ln = net.getLayerNames()
try:
    ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]
except IndexError:
    ###Incase getUnconnectedOutLayers() returns 1D array when CUDA isn't available
    ln = [ln[i - 1] for i in net.getUnconnectedOutLayers()]
###Feed forward (inference) and get the network output
###Measeure time in seconds
start = time.perf_counter()
layer_output = net.forward(ln)
time_took = time.perf_counter() - start
print(f"Time took: {time_took:.2f}s")

###Iterating over the neural network outputs and discarding any objects with confidence less than 0.5
font_scale = 1
thickness = 2
boxes, confidences, class_ids = [],[],[]
###loop over each layer of outputs
for output in layer_output:
    ###loop over each object detection
    for detection in output:
        ###extract the class_id(label) and confidence(probability)
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]
        ###Discard weak predictions
        if confidence > CONFIDENCE:
            ###scale bounding box coords back relative to size of image
            box = detection[:4] * np.array([w, h, w, h])
            (centerX, centerY, width, height) = box.astype("int")
            ###Use center coords to derive top and left corner of box
            x = int(centerX - (width/2))
            y = int(centerY - (height/2))
            ###Update lists
            boxes.append([x, y, int(width), int(height)])
            confidences.append(float(confidence))
            class_ids.append(class_id)

print(detection.shape)

#--------------Drawing detected objects----------------------
##-------Needs NMS---------
###Loop over indexes we are keeping
# for i in range(len(boxes)):
#     ###extract box coords
#     x, y = boxes[i][0], boxes[i][1]
#     w, h = boxes[i][2], boxes[i][3]
#     ###Draw box and labels
#     color = [int(c) for c in colors[class_ids[i]]]
#     cv2.rectangle(image, (x, y), (x + w, y + h), color=color, thickness=thickness)
#     text = f"{labels[class_ids[i]]}: {confidences[i]:.2f}"
#     ###Calculate transparent box as background of text
#     (text_width, text_height) = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, fontScale=font_scale, thickness=thickness)[0]
#     text_offest_x = x
#     text_offset_y = y - 5
#     box_coords = ((text_offest_x, text_offset_y), (text_offest_x + text_width + 2, text_offset_y - text_height))
#     overlay = image.copy()
#     cv2.rectangle(overlay, box_coords[0], box_coords[1], color=color, thickness=cv2.FILLED)
#     ###add opacity
#     image = cv2.addWeighted(overlay, 0.6, image, 0.4, 0)
#     ###add text
#     cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,fontScale=font_scale, color=(0, 0, 0), thickness=thickness)
#
# cv2.imwrite("output/" + filename + "_yolo3." + ext, image)

#---------------Non-maximal Suppression-----------------------
###perform NMS given the scores defined before
idxs = cv2.dnn.NMSBoxes(boxes,confidences, score_threshold, IoU_threshold)
###Ensure at least one detection exists
if len(idxs) > 0:
    ###loop over the indexes being kept
    for i in idxs.flatten():
        ###extract the bounding box coords
        x, y = boxes[i][0], boxes[i][1]
        w, h = boxes[i][2], boxes[i][3]
        ###draw bounding box rectangle and label
        color = [int(c) for c in colors[class_ids[i]]]
        cv2.rectangle(image, (x, y), (x + w, y + h), color=color, thickness=thickness)
        text = f"{labels[class_ids[i]]}: {confidences[i]:.2f}"
        ###Calculate text width and height to draw the transparent boxes as background of the text
        (text_width, text_height) = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, fontScale=font_scale,thickness=thickness)[0]
        text_offest_x = x
        text_offset_y = y - 5
        box_coords = ((text_offest_x,text_offset_y), (text_offest_x + text_width + 2, text_offset_y - text_height))
        overlay = image.copy()
        cv2.rectangle(overlay, box_coords[0], box_coords[1], color=color, thickness=cv2.FILLED)
        ###Add opacity to box
        image = cv2.addWeighted(overlay, 0.6, image, 0.4, 0)
        ###Add label
        cv2.putText(image, text, (x,y - 5), cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=font_scale,color=(0,0,0), thickness=thickness)
cv2.imwrite("output/" + filename + "_yolo3_NMS." + ext, image)