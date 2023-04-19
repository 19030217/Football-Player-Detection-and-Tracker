import numpy as np
import cv2
import time
import os
import imutils
import math

# Setup  up YOLO Model
labels = open("data/coco.names").read().strip().split("\n")

config_path = "cfg/yolov3.cfg"
weights_path = "weights/yolov3.weights"
net = cv2.dnn.readNetFromDarknet(config_path, weights_path)

# Get layer names
ln = net.getLayerNames()
try:
    ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]
except IndexError:
    ln = [ln[i - 1] for i in net.getUnconnectedOutLayers()]

# ----------------Video Specific tasks-----------------------
path_name = "videos/2e57b9_1.mp4"
video = cv2.VideoCapture(path_name)
writer1 = None
writer2 = None
# writer3 = None
(W, H) = (None, None)
file_name = os.path.basename(path_name)
filename, ext = file_name.split(".")

# Try to determine total number of frames in the video
try:
    prop = cv2.cv.CV_CAP_PROP_FRAME_COUNT if imutils.is_cv2() \
        else cv2.CAP_PROP_FRAME_COUNT
    totalFrame = int(video.get(prop))
    print("[INFO] {} total frames in video".format(totalFrame))
except:
    print("Could not determine number of frames")
    totalFrame = -1


# -------------------Track Functions----------------------
def match_features(bbox1, bbox2, threshold=0):
    """
        Function which compares two bbox's

        :return: True if intersection > threshold
    """
    # calculate the area of each bounding box
    area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
    area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])

    # calculate the coordinates of the intersection rectangle
    x1 = max(bbox1[0], bbox2[0])
    y1 = max(bbox1[1], bbox2[1])
    x2 = min(bbox1[2], bbox2[2])
    y2 = min(bbox1[3], bbox2[3])

    # calculate the area of the intersection rectangle
    intersection_area = max(0, x2 - x1) * max(0, y2 - y1)

    # calculate the area of union
    union_area = area1 + area2 - intersection_area

    # calculate the IoU
    iou = intersection_area / union_area

    return iou > threshold


def track(box, uniqueObjects, frameCount):
    """
        Function which updates the tracker by comparing bbox using matchFeatures

        :return: Updated list of Unique objects
    """
    x, y, w, h = box
    newBox = [x, y, x + w, y + h]
    matchedID = None
    # checking to see if objects is being tracked
    for j, obj in enumerate(uniqueObjects):
        if match_features(newBox, obj['box']):
            if obj['color'] == kColor:
                matchedID = j
                break

    # update obj details or create new object
    if matchedID is not None and frameCount > 0:
        obj = uniqueObjects[matchedID]
        obj['box'] = newBox
        obj['lastSeen'] = frameCount
        obj['color'] = kColor
    else:
        obj = {'id': len(uniqueObjects),
               'box': newBox,
               'lastSeen': frameCount,
               'color': kColor}
        uniqueObjects.append(obj)

    return obj


# ------------------Kit color function---------------------
def get_colour(new_box, frameCopy):
    """
        Function to get the kit color from a bounding box, returns BGR value

        :return: Dominant colour of area inputted
    """
    x, y, w, h = new_box
    newBox = [int(x), int(y), int(x + w), int(y + h)]
    [x1, y1, x2, y2] = newBox

    roi = frameCopy[y1:y2, x1:x2]

    # convert ROI to HSV
    hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    color_ranges = {
        'red': [(0, 70, 50), (10, 255, 255)],
        'blue': [(90, 70, 50), (130, 255, 255)],
        'black': [(0, 0, 0), (180, 255, 50)],
        'white': [(0, 0, 200), (180, 30, 255)]
    }

    max_area = 0
    max_color = None
    for color, (lower, upper) in color_ranges.items():
        # apply binary masks
        mask = cv2.inRange(hsv_roi, np.array(lower), np.array(upper))
        area = cv2.countNonZero(mask)

        if area > max_area:
            max_area = area
            max_color = color

    if max_color == 'red':
        max_color = (0, 0, 255)
    if max_color == 'black':
        max_color = (0, 0, 0)
    if max_color == 'blue':
        max_color = (255, 0, 0)
    if max_color == 'white':
        max_color = (255, 255, 255)

    return max_color


# -------------- Draw function --------------
# function which draw ellipse round base of players feet
def drawEllipse(frame, bbox, color, thickness):
    x, y, w, h = bbox
    cv2.ellipse(frame,
                center=(int(x + (w / 2)), (y + h)),
                axes=(int(w), int(0.35 * w)),
                angle=0,
                startAngle=-45,
                endAngle=235,
                color=(0, 255, 0),
                thickness=thickness)
    # Add the team color to the ellipse
    cv2.ellipse(frame,
                center=(int(x + (w / 2)), (y + h)),
                axes=(int(w), int(0.35 * w)),
                angle=0,
                startAngle=30,
                endAngle=160,
                color=color,
                thickness=thickness)


# draws frame count onto frame
def drawFrameCount(frame, count):
    # Adding the frame count in the top left of the screen
    cv2.putText(frame, f"Frame Count: {count}", (100, 100), cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=2, color=(20, 20, 20), thickness=3)
    cv2.putText(frame, f"Frame Count: {count}", (100, 100), cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=2, color=(20, 20, 20), thickness=3)


# ------------ Functions used in perspective transform ----------
def getPitch(im):
    """
        Function to get the pitch lines and corners

        :return: Points - array of corner points
    """
    # convert the image to HSV color space
    im_hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)

    lower_green = np.array([40, 50, 50])
    upper_green = np.array([80, 255, 255])

    # binary mask for colour green
    maskG = cv2.inRange(im_hsv, lower_green, upper_green)

    # apply a morphological opening to the mask
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (8, 8))
    mask = cv2.morphologyEx(maskG, cv2.MORPH_OPEN, kernel)

    # find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)

    # combine the two largest contours
    # (each half of the pitch visible) into a single contour
    combined_contour = None
    for contour in sorted_contours[:2]:
        if combined_contour is None:
            combined_contour = contour
        else:
            combined_contour = np.concatenate((combined_contour, contour), axis=0)

    # draw the combined contour on a new image
    # im_pitch = im.copy()
    if combined_contour is not None:
        # obtain the convex hull of the contour
        convex_hull = cv2.convexHull(combined_contour)
        epsilon = 0.0045 * cv2.arcLength(convex_hull, True)
        approx = cv2.approxPolyDP(convex_hull, epsilon, True)
        corners = np.squeeze(approx, axis=1)
        # draw polygon approximation
        # im_pitch = cv2.drawContours(im_pitch, [convex_hull], -1, (0, 0, 255), 3)

        corners_list = [tuple(corners[i]) for i in range(corners.shape[0])]
        # get the top corner
        highest_point = min(corners_list, key=lambda p: p[1])
        # get the top-left corner
        top_left = min(corners_list, key=lambda p: p[0] + p[1])
        # get the bottom-left corner
        bottom_left = max(corners_list, key=lambda p: p[1] - p[0])
        # get the bottom-right corner
        bottom_right = max(corners_list, key=lambda p: p[0] + p[1])
        # get the top-right corner
        top_right = max(corners_list, key=lambda p: p[0] - p[1])

        # put the corner points in a list in the correct order
        points = np.float32([[highest_point],
                             [top_left],
                             [top_right],
                             [bottom_left],
                             [bottom_right]])

        # draw coloured dots on each corner
        # im_pitch = cv2.circle(im_pitch, top_left, 35, (255, 255, 100), -1)
        # im_pitch = cv2.circle(im_pitch, bottom_left, 35, (100, 255, 100), -1)
        # im_pitch = cv2.circle(im_pitch, bottom_right, 35, (255, 100, 100), -1)
        # im_pitch = cv2.circle(im_pitch, top_right, 35, (255, 100, 255), -1)
        # im_pitch = cv2.circle(im_pitch, highest_point, 35, (100, 100, 255), -1)
        # drawFrameCount(im_pitch, frameCount)

    # return im_pitch, points
    return points


# Euclidean distance function
def euclidean_distance(point1, point2):
    x1, y1 = point1[0]
    x2, y2 = point2[0]
    distance = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    return distance


def getPoints(points):
    """
        Function to calculate the points used in perspective transform

        :return: src_pts - array of source corner points
                 dst_pts - array of destination corner points
    """
    t, lt, rt, ll, rl = points
    if euclidean_distance(t, lt) < euclidean_distance(t, rt):
        src_lt = t
        src_rt = rt

        top_dist = euclidean_distance(src_lt, src_rt)
        top_dist = round((top_dist / 19) * 16)
        dst_lt = np.array([[100, 29]], dtype=np.float32)
        dst_rt = np.array([[dst_lt[0][0] + top_dist, dst_lt[0][1]]], dtype=np.float32)

    elif euclidean_distance(t, lt) > euclidean_distance(t, rt):
        src_lt = lt
        src_rt = t

        top_dist = euclidean_distance(src_lt, src_rt)
        top_dist = round((top_dist / 19) * 16)
        dst_rt = np.array([[1760, 29]], dtype=np.float32)
        dst_lt = np.array([[dst_rt[0][0] - top_dist, dst_rt[0][1]]], dtype=np.float32)

    dst_ll = np.array([[dst_lt[0][0] + 250, 1170]], dtype=np.float32)
    dst_rl = np.array([[dst_rt[0][0] - 250, 1170]], dtype=np.float32)

    src_pts = np.float32([[src_lt],
                          [ll],
                          [rl],
                          [src_rt]]).reshape(4, 2)

    dst_pts = np.float32([[dst_lt],
                          [dst_ll],
                          [dst_rl],
                          [dst_rt]]).reshape(4, 2)

    return src_pts, dst_pts


def transform(point, src_pts, dst_points):
    """
    Creates the transformation matrix and applys it to the points inputted
    :return: perspective transformed point
    """

    transform_matrix = cv2.getPerspectiveTransform(src_pts, dst_points)
    # Apply matrix to center point
    transformed_point = cv2.perspectiveTransform(point, transform_matrix)

    return transformed_point

# ----------------Frame Processing-----------------
# Loop over frames from video
# Defining some variables and parameters
CONFIDENCE = 0.5
score_threshold = 0.5
IoU_threshold = 0.1
frameCount = -1
uniqueObjects = []

# Read each frame in video
while video.isOpened():
    (grabbed, frame) = video.read()

    # Get the frame count and display in console
    if grabbed:
        frameCount += 1
        print("--------------------")
        print(f'Frame number {frameCount}')
        print("--------------------")

    # if frame not grabbed, we have reached end of video
    if not grabbed:
        break
    # if frame dimensions are empty -> grab them
    if W is None or H is None:
        (H, W) = frame.shape[:2]

    # construct a blob from the frame and perform a forward pass of YOLOv3
    blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    start = time.time()
    layerOutputs = net.forward(ln)
    end = time.time()

    # Update points used for transformation matrix
    if frameCount % 12 == 0:
        mask_frame = frame.copy()
        points = getPitch(mask_frame)
        src_points, dst_points = getPoints(points)
    font_scale = 0.8
    thickness = 2
    boxes, confidences, class_ids = [], [], []
    # loop over each layer of outputs
    for output in layerOutputs:
        # loop over each object detection
        for detection in output:
            # extract the class_id(label) and confidence(probability)
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            # Discard weak predictions
            if confidence > CONFIDENCE and labels[class_id] == 'person':
                # scale bounding box coords back relative to size of image
                box = detection[:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")
                # Use center coords to derive top and left corner of box
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))
                # Update lists
                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    transform_img = cv2.imread("images/plain_pitch.png")

    # -------------------Non-maximal suppression--------------------
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, score_threshold, IoU_threshold)
    if len(idxs) > 0:
        # loop over indexes being kept
        for i in idxs.flatten():
            # extract bounding box coords
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])
            bbox = (x, y, w, h)
            # --------------- Get kit colour--------------
            # Find kit color
            frameCopy = frame.copy()
            kColor = get_colour(bbox, frameCopy)
            # --------------- Tracking detections---------------------
            obj = track(bbox, uniqueObjects, frameCount)

            # --------------- Birds Eye View ------------------
            centerPoint = np.float32([(x + (w / 2)), (y + h)]).reshape(1, 1, 2)
            t_point = transform(centerPoint, src_points, dst_points)
            u, v = t_point[0][0]
            # Draw a circle at the transformed point on the transformed image
            radius = 15
            color = obj['color']
            thicknes = -1
            cv2.circle(transform_img, (int(u), int(v)), radius, color, thicknes)

            # -------------Draw bounding boxes and labels on frame-----------
            drawEllipse(frame, bbox, obj['color'], thickness)
            # cv2.rectangle(frame, (x, y), (x + w, y + h), color=(255,0,0), thickness=thickness)

            # Adding the frame count in the top left of the screen
            drawFrameCount(frame, frameCount)
            drawFrameCount(transform_img, frameCount)

            # -------------Draw ID onto each detected player---------------
            if i < len(class_ids):
                text = f"{obj['id']}"
            cv2.putText(frame, text, (x, y + int(1.8 * h)), cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=font_scale, color=(20, 20, 20), thickness=thickness)

            cv2.putText(transform_img, text, (int(u), int(v) + int(h)), cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=font_scale, color=(20, 20, 20), thickness=thickness)
    # check if video writer is None
    if writer1 is None and writer2 is None:
        # initialise the video writer
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        # writer for object detection and tracking
        writer1 = cv2.VideoWriter(("output/" + filename + "_yolo3." + ext), fourcc, 25,
                                  (frame.shape[1], frame.shape[0]))
        # writer for transformed output image
        writer2 = cv2.VideoWriter(("output/" + filename + "_transformed." + ext), fourcc, 30,
                                  (transform_img.shape[1], transform_img.shape[0]))
        # writer for pitch lines
        # writer3 = cv2.VideoWriter(("output/" + filename + "_mask." + ext), fourcc, 30,
        #                           (masked_im.shape[1], masked_im.shape[0]))
    if totalFrame > 0:
        elap = (end - start)
        print("[INFO] single frame took {:.4f} seconds".format(elap))
        # print("[INFO] estimated total time to finish: {:.4f}".format(
        #     elap * totalFrame))
    # write the output frame to disk
    writer1.write(frame)
    writer2.write(transform_img)
    # writer3.write(masked_im)

# Relsease the file pointers
writer1.release()
writer2.release()
# writer3.release()
video.release()