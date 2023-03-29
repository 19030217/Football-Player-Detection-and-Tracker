###Import modules
import numpy as np
import cv2
import time
import os
import imutils
import math

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
path_name = "videos/a9f16c_3.mp4"
video = cv2.VideoCapture(path_name)
writer = None
writer2 = None
writer3 = None
writer4 = None

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


# Function to get the kit color from a bounding box, returns BGR value
def get_color(new_box, frame):
    (x1, y1, x2, y2) = new_box

    roi = frame[y1:y2, x1:x2]
    hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    color_ranges = {
        'red': [(0, 70, 50), (10, 255, 255)],
        # 'blue': [(90, 70, 50), (130, 255, 255)],
        # 'yellow': [(20, 70, 50), (40, 255, 255)],
        # 'green': [(50, 70, 50), (70, 255, 255)],
        # 'orange': [(10, 70, 50), (20, 255, 255)],
        'black': [(0, 0, 0), (180, 255, 50)]
    }

    max_area = 0
    max_color = None
    for color, (lower, upper) in color_ranges.items():
        mask = cv2.inRange(hsv_roi, np.array(lower), np.array(upper))
        area = cv2.countNonZero(mask)
        if area > max_area:
            max_area = area
            max_color = color

        color_confidence = max_area / (roi.shape[0] * roi.shape[1])
        if max_color == 'red':
            max_color = (0, 0, 255)
        if max_color == 'black':
            max_color = (0, 0, 0)
        if max_color == 'green':
            max_color = (0, 255, 0)

    return max_color, color_confidence


# Function which updates the tracker by comparing bbox using matchFeatures
def track(box, class_id):
    x, y, w, h = box
    newBox = [x, y, x + w, y + h]
    matchedID = None

    # Find kit color
    kColor, c_confidence = get_color(newBox, frame)

    # checking to see if objects is being tracked
    for j, obj in enumerate(uniqueObjects):
        # time_since_seen = frameCount - obj['lastSeen']
        if match_features(newBox, obj['box']):
            matchedID = j
            break

    # update obj details or create new object
    if matchedID is not None and frameCount > 0:
        obj = uniqueObjects[matchedID]
        obj['box'] = newBox
        obj['lastSeen'] = frameCount
        obj['class_id'] = class_id
        obj['color'] = kColor
    else:
        obj = {'id': len(uniqueObjects),
               'box': newBox,
               'lastSeen': frameCount,
               'class_id': class_id,
               'color': kColor}
        uniqueObjects.append(obj)

    return obj


# -------------- Draws ellipse around detected players feet --------------
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


def drawFrameCount(frame, count):
    # Adding the frame count in the top left of the screen
    cv2.putText(frame, f"Frame Count: {count}", (100, 100), cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=2, color=(20, 20, 20), thickness=3)
    cv2.putText(frame, f"Frame Count: {count}", (100, 100), cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=2, color=(20, 20, 20), thickness=3)
def getMidPoint(contour):
    # extract the left and right highest points from the combined contour
    highest_points = sorted(contour, key=lambda x: (x[0][1], x[0][0]))
    highest_point = tuple(highest_points[0][0])

    midpoint = (highest_points[0][0][0] + highest_points[-1][0][0]) // 2
    left_highest_points = [pt for pt in highest_points if pt[0][0] <= midpoint]
    right_highest_points = [pt for pt in highest_points if pt[0][0] > midpoint]

    leftmost_highest_point = tuple(sorted(left_highest_points, key=lambda x: x[0][0])[0][0])
    rightmost_highest_point = tuple(sorted(right_highest_points, key=lambda x: x[0][0], reverse=True)[0][0])

    if leftmost_highest_point[1] < rightmost_highest_point[1]:
        mid = highest_point
    else:
        mid = leftmost_highest_point

    return mid

def getPitch(im):
    # convert the image to HSV color space
    im_hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)

    # define a lower and upper threshold for green color in HSV
    lower_green = np.array([40, 50, 50])
    upper_green = np.array([80, 255, 255])

    # create a mask based on the green color channel
    mask = cv2.inRange(im_hsv, lower_green, upper_green)

    # apply a morphological opening to the mask to remove small objects
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 10))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    # find contours in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    # sort the contours by area in descending order
    sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)

    # combine the two largest contours into a single contour
    combined_contour = None
    for contour in sorted_contours[:2]:
        if combined_contour is None:
            combined_contour = contour
        else:
            combined_contour = np.concatenate((combined_contour, contour), axis=0)

    # for contour in sorted_contours[:2]:
    mid = getMidPoint(contour)
    # print(f'Mid : {mid}')
    # draw the combined contour on a new image
    im_pitch = im.copy()
    if combined_contour is not None:
        # obtain the convex hull of the contour
        convex_hull = cv2.convexHull(combined_contour)
        # simplify the convex hull to a polygon approximation
        epsilon = 0.005 * cv2.arcLength(convex_hull, True)
        approx = cv2.approxPolyDP(convex_hull, epsilon, True)

        corners = np.squeeze(approx, axis=1)
        print(corners.shape)
        im_pitch = cv2.drawContours(im_pitch, [approx], -1, (0, 0, 255), 3)

        corners_list = [tuple(corners[i]) for i in range(corners.shape[0])]
        # get the top point
        highest_point = min(corners_list, key=lambda p: p[1])

        # get the top-left point
        top_left = min(corners_list, key=lambda p: p[0] + p[1])

        # get the bottom-left point
        bottom_left = max(corners_list, key=lambda p: p[1] - p[0])

        # get the bottom-right point
        bottom_right = max(corners_list, key=lambda p: p[0] + p[1])

        # get the top-right point
        top_right = max(corners_list, key=lambda p: p[0] - p[1])

        # put the corner points in a list in the correct order
        points = np.float32([[highest_point],
                             [top_left],
                             [top_right],
                             [bottom_left],
                             [bottom_right]])
        print(points.shape)
        im_pitch = cv2.circle(im_pitch, top_left, 35, (255, 255, 100), -1)
        im_pitch = cv2.circle(im_pitch, bottom_left, 35, (100, 255, 100), -1)
        im_pitch = cv2.circle(im_pitch, bottom_right, 35, (255, 100, 100), -1)
        im_pitch = cv2.circle(im_pitch, top_right, 35, (255, 100, 255), -1)
        im_pitch = cv2.circle(im_pitch, highest_point, 35, (100, 100, 255), -1)

        print(f'Corners: {corners}')
        print(f'points: {points}')


        # # extract the left and right highest points from the combined contour
        # highest_points = sorted(corners, key=lambda x: (x[0][1], x[0][0]))
        # highest_point = tuple(highest_points[0][0])
        # midpoint = (highest_points[0][0][0] + highest_points[-1][0][0]) // 2
        # left_highest_points = [pt for pt in highest_points if pt[0][0] <= midpoint]
        # right_highest_points = [pt for pt in highest_points if pt[0][0] > midpoint]
        #
        # leftmost_highest_point = tuple(sorted(left_highest_points, key=lambda x: x[0][0])[0][0])
        # rightmost_highest_point = tuple(sorted(right_highest_points, key=lambda x: x[0][0], reverse=True)[0][0])
        #
        # im_pitch = cv2.circle(im_pitch, leftmost_highest_point, 20, (0, 0, 255), -1)
        # im_pitch = cv2.circle(im_pitch, rightmost_highest_point, 20, (255, 0, 0), -1)
        # im_pitch = cv2.circle(im_pitch, highest_point, 20, (255, 0, 255), -1)
        #
        # im_pitch = cv2.circle(im_pitch, mid, 20, (255, 255, 0), -1)
        #
        # # lowest points
        # lowest_points = sorted(corners, key=lambda x: (-x[0][1], x[0][0]))
        # midpoint = (lowest_points[0][0][0] + lowest_points[-1][0][0]) // 2
        # left_lowest_points = [pt for pt in lowest_points if pt[0][0] <= midpoint]
        # right_lowest_points = [pt for pt in lowest_points if pt[0][0] > midpoint]
        #
        # leftmost_lowest_point = tuple(sorted(left_lowest_points, key=lambda x: x[0][0])[0][0])
        # rightmost_lowest_point = tuple(sorted(right_lowest_points, key=lambda x: x[0][0], reverse=True)[0][0])
        #
        # im_pitch = cv2.circle(im_pitch, leftmost_lowest_point, 20, (0, 0, 255), -1)
        # im_pitch = cv2.circle(im_pitch, rightmost_lowest_point, 20, (255, 0, 0), -1)
        #
        # points = np.array([highest_point,
        #                    leftmost_highest_point,
        #                    rightmost_highest_point,
        #                    leftmost_lowest_point,
        #                    rightmost_lowest_point], dtype=np.float32)
    return im_pitch, points, mid


def euclidean_distance(point1, point2):
    x1, y1 = point1[0]
    x2, y2 = point2[0]
    distance = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    print(distance)
    return distance


def getSrcPoints(points):
    t, lt, rt, ll, rl = points
    if euclidean_distance(t, lt) < euclidean_distance(t, rt):
        src_lt = t
        src_rt = rt
        coords = np.float32([[60, 29],      # lt
                         [345, 1170],    # ll
                         [1300, 1170],  # rl
                         [1400, 29]])   # rt
    elif euclidean_distance(t, lt) > euclidean_distance(t, rt):
        src_lt = lt
        src_rt = t
        coords = np.float32([[400, 29],  # lt
                             [700, 1170],  # ll
                             [1300, 1170],  # rl
                             [1400, 29]])  # rt

    src_pts = np.float32([[src_lt],
                          [ll],
                          [rl],
                          [src_rt]]).reshape(4,2)
    print(f'src_pints: {src_pts}')
    return src_pts

def getDistance(mid, points):
    if points[0][0][1] > points[3][0][1]:
        top_dist = euclidean_distance(mid,points[0])
        othertop_dist = euclidean_distance(mid, points[3])
    else:
        top_dist = euclidean_distance(mid, points[3])
        othertop_dist = euclidean_distance(mid, points[0])

    dist_factor = othertop_dist / top_dist

    return dist_factor
def getDstPoints(points, factor):
    lt, ll, rl, rt = points
    print(points)
    if lt[0][1] < rt[0][1]:
        dst_lt = (200, 40)
        dst_rt = (dst_lt[0] + 850 + 850*factor, dst_lt[1])
        dst_ll = (300, 1170)
        dst_rl = (dst_ll[0] + 850 + 850*factor, dst_ll[1])
    else:
        dst_rt = (1760, 40)
        dst_lt = (dst_rt[0] - 850 - 850*factor, dst_rt[1])
        dst_rl = (1600, 1170)
        dst_ll = (dst_rl[0] - 850 - 850*factor, dst_rl[1])

    dst_points = np.float32([[dst_lt],
                             [dst_ll],
                             [dst_rl],
                             [dst_rt]])
    print(dst_points)
    return dst_points


def transform(point, src_pts, dst_points):
    # Coord from frame
    coord1 = np.float32([[12, 321],  # upper left
                         [1416, 1073],  # bottom left
                         [1913, 1032],  # bottom right
                         [1699, 258]])  # top right
    # Coord on new image to transform to
    coord2 = np.float32([[372, 32],  # upper left
                         [1086, 1172],  # bottom left
                         [1297, 1169],  # bottom right
                         [1753, 29]])  # top right

    coord3 = np.float32([[100, 29],      # lt
                         [345, 1170],    # ll
                         [1479, 1170],  # rl
                         [1760, 29]])   # rt
    # Get the tansformation matrix
    transform_matrix = cv2.getPerspectiveTransform(src_pts, coord3)
    # Apply matrix to center point
    transformed_point = cv2.perspectiveTransform(point, transform_matrix)

    return transformed_point


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
    font_scale = 0.8
    thickness = 2
    boxes, confidences, class_ids = [], [], []

    if frameCount % 12 == 0:
        mask_frame = frame.copy()

        masked_im, points, mid = getPitch(mask_frame)

        src_points = getSrcPoints(points)

        # factor = getDistance(mid, points)
        #
        # dst_points = getDstPoints(src_points, factor)

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

    output_img = cv2.imread("images/plain_pitch.png")
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

            # --------------- Tracking detections---------------------
            obj = track(bbox, class_id)
            # print(obj['color'])

            # --------------- Birds Eye View ------------------
            centerPoint = np.float32([(x + (w / 2)), (y + h)]).reshape(1, 1, 2)
            t_point = transform(centerPoint, src_points, dst_points = None)
            u, v = t_point[0][0]

            # Draw a circle at the transformed point location on the transformed image
            radius = 15
            color = obj['color']
            thicknes = -1
            cv2.circle(output_img, (int(u), int(v)), radius, color, thicknes)

            # -------------Draw bounding boxes and labels on frame-----------
            drawEllipse(frame, bbox, obj['color'], thickness)
            # cv2.rectangle(frame, (x, y), (x + w, y + h), color=(255,0,0), thickness=thickness)

            # Adding the frame count in the top left of the screen
            drawFrameCount(frame, frameCount)
            drawFrameCount(output_img, frameCount)

            # writing the ID for each object
            if i < len(class_ids):
                text = f"{obj['id']}"
            cv2.putText(frame, text, (x, y + int(1.8 * h)), cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=font_scale, color=(20, 20, 20), thickness=thickness)
            cv2.putText(output_img, text, (int(u), int(v) + int(h)), cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=font_scale, color=(20, 20, 20), thickness=thickness)
    ###check if video writer is None
    if writer is None and writer2 is None and writer3 is None and writer4 is None:
        ###initialise the video writer
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        # writer for object detection and tracking
        writer = cv2.VideoWriter(("output/" + filename + "_yolo3." + ext), fourcc, 30,
                                 (frame.shape[1], frame.shape[0]))
        # writer for transformed output image
        writer2 = cv2.VideoWriter(("output/" + filename + "_transformed." + ext), fourcc, 30,
                                  (output_img.shape[1], output_img.shape[0]))
        # writer for pitch lines
        # writer3 = cv2.VideoWriter(("output/" + filename + "_contours." + ext), fourcc, 30,
        #                           (im.shape[1], im.shape[0]))
        # writer for whole pitch detection
        writer4 = cv2.VideoWriter(("output/" + filename + "_mask." + ext), fourcc, 30,
                                  (masked_im.shape[1], masked_im.shape[0]))
        if totalFrame > 0:
            elap = (end - start)
            print("[INFO] single frame took {:.4f} seconds".format(elap))
            print("[INFO] estimated total time to finish: {:.4f}".format(
                elap * totalFrame))
    ###write the output frame to disk
    writer.write(frame)
    writer2.write(output_img)
    # writer3.write(im)
    writer4.write(masked_im)
    # print(f'uniqueObjects {uniqueObjects}')
###Relsease the file pointers
writer.release()
writer2.release()
# writer3.release()
writer4.release()
video.release()
