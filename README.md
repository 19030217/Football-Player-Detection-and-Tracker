## :soccer: Player detection and tracking using YOLOv3

### :weight_lifting: YOLOv3 Weights file can be downloaded from [darknet](https://pjreddie.com/darknet/yolo/)
The aim of this project is to create a program which can detect multiple football players on a pitch, track them across multiple frames, extract valuable information about the players for use in sports analysis, and then to visualise the game from a bird eye view.

## Functional and Non-functional requirements
-	FR-1: Application will accept a video or image as an input. 
-	FR-2: Application will detect objects in each frame.
-	FR-3: Application will annotate selected detections (Person) with bounding boxes.
-	FR-4: Application will assign new detections with unique ID.
-	FR-5: Application will maintain list of unique tracked objects.
-	FR-6: Application will identify colour of detected player’s kit.
-	FR-7: Application will output original video with annotated bounding boxes and ID.
-	FR-8: Application will output video of 2D football pitch with players as dots.
-	FR-9: Application will calculate statics about the game such as, heatmaps or player speed.

-	NFR-1: Application can process a single frame in under 1 seconds.
-	NFR-2: Application should be easy to maintain and scale with clear and organised code.


## Output Example:

![Transformed](https://user-images.githubusercontent.com/116662024/231203575-fd0a12de-923d-4de0-b8ed-f86e5aea4c11.png)
