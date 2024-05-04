import cv2
import matplotlib.pyplot as plt

##
## Some Helper Functions
##


##
## Draw a rectangle on the frame
##
def drawRectangle(frame, bbox):
    bbox_x, bbox_y, bbox_w, bbox_h = bbox

    point_1 = (int(bbox_x), int(bbox_y))
    point_2 = (int(bbox_x + bbox_w), int(bbox_y + bbox_h))

    BLUE_COLOR = (255, 0, 0)

    cv2.rectangle(frame, point_1, point_2, BLUE_COLOR, 2)


##
## Display the rectangle on the frame
##
def displayRectangle(frame, bbox):
    fig_x, fig_y = [15, 15]
    plt.figure(figsize=(fig_x, fig_y))

    frameCopy = frame.copy()
    drawRectangle(frameCopy, bbox)

    frameCopy = cv2.cvtColor(frameCopy, cv2.COLOR_RGB2BGR)

    plt.imshow(frameCopy)
    plt.axis("off")


##
## Some different tracker types include:
##
## BOOSTING
## MIL
## KCF
## TLD
## MEDIANFLOW
## GOTURN
## MOSSE
## CSRT
##

## We'll use GOTURN for this example
tracker = cv2.TrackerGOTURN.create()

## Read the video file and store it in a variable
video = cv2.VideoCapture("race_car.mp4")

## If the video failed to open, then exit the program
if not video.isOpened():
    exit()

else:
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Make the window a quarter of the original size
    cv2.namedWindow("Object Tracker", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Object Tracker", width // 2, height // 2)


## Read the first frame
ok, frame = video.read()

## If the frame was not read, then exit the program
if not ok:
    exit()

## Define the initial bounding box
bbox = (1300, 405, 160, 120)  ## Copy / paste this for our specific case

## Display the bounding box
displayRectangle(frame, bbox)

## Initialize the tracker
ok = tracker.init(frame, bbox)

while True:
    ## Read a new frame
    ok, frame = video.read()

    ## If the frame was not read, then exit the loop
    if not ok:
        break

    ## Update the tracker
    ok, bbox = tracker.update(frame)

    ## If the tracker was updated successfully, then draw the rectangle
    if ok:
        drawRectangle(frame, bbox)

    ## Show the frame
    cv2.imshow("Object Tracker", frame)


## Release the video and destroy the windows
video.release()
cv2.destroyAllWindows()
