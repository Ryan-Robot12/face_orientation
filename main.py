from FaceTracker import get_data
import socket
import cv2

# camera object
cap = cv2.VideoCapture(0)

# for controlling an avatar hopefully
# first I need to learn some 3D modelling
# address = ('127.0.0.1', 5066)
# s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
# s.connect(address)

"""
Data from get_data:
face: If a face was detected
face_bbox: The face bounding box
rotation3d: pitch, yaw, roll in a dictionary.
    Up is positive pitch, right is positive yaw, rotation head clockwise is positive roll
left_eye: A dictionary with the structure {points, bbox, pupil: {center, radius}, is_open}
right_eye: A dictionary with the structure {points, bbox, pupil: {center, radius}, is_open}
mouth: A dictionary with structure [points, bbox]

Bounding boxes are [x1, y1, x2, y2]
Pitch/yaw/roll are in degrees
All other units are in pixels relative to the top left corner of the image
"""

while True:
    # read image
    ret, image = cap.read()

    # process image
    data = get_data(image)
    # if no face, loop back to start
    if not data["face"]:
        continue
    # outline the eyes and mouth
    for point in data["left_eye"]["points"]:
        cv2.circle(image, point, 2, (255, 255, 255), 1)
    for point in data["right_eye"]["points"]:
        cv2.circle(image, point, 2, (255, 255, 255), 1)
    for point in data["mouth"]["points"]:
        cv2.circle(image, point, 2, (255, 255, 255), 1)
    # draw a circle around the center
    cv2.circle(image, data["left_eye"]["pupil"]["center"], data["left_eye"]["pupil"]["radius"], (255, 255, 255), 1)
    cv2.circle(image, data["right_eye"]["pupil"]["center"], data["right_eye"]["pupil"]["radius"], (255, 255, 255), 1)
    print(data["rotation3d"])

    # show the image
    cv2.imshow("Image", image)

    # if user presses q, break
    if cv2.waitKey(1) == ord("q"):
        break
