from FaceTracker import get_data
import socket
import cv2

# webcam
cap = cv2.VideoCapture(0)

# for controlling an avatar hopefully
# first I need to learn some 3D modelling
# address = ('127.0.0.1', 5066)
# s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
# s.connect(address)

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

    # put text if the eyes are open/closed
    text = f"[{data['left_eye']['is_open']}, {data['right_eye']['is_open']}]"
    cv2.putText(image, text, (25, image.shape[0] - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 1)

    # show the image
    cv2.imshow("Image", image)

    # if user presses q, break
    if cv2.waitKey(1) == ord("q"):
        break
