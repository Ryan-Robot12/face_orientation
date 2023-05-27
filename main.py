from FaceTracker import get_data
import cv2

cap = cv2.VideoCapture(0)
# or can do:
# from PIL import Image
# import numpy as np
# frame = np.array(Image.open("filename.jpg"))
# and treat it like frame from cap.read()


while True:
    ret, image = cap.read()

    # image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    data = get_data(image)

    for point in data["left_eye"]["points"]:
        cv2.circle(image, point, 2, (255, 255, 255), 1)
    for point in data["right_eye"]["points"]:
        cv2.circle(image, point, 2, (255, 255, 255), 1)
    for point in data["mouth"]["points"]:
        cv2.circle(image, point, 2, (255, 255, 255), 1)
    print(data["rotation3d"])

    cv2.imshow("Image", image)

    if cv2.waitKey(10) == ord("q"):
        break

