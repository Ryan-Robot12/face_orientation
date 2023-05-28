from FaceTracker import get_data
import socket
import cv2

cap = cv2.VideoCapture(0)
# or can do:
# from PIL import Image
# import numpy as np
# frame = np.array(Image.open("filename.jpg"))
# and treat it like frame from cap.read()
# address = ('127.0.0.1', 5066)
# s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
# s.connect(address)


def translate(value, leftMin, leftMax, rightMin, rightMax):
    # Figure out how 'wide' each range is
    leftSpan = leftMax - leftMin
    rightSpan = rightMax - rightMin

    # Convert the left range into a 0-1 range (float)
    valueScaled = float(value - leftMin) / float(leftSpan)

    # Convert the 0-1 range into a value in the right range.
    return rightMin + (valueScaled * rightSpan)


while True:
    ret, image = cap.read()

    # image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    data = get_data(image)

    if not data["face"]:
        continue
    for point in data["left_eye"]["points"]:
        cv2.circle(image, point, 2, (255, 255, 255), 1)
    for point in data["right_eye"]["points"]:
        cv2.circle(image, point, 2, (255, 255, 255), 1)
    for point in data["mouth"]["points"]:
        cv2.circle(image, point, 2, (255, 255, 255), 1)
    cv2.circle(image, data["left_eye"]["pupil"]["center"], data["left_eye"]["pupil"]["radius"], (255, 255, 255), 1)
    cv2.circle(image, data["right_eye"]["pupil"]["center"], data["right_eye"]["pupil"]["radius"], (255, 255, 255), 1)

    # min eye aspect ratio, max eye aspect ratio, mouth distance, ???, ???
    # try:
    #     val = data["right_eye"]["aspect_ratio"] / data["left_eye"]["aspect_ratio"]
    #     # val = -1 * translate(data["right_eye"]["aspect_ratio"] / data["left_eye"]["aspect_ratio"], 0.2, 0.6, 0.3, 0.4)
    #     # val += 0.5
    #     msg = '%.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f' % \
    #           (data["rotation3d"]["roll"], data["rotation3d"]["pitch"], data["rotation3d"]["yaw"],
    #            min(data["left_eye"]["aspect_ratio"], data["right_eye"]["aspect_ratio"]),
    #            data["mouth"]["aspect_ratio"],
    #            val,
    #            0, 0)
    #     # print(1 / max(data["left_eye"]["aspect_ratio"], data["right_eye"]["aspect_ratio"]))
    #     print(val)
    #     # 0.3 (open) to 0.4 (closed)
    #            # max(data["left_eye"]["aspect_ratio"], data["right_eye"]["aspect_ratio"]),
    #            # data["mouth"]["aspect_ratio"],
    #            # data["mouth"]["distance"], 0, 0)
    #     s.send(bytes(msg, "utf-8"))
    # except KeyError:
    #     pass
    # steady_pose[6], steady_pose[7]
    # left_eye = data["left_eye"]["pupil"]
    # cv2.circle(image, left_eye["center"], left_eye["radius"], (255, 255, 255), 1)

    # bbox = data["right_eye"]["bbox"]
    # cv2.rectangle(image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 255, 255), 1)

    cv2.imshow("Image", image)

    if cv2.waitKey(1) == ord("q"):
        break
