import mediapipe as mp
import numpy as np
import math
import cv2

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)
cap = cv2.VideoCapture(0)

left_eye_indexes = [362, 398, 384, 385, 386, 387, 388, 467, 263, 249, 390, 373, 374, 380, 381, 382]
right_eye_indexes = [33, 246, 161, 160, 159, 158, 157, 173, 133, 154, 153, 145, 144, 163, 7]
# mouth_indexes = [61, 185, 40, 39, 87, 0, 367, 269, 270, 409, 291, 375, 321, 314, 17, 84, 181, 91, 146]
mouth_indexes = [76, 184, 74, 73, 72, 11, 302, 303, 304, 408, 292, 320, 404, 315, 16, 85, 180, 90, 77]


def max_(data, index):
    # this will only be used for coordinates within the image so there is no need for negative numbers
    j = 0
    m = 0
    for idx, i in enumerate(data):
        if i[index] > m:
            m = i[index]
            j = idx
    return data[j]


def min_(data, index):
    j = 0
    m = 1000
    for idx, i in enumerate(data):
        if i[index] < m:
            m = i[index]
            j = idx
    return data[j]


while True:
    ret, image = cap.read()

    # image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = face_mesh.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    img_h, img_w, img_c = image.shape
    face_3d = []
    face_2d = []

    if results.multi_face_landmarks:
        # TODO: cleanup
        for face_landmarks in [results.multi_face_landmarks[0]]:
            for idx, lm in enumerate(face_landmarks.landmark):
                if idx == 33 or idx == 263 or idx == 1 or idx == 61 or idx == 291 or idx == 199:
                    if idx == 1:
                        nose_2d = (lm.x * img_w, lm.y * img_h)
                        nose_3d = (lm.x * img_w, lm.y * img_h, lm.z * 8000)

                    x, y = int(lm.x * img_w), int(lm.y * img_h)

                    # Get the 2D Coordinates
                    face_2d.append([x, y])

                    # Get the 3D Coordinates
                    face_3d.append([x, y, lm.z])

                    # Convert it to the NumPy array
            face_2d = np.array(face_2d, dtype=np.float64)

            # Convert it to the NumPy array
            face_3d = np.array(face_3d, dtype=np.float64)
            # The camera matrix
            focal_length = 1 * img_w

            cam_matrix = np.array([[focal_length, 0, img_h / 2],
                                   [0, focal_length, img_w / 2],
                                   [0, 0, 1]])
            dist_matrix = np.zeros((4, 1), dtype=np.float64)
            success, rot_vec, trans_vec = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_matrix)
            rmat, jac = cv2.Rodrigues(rot_vec)
            # Get angles
            angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)
            # Get the y rotation degree
            x = angles[0] * 720 - 26
            y = (angles[1] * 720) - 5
            l1 = face_landmarks.landmark[33]
            # left eye corner
            l2 = face_landmarks.landmark[263]
            xdiff = (l2.x - l1.x) * img_w
            ydiff = (l2.y - l1.y) * img_h
            z = math.atan2(ydiff/2, xdiff/2) * 180 / math.pi
            left_eye_points = [(int(face_landmarks.landmark[i].x * img_w), int(face_landmarks.landmark[i].y * img_h)) for i in left_eye_indexes]
            right_eye_points = [(int(face_landmarks.landmark[i].x * img_w), int(face_landmarks.landmark[i].y * img_h)) for i in right_eye_indexes]
            mouth_points = [(int(face_landmarks.landmark[i].x * img_w), int(face_landmarks.landmark[i].y * img_h)) for i in mouth_indexes]
            for point in left_eye_points:
                cv2.circle(image, point, 2, (255, 255, 255), 1)
            for point in right_eye_points:
                cv2.circle(image, point, 2, (255, 255, 255), 1)
            for point in mouth_points:
                cv2.circle(image, point, 2, (255, 255, 255), 1)

            # print(x, y, z)

            cv2.putText(image, "pitch:%.2f, yaw:%.2f, roll:%.2f" % (x, y, z), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            cv2.imshow('Head Pose Estimation', image)
        if cv2.waitKey(10) == ord("q"):
                break

cap.release()
cv2.destroyAllWindows()
