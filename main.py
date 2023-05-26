import mediapipe as mp
import numpy as np
import math
import cv2

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)
cap = cv2.VideoCapture(0)

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
            ydiff = (l2.y - l1.y) * img_w
            # center x
            # cx = (l1.x * img_w) + xdiff / 2
            z = math.atan2(ydiff/2, xdiff/2) * 180 / math.pi

            print(x, y, z)

            # See where the user's head tilting
            # if y < -10:
            #     text = "Looking Left"
            # elif y > 10:
            #     text = "Looking Right"
            # elif x < -10:
            #     text = "Looking Down"
            # elif x > 10:
            #     text = "Looking up"
            # else:
            #     text = "Forward"

            # Add the text on the image
            cv2.putText(image, "pitch:%.2f, yaw:%.2f, roll:%.2f" % (x, y, z), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            # Display the nose direction
            # nose_3d_projection, jacobian = cv2.projectPoints(nose_3d, rot_vec, trans_vec, cam_matrix, dist_matrix)
            #
            # p1 = (int(nose_2d[0]), int(nose_2d[1]))
            # p2 = (int(nose_3d_projection[0][0][0]), int(nose_3d_projection[0][0][1]))
            #
            # cv2.line(image, p1, p2, (255, 0, 0), 2)

            cv2.imshow('Head Pose Estimation', image)
        if cv2.waitKey(10) == ord("q"):
                break

cap.release()
cv2.destroyAllWindows()
