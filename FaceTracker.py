import mediapipe as mp
import numpy as np
import math
import cv2

__all__ = ["get_data"]

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)
cap = cv2.VideoCapture(0)

left_eye_indexes = [362, 398, 384, 385, 386, 387, 388, 263, 249, 390, 373, 374, 380, 381, 382]
right_eye_indexes = [33, 246, 161, 160, 159, 158, 157, 173, 133, 154, 153, 145, 144, 163, 7]
# outside of mouth
# mouth_indexes = [61, 185, 40, 39, 87, 0, 367, 269, 270, 409, 291, 375, 321, 314, 17, 84, 181, 91, 146]
# outer edge of lip
# mouth_indexes = [76, 184, 74, 73, 72, 11, 302, 303, 304, 408, 292, 320, 404, 315, 16, 85, 180, 90, 77]
# inner edge of lip
mouth_indexes = [78, 183, 42, 41, 38, 12, 268, 271, 272, 407, 292, 325, 319, 403, 316, 15, 86, 179, 89, 96]

# open/closed, open=True
# left, right
last_state = [True, True]

min_consec_frames = 2
current_frames = [0, 0]
# pixel change
change_threshold = 15


def is_open(box, pupil_loc, is_left):
    global current_frames, last_state, min_consec_frames, change_threshold
    index = 0 if is_left else 1
    # y1, y2
    tmp = [0, 0]
    loc = list(pupil_loc)
    value = box[loc[1], loc[0]]
    initial = value
    try:
        while abs(value - initial) < change_threshold:
            loc[1] -= 2
            value = box[loc[1], loc[0]]
            # cv2.circle(frame, [loc[0] + right_eye[0], loc[1] + right_eye[1]], 2, (255, 255, 255), 1)
    except IndexError:
        pass
    tmp[0] = loc[1]
    loc = list(pupil_loc)
    value = box[loc[1], loc[0]]
    initial = value
    try:
        while abs(value - initial) < change_threshold:
            loc[1] += 2
            value = box[loc[1], loc[0]]
            # cv2.circle(frame, [loc[0] + right_eye[0], loc[1] + right_eye[1]], 2, (255, 255, 255), 1)
    except IndexError:
        pass
    tmp[1] = loc[1]
    ratio = (box.shape[1]) / (tmp[1] - tmp[0])
    # cv2.putText(frame, str(ratio), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 2)
    if ratio >= 4:
        if last_state[index]:
            current_frames[index] += 1
        else:
            current_frames[index] = 0
    else:
        if not last_state[index]:
            current_frames[index] += 1
        else:
            current_frames[index] = 0
    if current_frames[index] > min_consec_frames:
        if ratio < 4:
            last_state[index] = True
        else:
            last_state[index] = False
    return last_state[index]


def get_data(image: np.ndarray):
    """
    Gets the face data from an image
    :param image: A numpy array of image in color format RGB
    :return: The data
    formatted as follows: {rotation3d: {roll, pitch, yaw}, left_eye: {points, bbox, pupil: {center, radius},
    right_eye: {points, bbox, pupil: {center, radius},mouth: {points, bbox}}. bbox = [x1, y1, x2, y2]. Points are pixel
    coordinates. roll/pitch/yaw are in degrees.
    """
    image.flags.writeable = False
    results = face_mesh.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    img_h, img_w, img_c = image.shape
    face_3d = []
    face_2d = []

    data = {
        "face": False,
        "rotation3d": {"roll": 0, "pitch": 0, "yaw": 0},
        "left_eye": {"points": [], "bbox": [], "pupil": {"center": [], "radius": 0}},
        "right_eye": {"points": [], "bbox": [], "pupil": {"center": [], "radius": 0}},
        "mouth": {"points": [], "bbox": []}
    }
    if results.multi_face_landmarks:
        data["face"] = True
        face_landmarks = results.multi_face_landmarks[0]
        for idx, lm in enumerate(face_landmarks.landmark):
            if idx == 33 or idx == 263 or idx == 1 or idx == 61 or idx == 291 or idx == 199:
                # if idx == 1:
                # nose_2d = (lm.x * img_w, lm.y * img_h)
                # nose_3d = (lm.x * img_w, lm.y * img_h, lm.z * 8000)

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

        _, rot_vec, _ = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_matrix)
        rmat, _ = cv2.Rodrigues(rot_vec)

        # angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)
        angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)
        x = angles[0] * 720 - 26
        y = (angles[1] * 720) - 5

        l1 = face_landmarks.landmark[33]
        # left eye left corner
        l2 = face_landmarks.landmark[263]
        xdiff = (l2.x - l1.x) * img_w
        ydiff = (l2.y - l1.y) * img_h
        z = (math.atan2(ydiff / 2, xdiff / 2) * 180 / math.pi) + 4
        data["rotation3d"] = {"pitch": x, "yaw": y, "roll": z}

        left_eye_points = [(int(face_landmarks.landmark[i].x * img_w), int(face_landmarks.landmark[i].y * img_h)) for i
                           in left_eye_indexes]
        right_eye_points = [(int(face_landmarks.landmark[i].x * img_w), int(face_landmarks.landmark[i].y * img_h)) for i
                            in right_eye_indexes]
        mouth_points = [(int(face_landmarks.landmark[i].x * img_w), int(face_landmarks.landmark[i].y * img_h)) for i in
                        mouth_indexes]

        y1 = min(left_eye_points, key=lambda p: p[1])[1]
        x1 = min(left_eye_points, key=lambda p: p[0])[0]
        y2 = max(left_eye_points, key=lambda p: p[1])[1]
        x2 = max(left_eye_points, key=lambda p: p[0])[0]
        bbox = [x1, y1, x2, y2]
        aspect_ratio = (face_landmarks.landmark[263].x - face_landmarks.landmark[362].x) / (
                    face_landmarks.landmark[374].y - face_landmarks.landmark[386].y)
        left_cropped = image[bbox[1]:bbox[3], bbox[0]+2:bbox[2]-2]
        left_cropped = cv2.cvtColor(left_cropped, cv2.COLOR_RGB2GRAY)
        _, _, min_loc, _ = cv2.minMaxLoc(left_cropped)
        data["left_eye"] = {"points": left_eye_points, "bbox": bbox,
                            "pupil": {"center": [min_loc[0]+bbox[0], min_loc[1]+bbox[1]], "radius": 5}, "is_open": is_open(left_cropped, min_loc, True)}
        data["left_eye"]["aspect_ratio"] = aspect_ratio

        y1 = min(right_eye_points, key=lambda p: p[1])[1]
        x1 = min(right_eye_points, key=lambda p: p[0])[0]
        y2 = max(right_eye_points, key=lambda p: p[1])[1]
        x2 = max(right_eye_points, key=lambda p: p[0])[0]
        bbox = [x1, y1, x2, y2]
        right_cropped = image[bbox[1]:bbox[3], bbox[0]+2:bbox[2]-2]
        right_cropped = cv2.cvtColor(right_cropped, cv2.COLOR_RGB2GRAY)
        _, _, min_loc, _ = cv2.minMaxLoc(right_cropped)
        data["right_eye"] = {"points": right_eye_points, "bbox": bbox,
                             "pupil": {"center": [min_loc[0]+bbox[0], min_loc[1]+bbox[1]], "radius": 5}, "is_open": is_open(right_cropped, min_loc, False)}

        # horizontal distance of eye
        aspect_ratio = (face_landmarks.landmark[45].y - face_landmarks.landmark[159].y) / (
                    face_landmarks.landmark[133].x - face_landmarks.landmark[33].x)
        # right_cropped = frame[right_eye[1]:right_eye[3], right_eye[0]:right_eye[2]]
        # right_cropped = cv2.cvtColor(right_cropped, cv2.COLOR_RGB2GRAY)
        # _, _, min_loc, _ = cv2.minMaxLoc(right_cropped)
        # cv2.circle(frame, [min_loc[0]+right_eye[0], min_loc[1]+right_eye[1]], 5, (255, 255, 255), 1)
        data["right_eye"]["aspect_ratio"] = aspect_ratio
        # tmp2 = [x2, y1, y2]

        # y1 = min(tmp1[1], tmp1[2], tmp2[1], tmp2[3])
        # y2 = max(tmp1[1], tmp1[2], tmp2[1], tmp2[3])

        y1 = min(mouth_points, key=lambda p: p[1])[1]
        x1 = min(mouth_points, key=lambda p: p[0])[0]
        y2 = max(mouth_points, key=lambda p: p[1])[1]
        x1 = max(mouth_points, key=lambda p: p[0])[0]
        data["mouth"] = {"points": mouth_points, "bbox": [x1, y1, x2, y2]}
        aspect_ratio = (face_landmarks.landmark[15].y - face_landmarks.landmark[12].y) / (
                    face_landmarks.landmark[292].x - face_landmarks.landmark[78].x)
        # I have no clue what I'm doing
        data["mouth"]["aspect_ratio"] = aspect_ratio
        data["mouth"]["distance"] = (face_landmarks.landmark[292].x - face_landmarks.landmark[78].x) / (
                max(face_landmarks.landmark, key=lambda x: x.x).x - max(face_landmarks.landmark, key=lambda y: y.y).y)

        data["face"] = True
    return data
