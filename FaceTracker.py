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
mouth_indexes = [76, 184, 74, 73, 72, 11, 302, 303, 304, 408, 292, 320, 404, 315, 16, 85, 180, 90, 77]
# inner edge of lip
# mouth_indexes = [78, 183, 42, 41, 38, 12, 268, 271, 272, 407, 292, 325, 319, 403, 316, 15, 86, 179, 89, 96]

# open/closed, open=True
# left, right
last_state = [True, True]

min_consec_frames = 1
current_frames = [0, 0]
# pixel change
change_threshold = 15


def is_open(box, pupil_loc, is_left):
    """
    Checks if an eye is open using some sketchy logic
    :param box: The GRAYSCALE portion of the frame containing the eye
    :param pupil_loc: Location of the pupil
    :param is_left: If it is the left eye or not
    :return: Whether the eye is open
    """
    global current_frames, last_state, min_consec_frames, change_threshold
    index = 0 if is_left else 1
    # y1, y2
    tmp = [0, 0]
    # pupil_loc is returned by cv2.minMaxLoc and is a tuple, therefore unassignable
    loc = list(pupil_loc)
    value = box[loc[1], loc[0]]
    initial = value
    try:
        # move up in the image until the pixel is a different color
        # usually iris --> eyelid
        while abs(value - initial) < change_threshold:
            loc[1] -= 2
            value = box[loc[1], loc[0]]
    except IndexError:
        pass
    tmp[0] = loc[1]
    loc = list(pupil_loc)
    value = box[loc[1], loc[0]]
    initial = value
    try:
        # repeat above except moving down
        while abs(value - initial) < change_threshold:
            loc[1] += 2
            value = box[loc[1], loc[0]]
    except IndexError:
        pass
    tmp[1] = loc[1]
    ratio = (box.shape[1]) / (tmp[1] - tmp[0])

    # This code is terrible, but it works, so I'm not gonna touch it
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
    Gets the face data from an image.
    The data is formatted as follows:
    \n     face: If a face was detected
    | face_bbox: The face bounding box
    | rotation3d: pitch, yaw, roll in a dictionary.
    | Looking up is positive pitch, looking right is positive yaw, head rotation clockwise is positive roll
    | left_eye: A dictionary with the structure {points, bbox, pupil: {center, radius}, is_open}
    | right_eye: A dictionary with the structure {points, bbox, pupil: {center, radius}, is_open}
    | mouth: A dictionary with structure [points, bbox]
    |
    | Bounding boxes are [x1, y1, x2, y2]
    | Pitch/yaw/roll are in degrees
    | All other units are in pixels relative to the top left corner of the image
    :param image: A numpy array of image in color format RGB
    :return: The image data
    """
    # performance yay
    image.flags.writeable = False
    results = face_mesh.process(image)
    image.flags.writeable = True
    # image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    data = {
        "face": False,
        "face_bbox": [],
        "rotation3d": {"roll": 0, "pitch": 0, "yaw": 0},
        "left_eye": {"points": [], "bbox": [], "pupil": {"center": [], "radius": 0}, "is_open": last_state[0]},
        "right_eye": {"points": [], "bbox": [], "pupil": {"center": [], "radius": 0}, "is_open": last_state[1]},
        "mouth": {"points": [], "bbox": []}
    }

    # not my code
    # https://towardsdatascience.com/head-pose-estimation-using-python-d165d3541600
    img_h, img_w, img_c = image.shape
    face_3d = []
    face_2d = []

    if results.multi_face_landmarks:
        data["face"] = True
        face_landmarks = results.multi_face_landmarks[0]
        x1 = min(face_landmarks.landmark, key=lambda p: p.x)
        y1 = max(face_landmarks.landmark, key=lambda p: p.y)
        x2 = max(face_landmarks.landmark, key=lambda p: p.x)
        y2 = max(face_landmarks.landmark, key=lambda p: p.y)
        data["face_bbox"] = [x1, y1, x2, y2]
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

        # _ --> basically ignore this
        _, rot_vec, _ = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_matrix)
        rmat, _ = cv2.Rodrigues(rot_vec)

        # angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)
        angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)
        # offset the angles because they would be wrong otherwise
        x = angles[0] * 720 - 26
        y = (angles[1] * 720) - 5

        # back to my (bad) code
        # right eye, right corner
        l1 = face_landmarks.landmark[33]
        # left eye, left corner
        l2 = face_landmarks.landmark[263]
        # find the roll using trig
        xdiff = (l2.x - l1.x) * img_w
        ydiff = (l2.y - l1.y) * img_h
        z = (math.atan2(ydiff / 2, xdiff / 2) * 180 / math.pi) + 4
        data["rotation3d"] = {"pitch": x, "yaw": -1 * y, "roll": -1 * z}

        # yay list comprehension
        left_eye_points = [(int(face_landmarks.landmark[i].x * img_w), int(face_landmarks.landmark[i].y * img_h)) for i
                           in left_eye_indexes]
        right_eye_points = [(int(face_landmarks.landmark[i].x * img_w), int(face_landmarks.landmark[i].y * img_h)) for i
                            in right_eye_indexes]
        mouth_points = [(int(face_landmarks.landmark[i].x * img_w), int(face_landmarks.landmark[i].y * img_h)) for i in
                        mouth_indexes]

        # find bounding boxes
        y1 = min(left_eye_points, key=lambda p: p[1])[1]
        x1 = min(left_eye_points, key=lambda p: p[0])[0]
        y2 = max(left_eye_points, key=lambda p: p[1])[1]
        x2 = max(left_eye_points, key=lambda p: p[0])[0]
        bbox = [x1, y1, x2, y2]
        # find the pupil (darkest spot in image)
        left_cropped = image[bbox[1]:bbox[3], bbox[0]+2:bbox[2]-2]
        left_cropped = cv2.cvtColor(left_cropped, cv2.COLOR_RGB2GRAY)
        _, _, min_loc, _ = cv2.minMaxLoc(left_cropped)
        data["left_eye"] = {"points": left_eye_points, "bbox": bbox,
                            "pupil": {"center": [min_loc[0]+bbox[0], min_loc[1]+bbox[1]], "radius": 5}, "is_open": is_open(left_cropped, min_loc, True)}

        # repeat above
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

        # find the bounding box again
        y1 = min(mouth_points, key=lambda p: p[1])[1]
        x1 = min(mouth_points, key=lambda p: p[0])[0]
        y2 = max(mouth_points, key=lambda p: p[1])[1]
        x2 = max(mouth_points, key=lambda p: p[0])[0]
        data["mouth"] = {"points": mouth_points, "bbox": [x1, y1, x2, y2]}

        data["face"] = True
    return data
