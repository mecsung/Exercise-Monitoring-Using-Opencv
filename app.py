import cv2
import mediapipe as mp
import numpy as np
import os
# source Scripts/activate

wCam, hCam = 640, 480

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)

# counter variables
counter = 0
stage = None
index = 0

# call for exercise images
folderPath = "images"
myList = os.listdir(folderPath)
print(myList)
overlayList = []
for imPath in myList:
    image = cv2.imread(f'{folderPath}/{imPath}')
    overlayList.append(image)


def calculate_angle(a, b, c):
    a = np.array(a)  # shoulder
    b = np.array(b)  # elbow
    c = np.array(c)  # wrist

    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians * 180.0/np.pi)
    angle = round(angle, 2)  # round angle by 2 decimal places

    if angle > 180.0:
        angle = 360-angle
    return angle

# mediapipe instance
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        ret, frame = cap.read()

        # overlay image in screen
        h, w, c = overlayList[index].shape
        # 398 + 242 (image width) = 640 (cap width size)
        frame[0:h, 398:w+398] = overlayList[index]

        # recolor to RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        # Detection Process
        results = pose.process(image)

        # recolor to BGR
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # extract landmarks
        try:
            landmarks = results.pose_landmarks.landmark

            # tracking curl of elbow
            shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                        landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                     landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
            wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                     landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]

            # calculate angle function call
            angle = calculate_angle(shoulder, elbow, wrist)

            # visual size
            cv2.putText(image, str(angle),
                        tuple(np.multiply(elbow, [640, 480]).astype(int)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                        )
            # print(landmarks)

            # curl counter
            if angle > 120:
                stage = "down"
                index = 0
            if angle < 50 and stage == "down":
                stage = "up"
                index = 1
                counter += 1
                print(counter)

        except:
            pass

        # render curl counter
        # setup status box
        cv2.rectangle(image, (0, 0), (100, 73), (0, 0, 0), -1)
        cv2.putText(image, "Reps", (15, 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(image, str(counter), (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)

        # render landmarks
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                  mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2),
                                  mp_drawing.DrawingSpec(color=(0, 128, 0), thickness=2, circle_radius=2))

        cv2.imshow('Integrate App', image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()