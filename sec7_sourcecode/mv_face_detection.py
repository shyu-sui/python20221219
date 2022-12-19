import mediapipe as mp
import cv2

mp_face_detect = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils
kp_drawing_spec = mp_drawing.DrawingSpec(thickness=3, circle_radius=3, color=(0, 255, 0))
bbox_drawing_spec = mp_drawing.DrawingSpec(thickness=3, color=(0, 0, 255))

cap_file = cv2.VideoCapture('movie1.mp4')

with mp_face_detect.FaceDetection(
    min_detection_confidence=0.5) as face_detection:
    while cap_file.isOpened():
        success, image = cap_file.read()
        if not success:
            print("empty camera frame")
            break
        image = cv2.resize(image, dsize=None, fx=0.3, fy=0.3)
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        results = face_detection.process(rgb_image)

        for detection in results.detections:
            mp_drawing.draw_detection(image, detection,
                                        keypoint_drawing_spec = kp_drawing_spec,
                                        bbox_drawing_spec = bbox_drawing_spec
                                    )
            cv2.imshow('face detection', image)

        if cv2.waitKey(5) & 0xFF == 27:
            break

cap_file.release()