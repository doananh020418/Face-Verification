import numpy as np
import cv2
from deepface.detectors import OpenCvWrapper
import time
# Load the haarcascade file
opencv_path = OpenCvWrapper.get_opencv_path()
face_detector_path = opencv_path + "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(face_detector_path)
cap = cv2.VideoCapture(0)
ptime = 0

while (True):
    ret, frame = cap.read()
    if ret:
        frame = cv2.resize(frame, (600, 400))
        frame  =cv2.flip(frame,1)

        #faces = faceCascade.detectMultiScale2(frame, scaleFactor=1.1, minNeighbors=5, flags=cv2.CASCADE_SCALE_IMAGE)
        faces = faceCascade.detectMultiScale(frame, 1.3, 5)
        for (x, y, w, h) in faces:

            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 255), 1)
        ctime = time.time()
        fps = 1 / (ctime - ptime)
        ptime = ctime
        cv2.putText(frame, f'FPS: {str(int(fps))}', (10, 50), cv2.FONT_HERSHEY_SIMPLEX
                    , 1, (0, 0, 255), 1)
        cv2.imshow("Frame", frame)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()