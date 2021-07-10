from mtcnn import MTCNN
import cv2
import time
detector = MTCNN()
# Load a video, if we were using google colab we would
# need to upload the video to Google Colab
cap = cv2.VideoCapture(0)
ptime = 0
while (True):
    ret, frame = cap.read()
    frame = cv2.resize(frame, (600, 400))
    img = frame.copy()
    img = cv2.resize(img, (int(img.shape[1] * 0.25), int(img.shape[0] * 0.25)), interpolation=cv2.INTER_AREA)
    boxes = detector.detect_faces(img)
    if boxes:

        box = boxes[0]['box']
        conf = boxes[0]['confidence']
        x, y, w, h = box[0]*4, box[1]*4, box[2]*4, box[3]*4

        if conf > 0.5:
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