from facenet_pytorch import MTCNN
from PIL import Image
import torch
import cv2
import time
import glob
from tqdm.notebook import tqdm

device = 'cuda' if torch.cuda.is_available() else 'cpu'


mtcnn = MTCNN(
    margin=14,
    factor=0.6,
    keep_all=True,
    device=device
)
ptime = 0
cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    if ret:
        #frame = cv2.resize(frame, (600, 400))
        frame = cv2.flip(frame,1)
        img = frame.copy()
        img = cv2.resize(img,(int(img.shape[1]*0.25),int(img.shape[0]*0.25)),interpolation = cv2.INTER_AREA)
        # Here we are going to use the facenet detector
        boxes, conf = mtcnn.detect(img)

        # If there is no confidence that in the frame is a face, don't draw a rectangle around it
        if conf[0] != None:
            for (x, y, w, h) in boxes:
                if (w - x) > 50/4:
                    text = f"{conf[0] * 100:.2f}%"
                    x, y, w, h = int(x*4), int(y*4), int(w*4), int(h*4)

                    cv2.putText(frame, text, (x, y - 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (170, 170, 170), 1)
                    cv2.rectangle(frame, (x, y), (w, h), (255, 255, 255), 1)
        ctime = time.time()
        fps = 1 / (ctime - ptime)
        ptime = ctime
        cv2.putText(frame, f'FPS: {str(int(fps))}', (10, 50), cv2.FONT_HERSHEY_SIMPLEX
                    , 1, (0, 0, 255), 1)
    else:
        break

    # Show the result
    # If we were using Google Colab we would use their function cv2_imshow()

    # For displaying images/frames
    cv2.imshow("Frame", frame)
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()