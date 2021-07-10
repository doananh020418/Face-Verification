from facenet_pytorch import MTCNN
from PIL import Image
import torch
import cv2
import time
import glob
from tqdm.notebook import tqdm
import tensorflow as tf
import numpy as np
tf_version = int(tf.__version__.split(".")[0])


device = 'cuda' if torch.cuda.is_available() else 'cpu'


mtcnn = MTCNN(
    margin=14,
    factor=0.6,
    keep_all=True,
    device=device
)
def register(vid):
    scale = 1
    ptime = 0
    cap = cv2.VideoCapture(vid)
    count = 0
    frame_count = 0
    while frame_count < 10:
    #while True:
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)
        frame_count = 0
        if ret:
            if  count % 1 == 0:
                #frame = cv2.resize(frame, (600, 400))

                img = frame.copy()
                if img.shape[0] < img.shape[1]:
                    img = cv2.resize(img, (640, 480), interpolation=cv2.INTER_AREA)
                else:
                    img = cv2.resize(img, (480, 640), interpolation=cv2.INTER_AREA)

                img = cv2.resize(img,(int(img.shape[1]*scale),int(img.shape[0]*scale)),interpolation = cv2.INTER_AREA)
                # Here we are going to use the facenet detector
                boxes, conf = mtcnn.detect(img)

                # If there is no confidence that in the frame is a face, don't draw a rectangle around it
                if conf[0] != None:
                    for (x, y, w, h) in boxes:
                        print(w - x)
                        if (w - x) > 100*scale:
                            text = f"{conf[0] * 100:.2f}%"
                            x, y, w, h = int(x/scale), int(y/scale), int(w/scale), int(h/scale)
                            detected_face = frame[int(y):int(h), int(x):int(w)]
                            target_size = (224, 224)


                            frame_count = frame_count + 1
                            cv2.putText(frame, text, (x, y - 20),
                                        cv2.FONT_HERSHEY_SIMPLEX, 1, (170, 170, 170), 1)
                            cv2.rectangle(frame, (x, y), (w, h), (255, 255, 255), 1)
        ctime = time.time()
        fps = 1 / (ctime - ptime)
        ptime = ctime
        cv2.putText(frame, f'FPS: {str(int(fps))}', (10, 50), cv2.FONT_HERSHEY_SIMPLEX
                    , 1, (0, 0, 255), 1)

        count = count +1

        cv2.imshow("Frame", frame)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

register(0)