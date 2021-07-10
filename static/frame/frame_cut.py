from facenet_pytorch import MTCNN
import torch
import cv2
import time
import glob
import os
import numpy as np

from deepface.Face_recognition_vjpro import add_employee, get_df
from deepface import DeepFace


device = 'cuda' if torch.cuda.is_available() else 'cpu'
import time

mtcnn = MTCNN(
    margin=14,
    factor=0.6,
    keep_all=True,
    device=device
)
path1 = r'C:\Users\doank\PycharmProjects\Face-Verification-2\static\frame'
files = glob.glob(path1+'/*.jpg')
for f in files:
    os.remove(f)

def get_model():
    global model
    model = DeepFace.build_model('VGG-Face')
get_model()
def register(vid,id):
    # tic = time.time()
    foldername = str(id)
    path = os.path.join(os.path.abspath('./'), foldername)
    if not os.path.exists(path):
        os.mkdir(path)
    else:
        files = glob.glob(path + '/*')
        for f in files:
            os.remove(f)
    scale = 0.2
    ptime = 0
    cap = cv2.VideoCapture(vid)
    count = 0
    frame_count = 0
    while frame_count < 10:
    #while True:
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)
        if ret:

            #frame = cv2.resize(frame, (600, 400))

            img = frame.copy()
            img = cv2.resize(img,(int(img.shape[1]*scale),int(img.shape[0]*scale)),interpolation = cv2.INTER_AREA)
            # Here we are going to use the facenet detector
            boxes, conf = mtcnn.detect(img)

            # If there is no confidence that in the frame is a face, don't draw a rectangle around it
            if conf[0] != None:
                for (x, y, w, h) in boxes:
                    #print(w - x)
                    if (w - x) > 100*scale:
                        text = f"{conf[0] * 100:.2f}%"
                        x, y, w, h = int(x/scale), int(y/scale), int(w/scale), int(h/scale)
                        if count % 10 == 0:
                            cv2.imwrite(path+'/%d.jpg' % (count) , frame)
                            print("frame %d saved" % count)
                            frame_count = frame_count + 1
                        cv2.putText(frame, text, (x, y - 20),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (170, 170, 170), 1)
                        cv2.rectangle(frame, (x, y), (w, h), (255, 255, 255), 1)

        ctime = time.time()
        fps = 1 / (ctime - ptime)
        ptime = ctime
        cv2.putText(frame, f'FPS: {str(int(fps))}', (100, 40), cv2.FONT_HERSHEY_SIMPLEX
                    , 1, (255, 67, 67), 1)
        cv2.rectangle(frame, (10, 10), (90, 50), (255, 67, 67), -10)
        cv2.putText(frame, str(frame_count), (40, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255),
                    1)
        count = count +1

        cv2.imshow("Frame", frame)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
    tic = time.time()
    add_employee(id,path,model=model)
    toc = time.time()
    print('function last %d secs' %(toc - tic))

register(0,'vjpro')