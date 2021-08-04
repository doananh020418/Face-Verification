import base64
import glob
import io

import cv2
import numpy as np
import torch
from PIL import Image
from facenet_pytorch import MTCNN
from flask import Flask, request
from flask_socketio import SocketIO

from deepface.Face_recognition_vjpro import *

app = Flask(__name__)

app.config['SECRET_KEY'] = 'secret!'
app.config['DEBUG'] = False

socketio = SocketIO(app)
print('loading')
# use 0 for web camera
frame = 0
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

mtcnn = MTCNN(
    margin=14,
    factor=0.6,
    keep_all=False,
    device=device
)
ALLOWED_EXTENSIONS = set(['jpg'])


def get_model():
    global model
    model = DeepFace.build_model('Facenet')


get_model()


def load():
    global base_df
    base_df = get_df(db_path=os.path.abspath('static'), model=model)


load()


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def base64ToImage(base64_string):
    imgdata = base64.b64decode(base64_string)
    img = cv2.cvtColor(np.array(Image.open(io.BytesIO(imgdata))), cv2.COLOR_BGR2RGB)
    return img


def imageToBase64(image):
    retval, buffer = cv2.imencode('.jpg', image)
    jpg_as_text = base64.b64encode(buffer)
    image_data = jpg_as_text.decode("utf-8")
    image_data = "data:image/jpeg;base64," + str(image_data)
    return image_data


# convert PIL Image to an RGB image( technically a numpy array ) that's compatible with opencv
def toRGB(image):
    return cv2.cvtColor(np.array(image), cv2.COLOR_BGR2RGB)


face_included_frames = 0
frame_count = {}
users = {}
buff = []


@socketio.on('face_verify', namespace='/stream')
def face_verify(input, user_id):
    if user_id not in buff:
        buff.append(user_id)
        frame_count[user_id] = 0
    input = input.split(",")[1]
    users[user_id] = request.sid
    global base_df
    global model
    # df = base_df[base_df['name'] == user_id.strip()]
    frame = base64ToImage(input)
    frame, frame_count[user_id], label = process_single_frame(frame, base_df, model=model,
                                                              face_included_frames=frame_count[user_id])
    frame = imageToBase64(frame)
    if label == user_id:
        # return verify result
        socketio.emit('verify', {'result': True}, to=users[user_id])
        frame_count[user_id] = 0
    elif label != None:
        socketio.emit('verify', {'result': False}, to=users[user_id])
    # return frame verified
    socketio.emit("processed", {'image_data': frame}, to=users[user_id])


count = {}
reg_stt = {}
reg_frame_count = {}


@socketio.on('face_register', namespace='/reg')
def reg(input, user_id):  # add new employees
    global base_df
    global mtcnn
    global model
    if user_id not in buff:
        buff.append(user_id)
        reg_stt[user_id] = False
        reg_frame_count[user_id] = 0
        count[user_id] = 0
    input = input.split(",")[1]
    users[user_id] = request.sid

    foldername = str(user_id)
    path = os.path.join(os.path.abspath('static'), foldername)
    if not os.path.exists(path):
        os.mkdir(path)
    else:
        base_df = base_df[base_df['name'] != user_id]
        files = glob.glob(path + '/*')
        for f in files:
            os.remove(f)

    tmp_df = add_employee(user_id, path, model=model)
    base_df = base_df.append(tmp_df)

    frame = base64ToImage(input)
    reg_frame_count[user_id], count[user_id], frame = reg_frame(user_id, frame, reg_frame_count[user_id],
                                                                count[user_id], path, base_df, model=model,
                                                                mtcnn=mtcnn)
    frame = imageToBase64(frame)
    if reg_frame_count[user_id] == 10:
        count[user_id] = 0
        reg_frame_count[user_id] = 0
        # return status: True mean register completed
        # return processed frame which show the regions including face
        socketio.emit("registered", {'reg_stt': True, 'reg_data': frame}, to=users[user_id])
    else:
        socketio.emit("registered", {'reg_stt': False, 'reg_data': frame}, to=users[user_id])


if __name__ == "__main__":
    print('[INFO] Starting server at http://localhost:5001')
    socketio.run(app=app, host='0.0.0.0', port=5001)
