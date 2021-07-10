import glob
import os
import time

import cv2
import torch
from facenet_pytorch import MTCNN
from flask import Flask, Response, request, jsonify, render_template
from werkzeug.utils import secure_filename

from deepface import DeepFace
from deepface.Face_recognition_vjpro import add_employee, get_df
from deepface.commons import functions, distance as dst

app = Flask(__name__)
print('loading')
  # use 0 for web camera

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

mtcnn = MTCNN(
    margin=14,
    factor=0.6,
    keep_all=False,
    device=device
)
# ALLOWED_EXTENSIONS = set(['doc','docx', 'pdf', 'png', 'jpg', 'jpeg'])
ALLOWED_EXTENSIONS = set(['jpg'])


def get_model():
    global model
    model = DeepFace.build_model('VGG-Face')


get_model()


def load():
    global base_df
    base_df = get_df(db_path=os.path.abspath('static/hieu'), model=model)


load()


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def gen_frames(df):  # generate frame by frame from camera
    # print(len(df))
    cap = cv2.VideoCapture(0)
    scale = 0.25
    frame_threshold = 10
    time_threshold = 2
    delta = 0.8
    model_name = 'VGG-Face'
    distance_metric = 'euclidean_l2'
    input_shape = (224, 224)
    text_color = (255, 255, 255)
    input_shape_x = input_shape[0]
    input_shape_y = input_shape[1]
    pivot_img_size = 112

    freeze = False
    face_detected = False
    face_included_frames = 0
    freezed_frame = 0
    tic = time.time()
    while True:
        ret, img = cap.read()
        img = cv2.flip(img, 1)
        if img is None:
            break

        threshold = dst.findThreshold(model_name, distance_metric)

        raw_img = img.copy()

        resolution_x = img.shape[1]
        resolution_y = img.shape[0]
        frame = img.copy()
        frame = cv2.resize(frame, (int(frame.shape[1] * scale), int(frame.shape[0] * scale)),
                           interpolation=cv2.INTER_AREA)
        if freeze == False:
            faces, conf = mtcnn.detect(frame)
            if conf[0] == None:
                face_included_frames = 0
        else:
            faces = []

        detected_faces = []
        face_index = 0
        if conf[0] != None:
            for (x, y, w, h) in faces:
                if (w - x) > 100 * scale:
                    face_detected = True
                    if face_index == 0:
                        face_included_frames = face_included_frames + 1  # increase frame for a single face
                    x, y, w, h = int(x / scale), int(y / scale), int((w - x) / scale), int((h - y) / scale)
                    cv2.rectangle(img, (x, y), (x + w, y + h), (255, 67, 67), 1)  # draw rectangle to main image

                    cv2.putText(img, str(frame_threshold - face_included_frames), (int(x + w / 4), int(y + h / 1.5)),
                                cv2.FONT_HERSHEY_SIMPLEX, 4, (255, 255, 255), 2)

                    # -------------------------------------

                    detected_faces.append((x, y, w, h))
                    face_index = face_index + 1

            # -------------------------------------

        if face_detected == True and face_included_frames == frame_threshold and freeze == False:
            freeze = True
            base_img = raw_img.copy()
            detected_faces_final = detected_faces.copy()
            tic = time.time()

        if freeze:

            toc = time.time()
            if (toc - tic) <= time_threshold:

                if freezed_frame == 0:
                    freeze_img = base_img.copy()
                    # freeze_img = np.zeros(resolution, np.uint8) #here, np.uint8 handles showing white area issue

                    for detected_face in detected_faces_final:
                        x = detected_face[0]
                        y = detected_face[1]
                        w = detected_face[2]
                        h = detected_face[3]

                        cv2.rectangle(freeze_img, (x, y), (x + w, y + h), (255, 67, 67),
                                      1)

                        custom_face = base_img[y:y + h, x:x + w]

                        custom_face = functions.preprocess_face(img=custom_face,
                                                                target_size=(input_shape_y, input_shape_x),
                                                                enforce_detection=False, detector_backend='facenet')

                        if custom_face.shape[1:3] == input_shape:
                            if df.shape[0] > 0:
                                img1_representation = model.predict(custom_face)[0, :]

                                def findDistance(row):
                                    distance_metric = row['distance_metric']
                                    img2_representation = row['embedding']
                                    distance = 1000  # initialize very large value
                                    if distance_metric == 'cosine':
                                        distance = dst.findCosineDistance(img1_representation, img2_representation)
                                    elif distance_metric == 'euclidean':
                                        distance = dst.findEuclideanDistance(img1_representation, img2_representation)
                                    elif distance_metric == 'euclidean_l2':
                                        distance = dst.findEuclideanDistance(dst.l2_normalize(img1_representation),
                                                                             dst.l2_normalize(img2_representation))

                                    return distance

                                df['distance'] = df.apply(findDistance, axis=1)
                                df = df.sort_values(by=["distance"])

                                candidate = df.iloc[0]
                                employee_name = candidate['employee']
                                best_distance = candidate['distance']
                                name = candidate['name']
                                print(name)
                                if best_distance <= threshold * delta:
                                    display_img = cv2.imread(employee_name)

                                    display_img = cv2.resize(display_img, (pivot_img_size, pivot_img_size))

                                    label = name

                                    try:
                                        if y - pivot_img_size > 0 and x + w + pivot_img_size < resolution_x:
                                            # top right
                                            freeze_img[y - pivot_img_size:y, x + w:x + w + pivot_img_size] = display_img

                                            overlay = freeze_img.copy();
                                            opacity = 0.4
                                            cv2.rectangle(freeze_img, (x + w, y), (x + w + pivot_img_size, y + 20),
                                                          (46, 200, 255), cv2.FILLED)
                                            cv2.addWeighted(overlay, opacity, freeze_img, 1 - opacity, 0, freeze_img)

                                            cv2.putText(freeze_img, label, (x + w, y + 10), cv2.FONT_HERSHEY_SIMPLEX,
                                                        0.5, text_color, 1)

                                            # connect face and text
                                            cv2.line(freeze_img, (x + int(w / 2), y),
                                                     (x + 3 * int(w / 4), y - int(pivot_img_size / 2)), (255, 67, 67),
                                                     1)
                                            cv2.line(freeze_img, (x + 3 * int(w / 4), y - int(pivot_img_size / 2)),
                                                     (x + w, y - int(pivot_img_size / 2)), (255, 67, 67), 1)

                                        elif y + h + pivot_img_size < resolution_y and x - pivot_img_size > 0:
                                            # bottom left
                                            freeze_img[y + h:y + h + pivot_img_size, x - pivot_img_size:x] = display_img

                                            overlay = freeze_img.copy();
                                            opacity = 0.4
                                            cv2.rectangle(freeze_img, (x - pivot_img_size, y + h - 20), (x, y + h),
                                                          (46, 200, 255), cv2.FILLED)
                                            cv2.addWeighted(overlay, opacity, freeze_img, 1 - opacity, 0, freeze_img)

                                            cv2.putText(freeze_img, label, (x - pivot_img_size, y + h - 10),
                                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1)

                                            # connect face and text
                                            cv2.line(freeze_img, (x + int(w / 2), y + h),
                                                     (x + int(w / 2) - int(w / 4), y + h + int(pivot_img_size / 2)),
                                                     (255, 67, 67), 1)
                                            cv2.line(freeze_img,
                                                     (x + int(w / 2) - int(w / 4), y + h + int(pivot_img_size / 2)),
                                                     (x, y + h + int(pivot_img_size / 2)), (255, 67, 67), 1)

                                        elif y - pivot_img_size > 0 and x - pivot_img_size > 0:
                                            # top left
                                            freeze_img[y - pivot_img_size:y, x - pivot_img_size:x] = display_img

                                            overlay = freeze_img.copy();
                                            opacity = 0.4
                                            cv2.rectangle(freeze_img, (x - pivot_img_size, y), (x, y + 20),
                                                          (46, 200, 255), cv2.FILLED)
                                            cv2.addWeighted(overlay, opacity, freeze_img, 1 - opacity, 0, freeze_img)

                                            cv2.putText(freeze_img, label, (x - pivot_img_size, y + 10),
                                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1)

                                            # connect face and text
                                            cv2.line(freeze_img, (x + int(w / 2), y),
                                                     (x + int(w / 2) - int(w / 4), y - int(pivot_img_size / 2)),
                                                     (255, 67, 67), 1)
                                            cv2.line(freeze_img,
                                                     (x + int(w / 2) - int(w / 4), y - int(pivot_img_size / 2)),
                                                     (x, y - int(pivot_img_size / 2)), (255, 67, 67), 1)

                                        elif x + w + pivot_img_size < resolution_x and y + h + pivot_img_size < resolution_y:
                                            # bottom righ
                                            freeze_img[y + h:y + h + pivot_img_size,
                                            x + w:x + w + pivot_img_size] = display_img

                                            overlay = freeze_img.copy();
                                            opacity = 0.4
                                            cv2.rectangle(freeze_img, (x + w, y + h - 20),
                                                          (x + w + pivot_img_size, y + h), (46, 200, 255), cv2.FILLED)
                                            cv2.addWeighted(overlay, opacity, freeze_img, 1 - opacity, 0, freeze_img)

                                            cv2.putText(freeze_img, label, (x + w, y + h - 10),
                                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1)

                                            # connect face and text
                                            cv2.line(freeze_img, (x + int(w / 2), y + h),
                                                     (x + int(w / 2) + int(w / 4), y + h + int(pivot_img_size / 2)),
                                                     (255, 67, 67), 1)
                                            cv2.line(freeze_img,
                                                     (x + int(w / 2) + int(w / 4), y + h + int(pivot_img_size / 2)),
                                                     (x + w, y + h + int(pivot_img_size / 2)), (255, 67, 67), 1)
                                    except Exception as err:
                                        print(str(err))
                                    t = time.time()
                                else:
                                    t = toc - time_threshold

                        tic = time.time()

                    # -------------------------------

                time_left = int(time_threshold - (toc - tic) + 1)

                cv2.rectangle(freeze_img, (10, 10), (90, 50), (255, 67, 67), -10)
                cv2.putText(freeze_img, str(time_left), (40, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1)

                ret, buffer = cv2.imencode('.jpg', freeze_img)
                frame = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  # concat frame one by one and show result

                freezed_frame = freezed_frame + 1
            else:
                face_detected = False
                face_included_frames = 0
                freeze = False
                freezed_frame = 0
        else:
            ret, buffer = cv2.imencode('.jpg', img)
            img = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + img + b'\r\n')  # concat frame one by one and show result

    cap.release()
    #cv2.destroyAllWindows()


@app.route('/')
def home():
    ret = 'hello'
    return ret


@app.route('/stream')
def streamimg():
    global df
    global base_df
    name = request.args.get('id')
    df = base_df[base_df['name'] == name.strip()]
    print(len(df))
    return render_template('index.html')


@app.route('/upload-image', methods=['GET', 'POST'])
def upload_file_api():
    global base_df
    if request.method == "POST":
        id = request.args.get('id')
        foldername = str(id)
        path = os.path.join(os.path.abspath('static'), foldername)
        if not os.path.exists(path):
            os.mkdir(path)
        else:
            files = glob.glob(path + '/*')
            base_df = base_df.drop(base_df.loc[base_df['name'] == id].index)
            for f in files:
                os.remove(f)
        # check if the post request has the file part
        if 'image' not in request.files:
            resp = jsonify({'message': 'No file part in the request'})
            resp.status_code = 400
            return resp
        files = request.files.getlist('image')
        errors = {}
        success = False
        file_names = []
        for file in files:
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                file.save(os.path.join(path, file.filename))
                file_names.append(filename)
        tic = time.time()
        tmp_df = add_employee(id, path, model=model)
        # tmp_df.to_csv('tmp_df.csv', encoding='utf-8-sig', index=False)
        toc = time.time()
        print("Functions last %d secs" % (toc - tic))
        base_df = base_df.append(tmp_df)
        ret = 'Update employee ' + str(id) + ' successful! \nFound ' + str(len(file_names)) + ' new imgs!'
        print('len base_df', len(base_df))
        print(id)
    sc = jsonify({'message': ret})
    sc.status_code = 200
    return sc


def register():
    global id_reg
    global base_df
    vid = 0
    foldername = str(id_reg)
    path = os.path.join(os.path.abspath('static'), foldername)
    if not os.path.exists(path):
        os.mkdir(path)
    else:
        base_df = base_df.drop(base_df.loc[base_df['name'] == id_reg].index)
        files = glob.glob(path + '/*')
        for f in files:
            os.remove(f)
    scale = 0.5
    ptime = 0
    cap = cv2.VideoCapture(vid)
    count = 0
    frame_count = 0
    while frame_count < 10:
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)
        if ret:
            img = frame.copy()
            img = cv2.resize(img, (int(img.shape[1] * scale), int(img.shape[0] * scale)),
                             interpolation=cv2.INTER_AREA)
            boxes, conf = mtcnn.detect(img)

            if conf[0] != None:
                for (x, y, w, h) in boxes:
                    if (w - x) > 100 * scale:
                        text = f"{conf[0] * 100:.2f}%"
                        x, y, w, h = int(x / scale), int(y / scale), int(w / scale), int(h / scale)
                        if count % 10 == 0:
                            cv2.imwrite(path + '/%d.jpg' % (count), frame)
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
        count = count + 1
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    cap.release()
    tic = time.time()
    tmp_df = add_employee(id_reg, path, model=model)
    base_df = base_df.append(tmp_df)
    toc = time.time()
    print('len base_df: ', len(base_df))
    print('function last {} secs'.format(int(toc - tic)))

    #cv2.destroyAllWindows()



@app.route('/register')
def reg():
    return Response(register(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/reg')
def stream():
    global id_reg
    id_reg = request.args.get('id')
    return render_template('index1.html')


@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(df), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/verify', methods=['POST'])
def verify():
    foldername = 'verify'
    path = os.path.join(os.path.abspath('static'), foldername)
    if not os.path.exists(path):
        os.mkdir(path)
    else:
        files = glob.glob(path + '/*')
        for f in files:
            os.remove(f)
    global model
    model_name = "VGG-Face";
    distance_metric = "euclidean_l2";
    detector_backend = "facenet"

    # ----------------------

    instances = []
    instance = []
    img1 = request.files.get('img1')
    img2 = request.files.get('img2')
    img1.save(os.path.join(path, img1.filename))
    img2.save(os.path.join(path, img2.filename))
    instance.append(os.path.join(path, img1.filename));
    instance.append(os.path.join(path, img2.filename))
    instances.append(instance)

    # --------------------------

    if len(instances) == 0:
        return jsonify({'success': False, 'error': 'you must pass at least one img object in your request'}), 205

    print("Input request has ", len(instances), " pairs to verify")

    # --------------------------

    resp_obj = DeepFace.verify(instances
                               , model=model
                               , distance_metric=distance_metric
                               , detector_backend=detector_backend
                               )
    return resp_obj, 200


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5002, debug=False)
