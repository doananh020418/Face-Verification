import base64
import os
from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf

from deepface.detectors import FaceDetector

tf_version = int(tf.__version__.split(".")[0])

if tf_version == 1:
    from keras.preprocessing.image import load_img, save_img, img_to_array
    from keras.preprocessing import image
elif tf_version == 2:
    from tensorflow.keras.preprocessing.image import load_img, save_img, img_to_array
    from tensorflow.keras.preprocessing import image


# --------------------------------------------------

def initialize_input(img1_path, img2_path=None):
    if type(img1_path) == list:
        bulkProcess = True
        img_list = img1_path.copy()
    else:
        bulkProcess = False

        if (
                (type(img2_path) == str and img2_path != None)  # exact image path, base64 image
                or (isinstance(img2_path, np.ndarray) and img2_path.any())  # numpy array
        ):
            img_list = [[img1_path, img2_path]]
        else:  # analyze function passes just img1_path
            img_list = [img1_path]

    return img_list, bulkProcess


def initializeFolder():
    home = str(Path.home())

    if not os.path.exists(home + "/.deepface"):
        os.mkdir(home + "/.deepface")
        print("Directory ", home, "/.deepface created")

    if not os.path.exists(home + "/.deepface/weights"):
        os.mkdir(home + "/.deepface/weights")
        print("Directory ", home, "/.deepface/weights created")


def loadBase64Img(uri):
    encoded_data = uri.split(',')[1]
    nparr = np.fromstring(base64.b64decode(encoded_data), np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return img


def load_image(img):
    exact_image = False
    if type(img).__module__ == np.__name__:
        exact_image = True

    base64_img = False
    if len(img) > 11 and img[0:11] == "data:image/":
        base64_img = True

    # ---------------------------

    if base64_img == True:
        img = loadBase64Img(img)

    elif exact_image != True:  # image path passed as input
        # if not os.path.isfile(img):
        #     raise ValueError("Confirm that ", img, " exists")

        img = cv2.imread(img)
        if img.shape[0] < img.shape[1]:
            img = cv2.resize(img, (640, 480), interpolation=cv2.INTER_AREA)
        else:
            img = cv2.resize(img, (480, 640), interpolation=cv2.INTER_AREA)

    return img


def detect_face(img, detector_backend='facenet', grayscale=False,enforce_detection=True, align=True):
    img_region = [0, 0, img.shape[0], img.shape[1]]

    # detector stored in a global variable in FaceDetector object.
    # this call should be completed very fast because it will return found in memory
    # it will not build face detector model in each call (consider for loops)
    face_detector = FaceDetector.build_model(detector_backend)

    detected_face, img_region = FaceDetector.detect_face(face_detector, detector_backend, img, align)

    if (isinstance(detected_face, np.ndarray)):
        return detected_face, img_region
    else:
        if detected_face == None:
            if enforce_detection != True:
                return img, img_region
            else:
                raise ValueError(
                    "Face could not be detected. Please confirm that the picture is a face photo or consider to set enforce_detection param to False.")


def preprocess_face(img, target_size=(224, 224), grayscale=False, enforce_detection=True, detector_backend='mtcnn',scale = 0.25,threshold = 100,
                    return_region=False, align=True):
    img = load_image(img)
    base_img = img.copy()

    img, region = detect_face(img=img, detector_backend=detector_backend, grayscale=grayscale,
                              enforce_detection=enforce_detection, align=align)

    # --------------------------

    if img.shape[0] == 0 or img.shape[1] == 0:
        if enforce_detection == True:
            raise ValueError("Detected face shape is ", img.shape,
                             ". Consider to set enforce_detection argument to False.")
        else:
            img = base_img.copy()

    if grayscale == True:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    img = cv2.resize(img, target_size)

    img_pixels = image.img_to_array(img)
    img_pixels = np.expand_dims(img_pixels, axis=0)
    img_pixels /= 255
    if return_region == True:
        return img_pixels, region
    else:
        return img_pixels


def find_input_shape(model):

    input_shape = model.layers[0].input_shape

    if type(input_shape) == list:
        input_shape = input_shape[0][1:3]
    else:
        input_shape = input_shape[1:3]

    if type(input_shape) == list:  # issue 197: some people got array here instead of tuple
        input_shape = tuple(input_shape)

    return input_shape
