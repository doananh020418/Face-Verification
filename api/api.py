import warnings
warnings.filterwarnings("ignore")

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

#------------------------------

from flask import Flask, jsonify, request, make_response

import argparse
import uuid
import json
import time
from tqdm import tqdm

#------------------------------

import tensorflow as tf
tf_version = int(tf.__version__.split(".")[0])

#------------------------------

if tf_version == 2:
	import logging
	tf.get_logger().setLevel(logging.ERROR)

#------------------------------

from deepface import DeepFace

#------------------------------

app = Flask(__name__)

#------------------------------

if tf_version == 1:
	graph = tf.get_default_graph()

#------------------------------
#Service API Interface

@app.route('/')
def index():
	return '<h1>Hello, world!</h1>'

@app.route('/analyze', methods=['POST'])
def analyze():

	global graph

	tic = time.time()
	req = request.get_json()
	trx_id = uuid.uuid4()

	#---------------------------

	if tf_version == 1:
		with graph.as_default():
			resp_obj = analyzeWrapper(req, trx_id)
	elif tf_version == 2:
		resp_obj = analyzeWrapper(req, trx_id)

	#---------------------------

	toc = time.time()

	resp_obj["trx_id"] = trx_id
	resp_obj["seconds"] = toc-tic

	return resp_obj, 200

def analyzeWrapper(req, trx_id = 0):
	resp_obj = jsonify({'success': False})

	instances = []
	if "img" in list(req.keys()):
		raw_content = req["img"] #list

		for item in raw_content: #item is in type of dict
			instances.append(item)

	if len(instances) == 0:
		return jsonify({'success': False, 'error': 'you must pass at least one img object in your request'}), 205

	print("Analyzing ", len(instances)," instances")

	#---------------------------

	detector_backend = 'opencv'

	actions= ['emotion', 'age', 'gender', 'race']

	if "actions" in list(req.keys()):
		actions = req["actions"]

	if "detector_backend" in list(req.keys()):
		detector_backend = req["detector_backend"]

	#---------------------------

	try:
		resp_obj = DeepFace.analyze(instances, actions = actions)
	except Exception as err:
		print("Exception: ", str(err))
		return jsonify({'success': False, 'error': str(err)}), 205

	#---------------
	#print(resp_obj)
	return resp_obj

@app.route('/verify', methods=['POST'])
def verify():
	model_name = "VGG-Face";
	distance_metric = "euclidean_l2";
	detector_backend = "facenet"

	# ----------------------

	instances = []
	instance = []
	img1 = request.files.get('img1')

	img2 = request.files.get('img2')
	instance.append(img1);
	instance.append(img2)
	instances.append(instance)

	# --------------------------

	if len(instances) == 0:
		return jsonify({'success': False, 'error': 'you must pass at least one img object in your request'}), 205

	print("Input request of ", trx_id, " has ", len(instances), " pairs to verify")

	# --------------------------


	resp_obj = DeepFace.verify(instances
							   , model_name=model_name
							   , distance_metric=distance_metric
							   , detector_backend=detector_backend
							   )


	return resp_obj, 200



@app.route('/represent', methods=['POST'])
def represent():

	global graph

	tic = time.time()
	req = request.get_json()
	trx_id = uuid.uuid4()

	resp_obj = jsonify({'success': False})

	if tf_version == 1:
		with graph.as_default():
			resp_obj = representWrapper(req, trx_id)
	elif tf_version == 2:
		resp_obj = representWrapper(req, trx_id)

	#--------------------------

	toc =  time.time()

	resp_obj["trx_id"] = trx_id
	resp_obj["seconds"] = toc-tic

	return resp_obj, 200

def representWrapper(req, trx_id = 0):

	resp_obj = jsonify({'success': False})

	#-------------------------------------
	#find out model

	model_name = "VGG-Face"; distance_metric = "cosine"; detector_backend = 'opencv'

	if "model_name" in list(req.keys()):
		model_name = req["model_name"]

	if "detector_backend" in list(req.keys()):
		detector_backend = req["detector_backend"]

	#-------------------------------------
	#retrieve images from request

	img = ""
	if "img" in list(req.keys()):
		img = req["img"] #list
		#print("img: ", img)

	validate_img = False
	if len(img) > 11 and img[0:11] == "data:image/":
		validate_img = True

	if validate_img != True:
		print("invalid image passed!")
		return jsonify({'success': False, 'error': 'you must pass img as base64 encoded string'}), 205

	#-------------------------------------
	#call represent function from the interface

	try:

		embedding = DeepFace.represent(img
			, model_name = model_name
			, detector_backend = detector_backend
		)

	except Exception as err:
		print("Exception: ",str(err))
		resp_obj = jsonify({'success': False, 'error': str(err)}), 205

	#-------------------------------------

	#print("embedding is ", len(embedding)," dimensional vector")
	resp_obj = {}
	resp_obj["embedding"] = embedding

	#-------------------------------------

	return resp_obj

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument(
		'-p', '--port',
		type=int,
		default=5000,
		help='Port of serving api')
	args = parser.parse_args()
	app.run(host='0.0.0.0', port=args.port)
