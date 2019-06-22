from flask import Flask, render_template, request, jsonify, Response, redirect, send_from_directory
from PIL import Image
from werkzeug import secure_filename
import os
import model_utils
import numpy as np

UPLOAD_FOLDER = './upload'

app = Flask(__name__)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

model = None
labels_dict = None

@app.route('/', methods=['GET'])
def homepage():
    return('<h1>Home page</h1>')

@app.route('/predict', methods=['GET'])
def prediction_page():
    global model, labels_dict
    # load model
    model, labels_dict = model_utils.init()
    # predict
    return render_template("prediction_page.html")

@app.route('/predict/upload', methods=['POST'])
def upload_file():
    global model, labels_dict
    if request.method == 'POST':
        f = request.files['file']
        filename = secure_filename(f.filename)
        f.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        print(os.path.join(app.config['UPLOAD_FOLDER'], filename))

        # Raw input
        image = Image.open(f)
        image = np.array(image)
        predict_label, prob = model_utils.predict(model, labels_dict, image)

        # Get CAM heat map
        model_utils.visualize_cam(model, model_utils.resize_image(image), last_conv_layer_index=-5, learning_phase=0, show=False, path_to_save=os.path.join(app.config['UPLOAD_FOLDER'], 'cam_' + filename))

        print('Predict label: ', predict_label)
    return render_template("prediction_page.html", image_filename='http://localhost:3333/upload/' + filename, cam_filename='http://localhost:3333/upload/' + 'cam_' + filename, predict_label = predict_label[0], prob=prob[0])

@app.route('/upload/<filename>')
def send_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)


def load_keras_model():

    return -1

def predict_cap(input_cap_image):

    return -1

if __name__ == '__main__':
    app.run(host='0.0.0.0', port= 3333, debug=True)