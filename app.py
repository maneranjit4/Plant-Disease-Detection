from flask import Flask, request, jsonify, render_template
import cv2
import numpy as np
import pandas as pd
import math
import keras
import pickle
import os
from keras.models import load_model

import google.generativeai as genai   
import pathlib
import textwrap
from IPython.display import display
from IPython.display import Markdown
import html

os.environ['GOOGLE_API_KEY']="##########"
genai.configure(api_key="############")

app = Flask(__name__, template_folder = "template" ,static_folder="uploads")

Apple_model = load_model('plant_disease_model_resnet50.h5')
Strawberry_model = load_model('plant_disease_model_xception_straw.h5')
Grape_model = load_model("plant_disease_model_vgg16_grape.h5")

ai_model = genai.GenerativeModel('gemini-pro')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods = ['GET','POST'])
def predict():
    global prediction_text
    if request.method == 'POST':
        model_name = request.form.get('crop')
        model_name = model_name.lower()
        imagefile = request.files['imageupload']
        image_path = os.path.join('uploads',imagefile.filename)
        imagefile.save(image_path)
        
        image_size = 200
        img = cv2.imread(image_path)
        img_array = cv2.resize(img,(image_size,image_size))

        img_array = np.array(img_array)
        img_array = img_array/255
        img_array = img_array.reshape(1, 200, 200, 3)

        if model_name == 'apple':
            model = Apple_model
            cate = ['Apple___Cedar_apple_rust', 'Apple___healthy', 'Apple___Black_rot', 'Apple___Apple_scab']
        elif model_name == 'strawberry':
            model = Strawberry_model
            cate = ['Strawberry___healthy', 'Strawberry___Leaf_scorch']
        else:
            model = Grape_model
            cate = ['Grape___Esca_(Black_Measles)', 'Grape___healthy', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)']

        pred = model.predict(img_array, verbose = False)
        pred = pred.argmax(axis = 1)
        op = cate[pred[0]]   
        prediction_text = str(op).replace('___','-').replace('__',' ').replace('_', ' ')
        response = ai_model.generate_content(f"crop disease is {prediction_text}. provide reasons and remedies (donot give output with '**' instead use <b></b>). Always have a title")
        gen_text = response.text
        
    return render_template('index.html', prediction_text = prediction_text, gen_text = gen_text, image_path = imagefile.filename)

if __name__=="__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)
