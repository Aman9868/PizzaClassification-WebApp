
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import load_img
import numpy as np
# Keras
import tensorflow as tf
from tensorflow import keras
# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename


#---------Load Model
def get_model():
    global model
    model=keras.models.load_model('models/model_vgg16.h5')
    print('Model Loaded')

#-----------Load Image
def load_img(img_pth):
    img=image.load_img(img_pth,target_size=(224,224))
    img_arr=image.img_to_array(img)
    img_tensr=np.expand_dims(img_arr,axis=0)
    return img_tensr
def prediction(img_pth):
    new_img=load_img(img_pth)
    preds=model.predict(new_img)
    preds = np.argmax(preds, axis=1)
    if preds == 1:
        return "Its a Pizza."
    else:
        return "Its not a Pizza."
# Define a flask app
app = Flask(__name__)
get_model()
@app.route('/',methods=['GET','POST'])
def home():
    return render_template('base.html')
@app.route('/predict',methods=['GET','POST'])
def predict():
    if request.method=='POST':
        file = request.files['file']
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(file.filename))
        file.save(file_path)
        food=prediction(file_path)
        print(food)
        return render_template('index.html',food = food, user_image = file_path)


if __name__ == '__main__':
    app.run(debug=True)
