from __future__ import division, print_function
import sys
import os
import glob
import re
from pathlib import Path

# Import fast.ai Library
from fastai import *
from fastai.vision import *

# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename

# Define a flask app
app = Flask(__name__)

PKL_OR_PTH = 'pkl'
NAME_OF_FILE = 'export.pkl' # Name of your exported file
PATH_TO_MODELS_DIR = Path('models') # by default just use /models in root dir

def setup_model_pkl(path_to_pkl_file, learner_name_to_load):
    defaults.device = torch.device('cpu')
    learn = load_learner(path_to_pkl_file, fname=learner_name_to_load)
    return learn

if PKL_OR_PTH is 'pth':
    classes = ['Apple Braeburn',
      'Apple Golden 1',
      'Apple Golden 2',
      'Apple Golden 3',
      'Apple Granny Smith',
      'Apple Red 1',
      'Apple Red 2',
      'Apple Red 3',
      'Apple Red Delicious',
      'Apple Red Yellow 1',
      'Apple Red Yellow 2',
      'Apricot',
      'Avocado',
      'Avocado ripe',
      'Banana',
      'Banana Lady Finger',
      'Banana Red',
      'Cactus fruit',
      'Cantaloupe 1',
      'Cantaloupe 2',
      'Carambula',
      'Cherry 1',
      'Cherry 2',
      'Cherry Rainier',
      'Cherry Wax Black',
      'Cherry Wax Red',
      'Cherry Wax Yellow',
      'Chestnut',
      'Clementine',
      'Cocos',
      'Dates',
      'Granadilla',
      'Grape Blue',
      'Grape Pink',
      'Grape White',
      'Grape White 2',
      'Grape White 3',
      'Grape White 4',
      'Grapefruit Pink',
      'Grapefruit White',
      'Guava',
      'Hazelnut',
      'Huckleberry',
      'Kaki',
      'Kiwi',
      'Kumquats',
      'Lemon',
      'Lemon Meyer',
      'Limes',
      'Lychee',
      'Mandarine',
      'Mango',
      'Mangostan',
      'Maracuja',
      'Melon Piel de Sapo',
      'Mulberry',
      'Nectarine',
      'Orange',
      'Papaya',
      'Passion Fruit',
      'Peach',
      'Peach 2',
      'Peach Flat',
      'Pear',
      'Pear Abate',
      'Pear Kaiser',
      'Pear Monster',
      'Pear Williams',
      'Pepino',
      'Physalis',
      'Physalis with Husk',
      'Pineapple',
      'Pineapple Mini',
      'Pitahaya Red',
      'Plum',
      'Plum 2',
      'Plum 3',
      'Pomegranate',
      'Pomelo Sweetie',
      'Quince',
      'Rambutan',
      'Raspberry',
      'Redcurrant',
      'Salak',
      'Strawberry',
      'Strawberry Wedge',
      'Tamarillo',
      'Tangelo',
      'Tomato 1',
      'Tomato 2',
      'Tomato 3',
      'Tomato 4',
      'Tomato Cherry Red',
      'Tomato Maroon',
      'Walnut']
    learn = setup_model_pth(PATH_TO_MODELS_DIR, NAME_OF_FILE, classes)
else:
    learn = setup_model_pkl(PATH_TO_MODELS_DIR, NAME_OF_FILE)

def setup_model_pth(path_to_pth_file, learner_name_to_load, classes):
    defaults.device = torch.device('cpu')
    data = ImageDataBunch.single_from_classes(path_to_pth_file, classes, ds_tfms=get_transforms(), size=224).normalize(imagenet_stats)
    learn = create_cnn(data, models.resnet34)
    learn.load(learner_name_to_load)

def model_predict(img_path):
    img = open_image(img_path)
    pred_class,pred_idx,outputs = learn.predict(img)
    print(pred_class)
    return learn.data.classes[pred_idx]


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        # Make prediction
        preds = model_predict(file_path)
        os.remove(file_path)
        return preds
    return 'OK'


if __name__ == '__main__':

    app.run()
