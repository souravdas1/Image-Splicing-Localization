import os
import numpy as np
import pandas as pd
import scipy.io
from skimage.transform import resize
import matplotlib.pyplot as plt
import cv2
import keras

import sys
import random
import segmentation_models as sm
import albumentations as A

from flask import Flask, jsonify, request, send_from_directory
from sklearn.externals import joblib
from sklearn import linear_model
from bs4 import BeautifulSoup
import re
from werkzeug.utils import secure_filename
from PIL import Image
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
from flask import send_file

# https://www.tutorialspoint.com/flask
import flask
app = Flask(__name__)

UPLOAD_FOLDER = './uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def hello_world():
    return 'Image Object Detection'


@app.route('/index')
def index():
    return flask.render_template('index.html')


@app.route('/pred', methods=['GET', 'POST'])
def pred():
    # helper function for data visualization
    def visualize(**images):
        """PLot images in one row."""
        n = len(images)
        plt.figure(figsize=(16, 5))
        for i, (name, image) in enumerate(images.items()):
            plt.subplot(1, n, i + 1)
            plt.xticks([])
            plt.yticks([])
            plt.title(' '.join(name.split('_')).title())
            plt.imshow(image)
        plt.savefig(os.path.join(app.config['UPLOAD_FOLDER'], "out.png"))

        
    # helper function for data visualization    
    def denormalize(x):
        """Scale image to range 0..1 for correct plot"""
        x_max = np.percentile(x, 98)
        x_min = np.percentile(x, 2)
        x = (x - x_min) / (x_max - x_min)
        x = x.clip(0, 1)
        return x
    
    def get_preprocessing(preprocessing_fn):
        """Construct preprocessing transform
        Args:
            preprocessing_fn (callbale): data normalization function 
                (can be specific for each pretrained neural network)
        Return:
            transform: albumentations.Compose
        """
        
        _transform = [
            A.Lambda(image=preprocessing_fn),
        ]
        return A.Compose(_transform)
    
    # classes for data loading and preprocessing
    class Dataset:
        """CamVid Dataset. Read images, apply augmentation and preprocessing transformations.
        
        Args:
            images_dir (str): path to images folder
            masks_dir (str): path to segmentation masks folder
            class_values (list): values of classes to extract from segmentation mask
            augmentation (albumentations.Compose): data transfromation pipeline 
                (e.g. flip, scale, etc.)
            preprocessing (albumentations.Compose): data preprocessing 
                (e.g. noralization, shape manipulation, etc.)
        """
        
        CLASSES = {'manipulated': 0, 'non_manipulated': 255}
        
        def __init__(self, images_dir, classes=None, augmentation=None, preprocessing=None, ):
            self.ids = os.listdir(images_dir)
            self.images_fps = [os.path.join(images_dir, image_id) for image_id in self.ids]
            
            # convert str names to class values on masks
            self.class_values = [self.CLASSES[cls] for cls in classes]
            self.preprocessing = preprocessing
        
        def __getitem__(self, i):
            
            # read data
            dim= (512,512)
            
            img = cv2.imread(self.images_fps[i], cv2.COLOR_BGR2RGB)
            image = cv2.resize(img, dim, interpolation = cv2.INTER_AREA) # Downscale without Preserving Aspect Ratio
           
            # apply preprocessing
            if self.preprocessing:
                sample = self.preprocessing(image=image)
                image = sample['image']
            
            image = np.array(image).astype(np.float32) / 255
                       
            return image
            
        def __len__(self):
            return len(self.images_fps)
        
        
    class Dataloder(keras.utils.Sequence):
        """Load data from dataset and form batches
        
        Args:
            dataset: instance of Dataset class for image loading and preprocessing.
            batch_size: Integet number of images in batch.
            shuffle: Boolean, if `True` shuffle image indexes each epoch.
        """
        
        def __init__(self, dataset, batch_size=1, shuffle=False):
            self.dataset = dataset
            self.batch_size = batch_size
            self.shuffle = shuffle
            self.indexes = np.arange(len(dataset))
            self.on_epoch_end()
            
   
        def __getitem__(self, i):
            
            # collect batch data
            start = i * self.batch_size
            stop = (i + 1) * self.batch_size
            data = []
            for j in range(start, stop):
                data.append(self.dataset[j])
            
            # transpose list of lists
            batch = [np.stack(samples, axis=0) for samples in zip(*data)]
            
            return batch
        
        def __len__(self):
            """Denotes the number of batches per epoch"""
            return len(self.indexes) // self.batch_size
        
        def on_epoch_end(self):
            """Callback function to shuffle indexes each epoch"""
            if self.shuffle:
                self.indexes = np.random.permutation(self.indexes)
        
    
    CLASSES = ['manipulated']
    BACKBONE = 'efficientnetb3'
    activation = 'sigmoid'
    n_classes = 1
    preprocess_input = sm.get_preprocessing(BACKBONE)
    model = sm.Unet(BACKBONE, classes=n_classes, activation=activation)
    
    model.load_weights('./best_model_columbia.h5')
    
    filelist = [ f for f in os.listdir(app.config['UPLOAD_FOLDER']) if f.endswith(".png") ]
    for f in filelist:
        os.remove(os.path.join(app.config['UPLOAD_FOLDER'], f))
    
    f = request.files['file']
    f.save(os.path.join(app.config['UPLOAD_FOLDER'], f.filename))

    test_dataset = Dataset(
        UPLOAD_FOLDER,
        classes=CLASSES,
        preprocessing=get_preprocessing(preprocess_input),
    )
    
    n = 2
    ids = np.random.choice(np.arange(len(test_dataset)), size=n)
    
    for i in ids:
        image = test_dataset[i]
        image = np.expand_dims(image, axis=0)
        pr_mask = model.predict(image).round()
        visualize(
            image=denormalize(image.squeeze()),
            pr_mask=pr_mask[..., 0].squeeze(),
        )
        return send_from_directory(app.config['UPLOAD_FOLDER'], "out.png")


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
