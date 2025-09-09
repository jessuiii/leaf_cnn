import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np

model = tf.keras.models.load_model('leaf_model.h5')

def preprocess_image(image_path):
    img = Image.open(image_path)
    img = ImageOps.fit(img, (224,224), Image.ANTIALIAS)
    img = np.asarray(img)
    img = img[np.newaxis, ...]
    return img

def predict(image_array):
    pred = model.predict(image_array)
    return np.argmax(pred), pred

