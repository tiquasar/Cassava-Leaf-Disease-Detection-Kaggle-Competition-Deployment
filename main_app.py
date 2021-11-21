import numpy as np
import streamlit as st
import cv2
from keras.models import load_model
import numpy as np
import pandas as pd
from keras.models import load_model
from keras.preprocessing import image
import numpy as np
from tensorflow.keras.applications import EfficientNetB3
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from tensorflow import keras
from efficientnet.tfkeras import EfficientNetB3
models = list()
models.append(keras.models.load_model('I:/modelsave_0.h5'))
models.append(keras.models.load_model('I:/modelsave_1.h5'))
models.append(keras.models.load_model('I:/modelsave_2.h5'))
models.append(keras.models.load_model('I:/modelsave_3.h5'))
models.append(keras.models.load_model('I:/modelsave_4.h5'))
class_dict = {'Cassava Bacterial Blight (CBB)': 0,
 'Cassava Brown Streak Disease (CBSD)': 1,
 'Cassava Green Mottle (CGM)': 2,
 'assava Mosaic Disease (CMD)': 3,
 'Healthy': 4
 }

class_names = list(class_dict.keys())
st.title("Plant Disease Detection")
st.markdown("Upload an image of the plant leaf")
plant_image = st.file_uploader("Choose an image...", type="jpg")
submit = st.button('Predict')
if submit:
    if plant_image is not None:
        file_bytes = np.asarray(bytearray(plant_image.read()), dtype=np.uint8)
        opencv_image = cv2.imdecode(file_bytes, 1)
        st.image(opencv_image, channels="BGR")
        st.write(opencv_image.shape)
        opencv_image = cv2.resize(opencv_image, (512,512))
        opencv_image.shape = (1,512,512,3)
        img = opencv_image
        img_class2 = np.argmax(models[4].predict(img), axis=-1)
        img_class = img_class2
        img_class_index = img_class.item()                           
        classname = class_names[img_class_index]
        img_prob0 = models[2].predict(img)
        img_prob1 = models[3].predict(img)
        img_prob2 = models[4].predict(img)
        img_prob = (img_prob0+img_prob1+img_prob2)/3
        prediction_prob = img_prob[0].max()
        pred_index = np.where(img_prob[0] == max(img_prob[0]) )
        pred_dict = {"Class":class_names[pred_index[0][0]], "Probability":prediction_prob}
        st.title(str("This is "+str(class_names[pred_index[0][0]])+ " diseased leaf with probability " + str(prediction_prob)))
