import glob
import streamlit as st
import wget
from PIL import Image
import torch
import cv2
import os
import time
import pandas
import numpy as np
import pybboxes as pbx
import cv2
import matplotlib.pyplot as plt


# st.set_page_config(layout="wide")

# def infer_image(img, size=None):
#     model.conf = confidence
#     result = model(img, size=size) if size else model(img)
#     result.render()
#     image = Image.fromarray(result.ims[0])
#     return image



# def load_model(path, device):
#     model_ = torch.hub.load('ultralytics/yolov5', 'custom', path=path, force_reload=True)
#     model_.to(device)
#     print("model to ", device)
#     return model_



# def download_model(url):
#     model_file = wget.download(url, out="models")
#     return model_file


# def get_user_model():
#     model_src = st.sidebar.radio("Model source", ["file upload", "url"])
#     model_file = None
#     if model_src == "file upload":
#         model_bytes = st.sidebar.file_uploader("Upload a model file", type=['pt'])
#         if model_bytes:
#             model_file = "models/uploaded_" + model_bytes.name
#             with open(model_file, 'wb') as out:
#                 out.write(model_bytes.read())
#     else:
#         url = st.sidebar.text_input("model url")
#         if url:
#             model_file_ = download_model(url)
#             if model_file_.split(".")[-1] == "pt":
#                 model_file = model_file_

#     return model_file

# # global variables
# global model, confidence, cfg_model_path

# cfg_model_path = 'models/fall_detection_custom17.pt'
# model = None
# confidence = .25

# user_model_path = get_user_model()
# cfg_model_path = user_model_path
# model = load_model(cfg_model_path, "cpu")

# confidence = 0.4

# model.classes = list(model.names.keys())

#!comment here
model = torch.hub.load('ultralytics/yolov5', 'custom', path='D:\Documents\GitHub\Yolo-Interface-using-Streamlit\models\\yolov5s.pt',force_reload=False)
#model = torch.hub.load('ultralytics/yolov5', 'yolov5s', classes=10)
model.conf = 0.4

results = model("download.jpg")

# crops = results.crop(save=True)  # cropped detections dictionary

# print(type(crops))

# print(crops)



#! tensorflow testing

# from keras.models import load_model
# import tensorflow as tf, numpy as np

# from skimage import io
# img = io.imread("download.jpg")
# io.imshow(img)

# tf_model = load_model('D:\Documents\GitHub\Yolo-Interface-using-Streamlit\models\\cnn.h5')

# resize_img = tf.keras.utils.load_img("download.jpg", target_size=(224,224))
# x = tf.keras.utils.img_to_array(resize_img)
# x = np.expand_dims(x, axis=0)

# images = np.vstack([x])
# classes = np.argmax(tf_model.predict(x), axis=-1)

# bird_name_list = ['falling','walking','sitting']

# print('Predicted: '+bird_name_list[classes[0]])
# print('Actual: Black Footed Albatross')

###############################################################################

# #load tensorflow model
# tf_model = load_model('D:\Documents\GitHub\Yolo-Interface-using-Streamlit\models\\transfer.h5')

# labels, cord_thres = results.xyxyn[0][:, -1].numpy(), results.xyxyn[0][:, :-1].numpy()

# pandas_result = results.pandas().xyxy[0]

# human_pandas = pandas_result.loc[pandas_result['name'] == 'person']

# array_results = human_pandas.to_numpy()

# array_results = array_results.tolist()

# array_bounding_box= []

# for item in array_results:
#     array_bounding_box.append([item[0],item[1],item[2],item[3]])

# img = cv2.imread("download.jpg")
# dh, dw,_ = img.shape

# def yolo_to_pascal_voc(x_center, y_center, w, h,  image_w, image_h):
#     w = w * image_w
#     h = h * image_h
#     x1 = ((2 * x_center * image_w) - w)/2
#     y1 = ((2 * y_center * image_h) - h)/2
#     x2 = x1 + w
#     y2 = y1 + h
#     return [x1, y1, x2, y2]

# for item in array_bounding_box:
#     yolo_to_pascal_voc(item[],item[],item[],item[],item[],item[])

# print(boundingbox)

# print(boundingbox)

# print(W,H)

# array_bounding_box_changed= []

# for item in array_bounding_box:
#     boundingbox = pbx.convert_bbox(item, from_type="yolo", to_type="voc", image_size=(300, 300))
#     array_bounding_box_changed.append(boundingbox)


# print(boundingbox)

# cv2.imshow('Cropped_image',img)
# cv2.waitKey(0)
# cv.destroyAllWindows()


# plt.imshow(img)
# plt.show()


# print(type(results))

# print(results.xyxyn)

# print(results)

# print(labels)

# print(cord_thres)


# !!Python program to explain cv2.rectangle() method

# # importing cv2
# import cv2

# # path
# path = 'download.jpg'

# # Reading an image in default mode
# image = cv2.imread(path)

# # Window name in which image is displayed
# window_name = 'Image'

# # Start coordinate, here (5, 5)
# # represents the top left corner of rectangle
# start_point = (84.75525665283203, 23.488630294799805)

# # Ending coordinate, here (220, 220)
# # represents the bottom right corner of rectangle
# end_point = (126.43452453613281, 150.18426513671875)

# # Blue color in BGR
# color = (255, 0, 0)

# # Line thickness of 2 px
# thickness = 2

# # Using cv2.rectangle() method
# # Draw a rectangle with blue line borders of thickness of 2 px
# image = cv2.rectangle(image, start_point, end_point, color, thickness)

# # Displaying the image
# cv2.imshow(window_name, image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


