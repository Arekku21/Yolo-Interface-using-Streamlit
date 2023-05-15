from streamlit_webrtc import webrtc_streamer,VideoHTMLAttributes
import av
import torch,streamlit as st, tensorflow as tf, numpy as np,cv2, math
from keras.models import load_model
from skimage.transform import resize

@st.cache_resource
def load_custom_model(model_type):
    # global model

    if model_type =="1 Step":
        model = torch.hub.load('ultralytics/yolov5', 'custom', path='D:\Documents\GitHub\Yolo-Interface-using-Streamlit\models\\fall_detection_custom17.pt', force_reload=False)
        model.conf = 0.4

    elif model_type == "2 Step":
        model = torch.hub.load('ultralytics/yolov5', 'custom', path='D:\Documents\GitHub\Yolo-Interface-using-Streamlit\models\\yolov5s.pt', force_reload=False)
        model.conf = 0.4

        #load tensorflow model
        tf_model = load_model('D:\Documents\GitHub\Yolo-Interface-using-Streamlit\models\\transfer.h5')

    elif model_type == "1 Step Low Light":
        model = torch.hub.load('ultralytics/yolov5', 'custom', path='D:\Documents\GitHub\Yolo-Interface-using-Streamlit\models\\yolov5s.pt', force_reload=False)
        model.conf = 0.4
    return model


def video_frame_callback(frame):
    img = frame.to_ndarray(format="bgr24")

    if model_choice =="1 Step":
        result = model_chosen(img)
        results = result.render()

        st.image(results)

    elif model_choice == "2 Step":
        result = model_chosen(img)
        
        pandas_result = result.pandas().xyxy[0]

        human_pandas = pandas_result.loc[pandas_result['name'] == 'person']

        array_results = human_pandas.to_numpy()

        array_results = array_results.tolist()

        array_bounding_box= []

        for item in array_results:
            array_bounding_box.append([item[0],item[1],item[2],item[3]])

        array_model_result = []

        for boxes in array_bounding_box:
            x1, y1 = int(math.floor(boxes[0])), int(math.floor(boxes[1]))  # top-left corner
            x2, y2 = int(math.floor(boxes[2])), int(math.floor(boxes[3]))  # bottom-right corner

            # crop the region defined by the bounding box
            cropped_img = img[y1:y2, x1:x2]

            #!tensorflow model predict
            tf_model = load_model('D:\Documents\GitHub\Yolo-Interface-using-Streamlit\models\\cnn.h5')

            # resize_img = tf.keras.utils.load_img(frame, target_size=(224,224))
            resized_img = resize(cropped_img, (224, 224), anti_aliasing=True)
            x = tf.keras.utils.img_to_array(resized_img)
            x = np.expand_dims(x, axis=0)

            # images = np.vstack([x])
            classes = np.argmax(tf_model.predict(x), axis=-1)

            classes_name_list = ['falling','walking','sitting']

            predictions = classes_name_list[classes[0]]

            print('Predicted: '+ predictions)

            array_model_result.append(predictions)
        

        for numbers in range(len(array_model_result)):

            x1, y1 = int(math.floor(array_bounding_box[numbers][0])), int(math.floor(array_bounding_box[numbers][1]))  # top-left corner
            x2, y2 = int(math.floor(array_bounding_box[numbers][2])), int(math.floor(array_bounding_box[numbers][3]))  # bottom-right corner

            
            # draw a rectangle over the image using the bounding box coordinates

            if array_model_result[numbers] == "falling":
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0,255), 1)
                cv2.putText(img, array_model_result[numbers], (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)
            elif array_model_result[numbers] == "walking":
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255,0), 1)
                cv2.putText(img, array_model_result[numbers], (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
            elif array_model_result[numbers] == "sitting":
                cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0,0), 1)
                cv2.putText(img, array_model_result[numbers], (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 1)

        st.image(img)

        
        st.write("PERSON FALLING !!!")
        

    elif model_choice == "1 Step Low Light":
        result = model_chosen(img)
        results = result.render()

        st.image(results)
    else:
        st.warning("something wrong")

    

    return av.VideoFrame.from_ndarray(img, format="bgr24")

def main():
    global model_choice,model_chosen,tf_model

    model_choice = st.radio("Choose Model:",["1 Step","2 Step","1 Step Low Light"])

    model_chosen = load_custom_model(model_choice)

    webrtc_streamer(key="example", video_frame_callback=video_frame_callback,
    video_html_attrs=VideoHTMLAttributes(autoPlay=True, controls=True, style={"width": "100%","height" : "100%"}, muted=True)
    )
