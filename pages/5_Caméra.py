import streamlit as st
from streamlit_extras.switch_page_button import switch_page  

import numpy as np

import cv2
import dlib

from mtcnn.mtcnn import MTCNN
import tensorflow as tf
       
    
st.set_page_config(
        page_title="Caméra",
        page_icon="💬",
    )      
st.title("Détection en Temps réel")
st.markdown('------')

def analyze_facial_features(image_array):

    # Initialize the face detector and landmark predictor
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(
        "shape_predictor_81_facee_landmarks.dat"
    )
    gray = cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY)
      # Detect faces in the image
    faces = detector(gray)
    image_with_landmarks = image_array.copy()
    # For each face detected
    for face in faces:
    # Detect landmarks
        landmarks = predictor(gray, face)
    
    # extract the point
        x_top = landmarks.part(71).x
        Y_top = landmarks.part(71).y
        cv2.circle(gray, (x_top, Y_top), 3, (0, 0, 0), -1)
        x_bottom = landmarks.part(8).x
        Y_bottom = landmarks.part(8).y
        cv2.circle(image_with_landmarks, (x_bottom, Y_bottom), 3, (0, 0, 0), -1)
        cv2.circle(image_with_landmarks, (x_top, Y_top), 3, (0, 0, 0), -1)
        cv2.putText(
            image_with_landmarks,
            "D3",
            (int((x_top + x_bottom) / 2), Y_top - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 0),
            1,
        )
        
        # Draw a line D3
        cv2.line(image_with_landmarks, (x_top, Y_top), (x_bottom, Y_bottom), (0, 0, 0), 1)
        # extract forhead landmarks
        x_right_forehead = landmarks.part(75).x
        Y_right_forehead = landmarks.part(75).y
        cv2.circle(image_with_landmarks, (x_right_forehead, Y_right_forehead), 3, (0, 0, 0), -1)
        x_left_forehead = landmarks.part(79).x
        Y_left_forehead = landmarks.part(79).y
        cv2.circle(image_with_landmarks, (x_left_forehead, Y_left_forehead), 3,(0, 0, 0), -1)
        # Draw line D1 for forehead width
        cv2.line(
            image_with_landmarks,
            (x_left_forehead, Y_left_forehead),
            (x_right_forehead, Y_right_forehead),
            (0, 0, 0),
            1,
        )

        cv2.putText(
            image_with_landmarks,
            "D2",
            (x_left_forehead - 50, int((Y_right_forehead + Y_left_forehead) / 2)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 0),
            1,
        )
        # extract jaw landmarks
        x_right_jaw = landmarks.part(12).x
        y_right_jaw = landmarks.part(12).y
        cv2.circle(image_with_landmarks, (x_right_jaw, y_right_jaw), 3, (0, 0, 0), -1)
        x_left_jaw = landmarks.part(4).x
        y_left_jaw = landmarks.part(4).y
        cv2.circle(image_with_landmarks, (x_left_jaw, y_left_jaw), 3, (0, 0, 0), -1)
        cv2.line(
            image_with_landmarks, (x_left_jaw, y_left_jaw), (x_right_jaw, y_right_jaw), (0, 0, 0), 1
        )
        cv2.putText(
            image_with_landmarks,
            "D5",
            (int((x_left_jaw + x_right_jaw) / 2), int((y_left_jaw + y_right_jaw) / 2)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 0),
            1,
        )

    # display d4  jawline length
        cv2.line(image_with_landmarks, (x_right_jaw, y_right_jaw), (x_bottom, Y_bottom), (0, 0, 0), 1)
        cv2.putText(
            image_with_landmarks,
            "D4",
            (int((x_right_jaw + x_bottom) / 2), int((y_right_jaw + Y_bottom) / 2)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 0),
            1,
        )
        
        # extract distance between ears landmarks
        x_left_ear = landmarks.part(2).x
        y_left_ear = landmarks.part(2).y
        cv2.circle(image_with_landmarks, (x_left_ear, y_left_ear), 3, (0, 0, 0), -1)
        x_right_ear = landmarks.part(14).x
        y_right_ear = landmarks.part(14).y
        cv2.circle(image_with_landmarks, (x_right_ear, y_right_ear), 3, (0, 0, 0), -1)
        cv2.line(
            image_with_landmarks, (x_left_ear, y_left_ear), (x_right_ear, y_right_ear), (0, 0, 0), 1
        )
        cv2.putText(
            image_with_landmarks,
            "D1",
            (int((x_left_ear + x_right_ear) / 2), int((y_left_ear + y_right_ear) / 2)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
           (0, 0, 0),
            1,
        )
        
        # extract chin landmarks//menton
        x_chin_width_left = landmarks.part(6).x
        y_chin_width_left = landmarks.part(6).y
        x_chin_width = landmarks.part(10).x
        y_chin_width = landmarks.part(10).y
        cv2.circle(image_with_landmarks, (x_chin_width, y_chin_width), 3, (0, 0, 0), -1)
        cv2.circle(image_with_landmarks, (x_chin_width_left, y_chin_width_left), 3, (0, 0, 0), -1)
        cv2.line(
            image_with_landmarks,
            (x_chin_width, y_chin_width),
            (x_chin_width_left, y_chin_width_left),
            (0, 0, 0),
            1,
        )
        cv2.putText(
            image_with_landmarks,
            "D6",
            (
                int((x_chin_width + x_chin_width_left) / 2),
                int((y_chin_width + y_chin_width_left) / 2),
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 0),
            1,
        )

        # extract jaw landmarks//machoire
        x_jaw = landmarks.part(7).x
        y_jaw = landmarks.part(7).y
        cv2.circle(image_with_landmarks, (x_jaw, y_jaw), 3, (0, 0, 0), -1)
        x_jaw_left = landmarks.part(9).x
        y_jaw_left = landmarks.part(9).y
        cv2.circle(image_with_landmarks, (x_jaw_left, y_jaw_left), 3, (0, 0, 0), -1)
        cv2.line(image_with_landmarks, (x_jaw, y_jaw), (x_jaw_left, y_jaw_left), (0, 0, 0), 1)
        return image_with_landmarks






def crop_and_resize(image, target_w=224, target_h=224):
    """this function crop & resize images to target size by keeping aspect ratio"""
    if image.ndim == 2:
        img_h, img_w = image.shape  # for Grayscale will be   img_h, img_w = img.shape
    elif image.ndim == 3:
        (
            img_h,
            img_w,
            channels,
        ) = image.shape  # for RGB will be   img_h, img_w, channels = img.shape
    target_aspect_ratio = target_w / target_h
    input_aspect_ratio = img_w / img_h
    if input_aspect_ratio > target_aspect_ratio:
        resize_w = int(input_aspect_ratio * target_h)
        resize_h = target_h
        img = cv2.resize(image, (resize_w, resize_h))
        crop_left = int((resize_w - target_w) / 2)  ## crop left/right equally
        crop_right = crop_left + target_w
        new_img = img[:, crop_left:crop_right]
    if input_aspect_ratio < target_aspect_ratio:
        resize_w = target_w
        resize_h = int(target_w / input_aspect_ratio)
        img = cv2.resize(image, (resize_w, resize_h))
        crop_top = int(
            (resize_h - target_h) / 4
        )  ## crop the top by 1/4 and bottom by 3/4 -- can be changed
        crop_bottom = crop_top + target_h
        new_img = img[crop_top:crop_bottom, :]
    if input_aspect_ratio == target_aspect_ratio:
        new_img = cv2.resize(image, (target_w, target_h))
    return new_img


detector = MTCNN()


def extract_face(img, target_size=(224, 224)):
    # 1. detect faces in an image
    results = detector.detect_faces(img)
    if (
        results == []
    ):  # if face is not detected, call function to crop & resize by keeping aspect ratio
        new_face = crop_and_resize(img, target_w=224, target_h=224)
    else:
        x1, y1, width, height = results[0]["box"]
        x2, y2 = x1 + width, y1 + height
        face = img[
            y1:y2, x1:x2
        ]  # this is the face image from the bounding box before expanding bbox
        # 2. expand the top & bottom of bounding box by 10 pixels to ensure it captures the whole face
        adj_h = 10
        # assign value of new y1
        if y1 - adj_h < 10:
            new_y1 = 0
        else:
            new_y1 = y1 - adj_h
        # assign value of new y2
        if y1 + height + adj_h < img.shape[0]:
            new_y2 = y1 + height + adj_h
        else:
            new_y2 = img.shape[0]
        new_height = new_y2 - new_y1
        # 3. crop the image to a square image by setting the width = new_height and expand the box to new width
        adj_w = int((new_height - width) / 2)
        # assign value of new x1
        if x1 - adj_w < 0:
            new_x1 = 0
        else:
            new_x1 = x1 - adj_w
        # assign value of new x2
        if x2 + adj_w > img.shape[1]:
            new_x2 = img.shape[1]
        else:
            new_x2 = x2 + adj_w
        new_face = img[
            new_y1:new_y2, new_x1:new_x2
        ]  # face-cropped square image based on original resolution
    # 4. resize image to the target pixel size
    sqr_img = cv2.resize(new_face, target_size)
    return sqr_img

# load the model
@st.cache_resource  # this code i want to run it ones
def load_my_model():
    face_shape_model = tf.keras.models.load_model("vgg16-face-1")
    return face_shape_model

y_label_dict = {0: "cœur", 1: "oblong", 2: "ovale", 3: "rond", 4: "carré"}

def predict_face_shape(image_array):
    # first extract the face using bounding box
    face_img = extract_face(
        image_array
    )  # call function to extract face with bounding box
    new_img = cv2.cvtColor(
        face_img, cv2.COLOR_BGR2RGB
    )  # convert to RGB -- use this for display
    # convert the image for modelling
    test_img = np.array(new_img, dtype=float)
    test_img = test_img / 255
    test_img = np.array(test_img).reshape(1, 224, 224, 3)
    # make predictions
    face_shape_model = load_my_model()
    pred = face_shape_model.predict(test_img)
    label = np.argmax(pred, axis=1)
    shape = y_label_dict[label[0]]
    pred = np.max(pred)
    pred = np.around(pred * 100, 2)
    return shape, pred, new_img






camera_photo = st.camera_input("Trouvez la forme de votre visage", label_visibility="hidden")

if camera_photo is not None:
    bytes_data = camera_photo.getvalue()
    
    
    cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)

    st.session_state.cv2_img=cv2_img
  
    shape, pred, new_img = predict_face_shape(cv2_img)
    img1  = analyze_facial_features(new_img)
    col1,col2,col3=st.columns(3)
    col2.image(img1)
    
    
    st.write(
    f"<h2>La forme de votre visage est : {shape} avec une probabilité de : {pred} % </h2>",
    unsafe_allow_html=True,)


    btn = st.markdown("""
            <style>
            div.stButton > button:first-child {
                background-color: #8fdeed;
            color:#3e07b4;
                font-size:25px;
                height:3em;
                width:15em;
                border-radius:0.75rem;
                
            }
            div.stButton > button:hover {
            background-color: #06f3ff;
            color:#ffffff;
            border: 2px solid white;                    
    }       </style>""", unsafe_allow_html=True)



    col1,col2,col3,col4,col5,col6=st.columns(6)
    btn_suivat=col6.button("suivant")   
    if btn_suivat:
        switch_page("Régles") 
    