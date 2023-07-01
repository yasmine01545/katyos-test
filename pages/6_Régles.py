
import streamlit as st
import numpy as np
import cv2
from mtcnn.mtcnn import MTCNN
import tensorflow as tf
import joblib
import pandas as pd

from streamlit_extras.switch_page_button import switch_page


st.set_page_config(
        page_title="Régles",
        page_icon="💬",
    )



# Load the trained model (monture)
loaded_model_monture_gb = joblib.load("model_monture_74/monture_gb_74_model.pkl")
# Load the label encoder dictionary
le_dict = joblib.load("model_monture_74/le_dict.pkl")
loaded_model_materiaux=joblib.load("model_monture_74/materiaux_model_99.pkl")


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




def make_prediction_monture(genre, select_type,shape, select_style, select_uti):
    input_data = {
        'genre': [genre],
        'type.type': [select_type],
        'visage.visage': [shape],
        'style': [select_style],
        'utilisation': [select_uti]
    }
    input_df = pd.DataFrame(input_data)
   

    # Encode input features
    for column in input_df.columns:
        le = le_dict[column]
        input_df[column] = le.transform(input_df[column])
    
    # Make prediction
    prediction_monture = loaded_model_monture_gb.predict(input_df)
    prediction_materaiux=loaded_model_materiaux.predict(input_df)
    
    return prediction_monture[0],prediction_materaiux[0]


if  'image_array' in st.session_state:
    
    image_array = st.session_state.image_array#appler variable
    shape, _, _ = predict_face_shape(image_array)
    st.write(f"<h4>la forme de votre  visage est : <span style='color: #3e07b4;'>{shape}</span> </h4> " , unsafe_allow_html=True,)




elif 'cv2_img' in st.session_state:
    cv2_img = st.session_state.cv2_img
    shape, _,_= predict_face_shape(cv2_img)
    st.write(f"<h4>la forme de votre  visage est : <span style='color: #3e07b4;'>{shape}</span> </h4> " , unsafe_allow_html=True,)

else:
    st.write(f"<h4>Veuillez SVP Télecharger une photo avant </h4> " , unsafe_allow_html=True,)

st.session_state.clear()

st.title("Veuillez sélectionner quelques options supplémentaires")
st.markdown('------')



genre = st.selectbox('Genre', ['homme', 'femme'])
select_type = st.selectbox(
    "Quel type de lunettes recherchez-vous ?", options=("de soleil", "de vue")
)

select_style = st.selectbox(
    "Quel style de lunettes recherchez-vous ?",
    options=(
        "élégant",
       
        "sportif",
        "unique",
        "jeune",
        "décontractée",
        "sérieux",
        "dynamique",
        "chic",
        
        "classique",
        " branché ",
        "rétro",
        "audacieux",
         "sophistiqué",
        "intellectuel",
        "vinatge",
        "rock",
        "tendance",
         "coquette",
        "mode",
        " branché ",
        "sympathique",
        "intemporel",
        "audacieux",
        "féminine",
        "estival",
        "coquine",
       
    ),
)
select_uti = st.selectbox(
    "Pour quel usage ?",
    options=(
         "usage quotidien",
        "activites sportif",
        "plage",
        
        "activité nautique",
     
        "activités professionnelles",
        "conduite",
        "sortie en ville",
        "activité en plein air",
        "occasion spécial",
    ),

)



# Get the image_array from session_state


   
col1, col2, col3,col4,col5=st.columns(5)
pre=col2.button('Suivant')

if pre : 
    st.session_state.clear()
    
    try:
        prediction_monture,prediction_materaiux=make_prediction_monture(genre, select_type, shape, select_style, select_uti)
       # st.write(f"<h4>la forme de votre  visage est : <span style='color: #3e07b4;'>{shape}</span> </h4> " , unsafe_allow_html=True,)

        st.write(f"<h4>La monture faite pour vous est  :<span style='color: #3e07b4;'>{prediction_monture}</span> </h4>",unsafe_allow_html=True,)


        message = f"<h4> Pour votre usage : {select_uti}, il est recommandé d'utiliser des montures qui sont fabriquées avec l'un de ces matériaux : <span style='color: #3e07b4;'>{prediction_materaiux}</span></h4>"
        st.markdown(message, unsafe_allow_html=True)
        #st.write(f"<h4>Pour votre usage : { select_uti } ,il est recommandé d'utiliser des montures qui sont fabriqués avec l'un de ces materiaux :</h4> " , prediction_materaiux,unsafe_allow_html=True,)
        st.write("")

        

        
        

        encoded_select_type = ""
        valid_formes_soleil_katyos = ["Aviator", "Carrées", "Masque", "Oeil de chat", "Ovales", "Papillon", "Pilote", "Rectangulaires", "Rondes"]
        valid_formes_de_vue_katyos = ["Carrées", "Oeil de chat", "Ovales", "Papillon", "Pilote", "Rectangulaires", "Rondes"]
        encoded_select_type = select_type.replace(" ", "-")
        encoded_prediction_monture=prediction_monture.replace(" ", "+")
        

        if select_type == "de soleil" and prediction_monture in valid_formes_soleil_katyos:
            
            url = f"https://katyos.com/6-lunettes-{encoded_select_type}-?q=Formes-{encoded_prediction_monture}"
        


            message = f'<h3>Vous pouvez trouver votre monture dans notre mall :" </h3> <a href="{url}" style="display: inline-block; padding: 10px; background-color: #3e07b4; color: white; text-decoration: none;">Cliquez ici pour accéder au site</a>'

            st.write(message, unsafe_allow_html=True)

        
        elif select_type == "de vue" and prediction_monture in valid_formes_de_vue_katyos:
            
            
            url = f"https://katyos.com/3-lunettes-{encoded_select_type}-?q=Formes-{encoded_prediction_monture}"
                            
            st.subheader("Vous pouvez trouver votre monture dans notre mall :")
            message = f"[Cliquez ici pour accéder au site]({url})"
            st.markdown(message, unsafe_allow_html=True)
        else:
            

            error_message = "Cette monture n'est pas encore disponible dans notre site ,Veuillez visiter notre site ultérieurement."
            st.subheader(error_message)

          
    except NameError:
        
    
        st.write("Please upload an image first")
   



    

btn = st.markdown("""
            <style>
            div.stButton > button:first-child {
                background-color: #8fdeed;
            color:#3e07b4;
                font-size:25px;
                height:3em;
                width:15em;
                margin-top:40px;
                border-radius:0.75rem;
            }
            div.stButton > button:hover {
            background-color: #06f3ff;
            color:#ffffff;
            border: 2px solid white;                    
    }       </style>""", unsafe_allow_html=True)
