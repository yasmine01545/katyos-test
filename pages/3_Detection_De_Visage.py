import streamlit as st
import json
from streamlit_lottie import st_lottie
from streamlit_extras.switch_page_button import switch_page
st.set_page_config(
        page_title="Detection",
        page_icon="💬",
    )

st.header("téléchargez une photo de face ou prenez une photo afin que notre IA puisse déterminer la forme de votre visage")
st.markdown("------")
col1, col2, col3 =st.columns(3)


with open("face-recognition.json") as source:
    face = json.load(source)  # for the annimation help


    st_lottie(face, width=500, height=500)
col1,col2,col3=st.columns(3)
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
button_photo=col1.button("photo")
button_video=col3.button("caméra")
if button_photo:
    switch_page("Photo")
if button_video:
    switch_page("Caméra")