import streamlit as st
from streamlit_extras.switch_page_button import switch_page
st.set_page_config(
        page_title="Conseil",
        page_icon="💬",
    )
st.title("     Quelques conseils avant de commencer     ")
st.markdown("------")
st.header("1 Placer toi dans un endroit bien éclairé")
st.header("2 Regarder droit vers la caméra")
st.header("3 Garder le visage fixe et ne souriez pas")
st.header("4 Enlever vos lunettes si vous en portez")
st.header("5 éloingner vos cheveux de votre visage")
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

col1, col2, col3,col4,col5=st.columns(5)
btn_start=col2.button("Commencer le Test")
if btn_start:
        switch_page("Detection_De_Visage")


