import streamlit as st
import json
from streamlit_lottie import st_lottie
from streamlit_extras.switch_page_button import switch_page


def main():
    # page configuration
    st.set_page_config(
        page_title="KatYos Virtual Assistant",
        page_icon="💬",
    )

    # to remove stremlit app in the footer and humburger
    hide_st_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
    st.markdown(hide_st_style, unsafe_allow_html=True)
    
    

    st.markdown(
        "<h1 style='text-align:center;'> visagisme Virtuel  </h1>",
        unsafe_allow_html=True,
    )
    st.markdown("------")
    # to design the rectangle shadow
    st.write(
        """<div style='
background-color: #3e07b4;
padding: 20px;border-radius: 
5px;box-shadow: 0 2px 8px rgba(0, 0, 0, 0.3);
'><h2 style='text-align: center; color: white; font-size: 35px;'>On vous aide à trouver la monture faite pour vous!</h2>
</div>
""",
        unsafe_allow_html=True,
    )

    with open("help.json") as source:
        animation = json.load(source)  # for the annimation help

    col1, col2, col3 = st.columns(3)
    with col3:
        st_lottie(animation, width=300, height=300)
    with col1:
                
        btn = st.markdown("""
                    <style>
                    div.stButton > button:first-child {
                        background-color: #8fdeed;
                        color:#3e07b4;
                        font-size:25px;
                        height:3em;
                        margin-top:90px;
                        
                    position:relative;rigth:50%;
                        width:15em;
                        border-radius:0.75rem;
                        
                    }
                    div.stButton > button:hover {
                    background-color: #06f3ff;
                    color:#ffffff;
                    border: 2px solid white;
                    

                    
            }       </style>""", unsafe_allow_html=True)
        btn = st.button("Avant de Commener")
        if btn:
            switch_page("Conseil")
        
            
    
if __name__ == "__main__":
    main()
