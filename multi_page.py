import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

import streamlit as st
from streamlit_option_menu import option_menu
import app,live_demo

# Setting the header for the page
st.title("Fall Detection System Protype")

#add navigation option menu
with st.sidebar:
    selected_page  = option_menu(
        menu_title= "Select Page",
        options = ["Live Video Feed", "Images and Video"],
    )

if selected_page == "Images and Video":
    app.main()
    
elif selected_page == "Live Video Feed":
    live_demo.main()
