import streamlit as st
from ultralytics import YOLO
from PIL import Image
#load a model
model = YOLO('yolov8n.pt')
with st.sidebar:
    thresh = st.slider('Threshold', max_value=0.99, min_value=0.1, value= 0.5)
with st.expander("about this app"):
    st.text("this app was created in class for fun")
img_file = st.file_uploader("Burv upload your image", type= ["png", "jpg", "jpeg"], help="this should only be png")
                                                         
if img_file:
    col1, col2 = st.columns(2)
    col1.image(img_file, caption = "original Imaage", use_column_width = True)

    # st.image(img_file, caption='Upload Successful!')
    Image.open(img_file).save(img_file.name)
    results = model(img_file.name, stream=False)
    results[0].save(filename='result.png')
    st.image('result.png', caption='Here are your predictions!')
    col2.image("results.png", caption = "Results from YOLO", use_column_width = True)