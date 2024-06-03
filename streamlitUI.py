import numpy as np
import cv2
from ultralytics import YOLO
from PIL import Image
import streamlit as st
from streamlit_option_menu import option_menu

model = YOLO('/Users/velmurugan/Desktop/@/python_works/computer vision/train2/weights/best.pt')

def image_detection(image,model,threshold):
    image = Image.open(image)
    image = np.array(image.convert('RGB'))
    
    results = model(image)[0]

    for result in results.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = result

        # Check if the detection confidence score is above the threshold
        if score > threshold:
            if int(class_id) == 0:
                # Draw a rectangle around the detected object
                cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 4)

                # Put the class name above the bounding box
                cv2.putText(image, results.names[int(class_id)].upper(), (int(x1), int(y1 - 10)),
                            cv2.FONT_HERSHEY_PLAIN, 2.2, (0, 255, 0), 3, cv2.LINE_AA)
            
            else:
                cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0,255), 4)

                # Put the class name above the bounding box
                cv2.putText(image, results.names[int(class_id)].upper(), (int(x1), int(y1 - 10)),
                            cv2.FONT_HERSHEY_PLAIN, 2, (255,0,0), 3, cv2.LINE_AA)
    return image

def video_detection(video_path,model,threshold):
    cap = cv2.VideoCapture(video_path)
    ret,frame = cap.read()
    H,W,C = frame.shape

    outpath = video_path.split('.')[0]+'out.mp4'
    out = cv2.VideoWriter(outpath, cv2.VideoWriter_fourcc(*'MP4V'), int(cap.get(cv2.CAP_PROP_FPS)), (W,H))

    while ret:
        results = model(frame)[0]

        for result in results.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = result

            # Check if the detection confidence score is above the threshold
            if score > threshold:
                if int(class_id) == 0:
                    # Draw a rectangle around the detected object
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 4)

                    # Put the class name above the bounding box
                    cv2.putText(frame, results.names[int(class_id)].upper(), (int(x1), int(y1 - 10)),
                                cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 3, cv2.LINE_AA)
            else:
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0,255), 4)

                # Put the class name above the bounding box
                cv2.putText(frame, results.names[int(class_id)].upper(), (int(x1), int(y1 - 10)),
                            cv2.FONT_HERSHEY_PLAIN, 3, (255,0,0), 3, cv2.LINE_AA)
        out.write(frame)
        ret,frame = cap.read()
    
    cap.release()
    out.release()

    return outpath

#streamlit part

def setting_bg():
    st.markdown(f""" <style>.stApp {{
                background: linear-gradient(to bottom, #A9A9A9, #808080, #404040, #000000);
            }}
           </style>""",
        unsafe_allow_html=True)

setting_bg()

with st.sidebar:
    options = option_menu("Main Menu", ["Home", 'Image Detection','Video Detection(Beta)'], 
        icons=['house', 'gear'], menu_icon="cast", default_index=1,
        styles={
            "container": {"padding": "0!important", "background-color": "#fafafa"},
            "icon": {"color": "orange", "font-size": "25px"}, 
            "nav-link": {
                "font-size": "16px",
                "text-align": "left",
                "margin": "0px",
                "--hover-color": "#eee",
            },
            "nav-link-selected": {"background-color": "green"},
        })
    
if options == 'Home':
    st.title('Object Detection')

    col1,col2 = st.columns([1,0.5])

    with col2:
        st.image('/Users/velmurugan/Desktop/@/python_works/computer vision/spiderman.gif')
        


elif options == 'Image Detection':
    col1,col2 = st.columns([1,0.3])
    with col2:
        st.image('/Users/velmurugan/Desktop/@/python_works/computer vision/images/train/s86.jpg')

    with col1:
        label = """
        <div style='color: yellow; font-size: 20px;'>
            select threshold value
        </div>
        """

        # Use st.markdown to display the label
        st.markdown(label, unsafe_allow_html=True)

        # Display the slider
        threshold = st.slider("", 0.0, 1.0, 0.5)
    
    label = """
        <div style='color: yellow; font-size: 20px;'>
            Choose an image
        </div>
        """
    
    st.markdown(label, unsafe_allow_html=True)

    uploaded_file = st.file_uploader("", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        if uploaded_file.type.startswith('image'):
            st.image(uploaded_file, caption='Uploaded Image.', use_column_width=True)
            detect_button = st.button('Detect Spiderman')
            if detect_button:
                processed_image = image_detection(uploaded_file, model, threshold)
                st.image(processed_image, caption='Processed Image.', use_column_width=True)


elif options == 'Video Detection(Beta)':
    col1,col2 = st.columns([1,0.3])
    with col2:
        st.image('/Users/velmurugan/Desktop/@/python_works/computer vision/no-way-home-spider-man-no-way-home.gif')

    with col1:
        label = """
        <div style='color: yellow; font-size: 20px;'>
            select threshold value
        </div>
        """

        # Use st.markdown to display the label
        st.markdown(label, unsafe_allow_html=True)

        # Display the slider
        threshold = st.slider("", 0.0, 1.0, 0.5)
    
    label = """
        <div style='color: yellow; font-size: 20px;'>
            Choose Video
        </div>
        """
    
    st.markdown(label, unsafe_allow_html=True)

    uploaded_file = st.file_uploader("", type=["mp4"])  

    if uploaded_file is not None:
        if uploaded_file.type.startswith('video'):
            video_bytes = uploaded_file.read()
            st.video(video_bytes)

            detect_button = st.button('Detect Spiderman')      

            if detect_button:
                with open('temporary.mp4','wb') as f:
                    f.write(video_bytes)

                processsed_video = video_detection('temporary.mp4',model,threshold)
                st.video(processsed_video)
