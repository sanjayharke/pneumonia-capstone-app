
"""Streamlit frontend; the backend owns model inference."""
from pathlib import Path
import io,os,base64
import requests
import streamlit as st
from pneumonia_core import inference_image

st.set_page_config(page_title='Chest X-ray capstone',layout='centered')
st.title('Chest X-ray classification: academic prototype')
st.caption('Predicts the dataset lung-opacity target. Not a diagnosis or a clinical decision tool.')
BACKEND_URL=os.environ.get('PNEUMONIA_BACKEND_URL','http://127.0.0.1:5000').rstrip('/')
upload=st.file_uploader('Upload a chest X-ray',type=['dcm','png','jpg','jpeg'])
if upload is not None:
    if upload.size>20*1024*1024:
        st.error('Use an image smaller than 20 MB.');st.stop()
    extension=Path(upload.name).suffix.lower()
    if extension!='.dcm':st.warning('PNG/JPEG is outside the DICOM validation pipeline; the score is exploratory.')
    try:
        raw=upload.getvalue()
        image=inference_image(io.BytesIO(raw),extension)
        st.image(image,caption='Preprocessed image used for inference',width=350)
        with st.spinner('Evaluating image…'):
            response=requests.post(BACKEND_URL+'/predict',files={'file':('image'+extension,raw,'application/octet-stream')},timeout=120)
        response.raise_for_status();result=response.json()
        if result['run_mode']!='full':st.warning('SMOKE-TEST MODEL: not a final project result.')
        st.subheader('Lung opacity target positive' if result['predicted_target'] else 'Lung opacity target negative')
        st.metric('Estimated positive-class probability',f"{100*result['positive_probability']:.2f}%")
        st.caption(f"Threshold: {result['threshold']:.3f}. Negative includes normal and other abnormalities.")
        if st.checkbox('Show experimental attribution'):
            with st.spinner('Computing positive-score Grad-CAM...'):
                xr=requests.post(BACKEND_URL+'/explain',files={'file':('image'+extension,raw,'application/octet-stream')},timeout=120)
            xr.raise_for_status();explanation=xr.json()
            st.image(base64.b64decode(explanation['png_base64']),caption=explanation['explanation'],width=350)
            if explanation['blank_map']:st.info('Rectified attribution is blank; this does not establish absence of disease.')

    except requests.RequestException:
        st.error('Inference service unavailable or image rejected. Confirm the application is running and try a valid image.')
    except Exception:
        st.error('Unable to read this image. Check format and decoder support.')
