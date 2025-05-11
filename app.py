import streamlit as st
from fastai.vision.all import *
import pathlib
import plotly.express as px
temp=pathlib.PosixPath
pathlib.PosixPath =pathlib.WindowsPath


st.title('Modul to classify images')

file=st.file_uploader("upload image",type=['jpg','png','jpeg','gif'])
if file:
   st.image(file)
   img=PILImage.create(file)

   model=load_learner('classification_model.pkl')

   pred,pred_id,probs=model.predict(img)
   st.success(f'Prediction:{pred}')
   st.info(f'Probability:{probs[pred_id]*100:.1f}%')

   fig=px.bar(x=probs*100,y=model.dls.vocab)
   st.plotly_chart(fig)
