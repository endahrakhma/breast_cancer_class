import streamlit as st
import pandas as pd
import numpy as np
import joblib
import base64
import pickle

from keras.models import load_model

import warnings
warnings.filterwarnings("ignore")

def set_bg_hack(main_bg):

    main_bg_ext = "png"
        
    st.markdown(
         f"""
         <style>
         .stApp {{
             background: url(data:image/{main_bg_ext};base64,{base64.b64encode(open(main_bg, "rb").read()).decode()});
             background-position: center;
             background-size: 720px 520px;
             background-repeat: no-repeat
         }}
         </style>
         """,
         unsafe_allow_html=True
     )
set_bg_hack('bg_breastcancer.png')

st.markdown("<h2 style='text-align: center;'>Breast Cancer Classification</h2>", unsafe_allow_html=True)
st.markdown('---'*10)

model_final = joblib.load('sklearn_pipeline.pkl')
model_final.named_steps['modeling'].model = load_model('model_keras.h5')

pilihan = st.selectbox('Apa yang ingin Anda lakukan?',['Prediksi dari file csv','Input Manual'])

if pilihan == 'Prediksi dari file csv':
    # Mengupload file
    upload_file = st.file_uploader('Pilih file csv', type='csv')
    if upload_file is not None:
        dataku1 = pd.read_csv(upload_file)
        dataku1.rename(columns={"concave points_mean":"concave_points_mean","concave points_se":"concave_points_se","concave points_worst":"concave_points_worst"}, inplace=True)
        dataku = dataku1.copy()
        dataku.drop(['Unnamed: 32','id'],axis=1,inplace=True)
        dataku.drop(columns='diagnosis',axis=0,inplace=True)
        st.write(dataku)
        st.success('File berhasil diupload')
        hasil = model_final.predict(dataku)
        #st.write('Prediksi',hasil)
        # Keputusan
        for i in range(len(hasil)):
            if hasil[i] == 1:
                st.write('Klasifikasi Breast Cancer',dataku1['id'][i],'= diprediksi Malignant')
            else:
                st.write('Klasifikasi Breast Cancer',dataku1['id'][i],'= diprediksi Benign')
    else:
        st.error('File yang diupload kosong, silakan pilih file yang valid')
        #st.markdown('File yang diupload kosong, silakan pilih file yang valid')
else:

    #1
    with st.container():
        col1, col2 = st.columns(2)
        with col1:
            radius_mean = st.number_input('radius_mean', value=17)
        with col2:
            texture_mean = st.number_input('texture_mean', value=18)
    
    #2
    with st.container():
        col1, col2 = st.columns(2)
        with col1:
            perimeter_mean = st.number_input('perimeter_mean', value=100)
        with col2:
            area_mean = st.number_input('area_mean', value=1000)
    
    #3
    with st.container():
        col1, col2 = st.columns(2)
        with col1:
            smoothness_mean = st.number_input('smoothness_mean', value=0.15)
        with col2:
            compactness_mean = st.number_input('compactness_mean', value=0.30)
    
    #4
    with st.container():
        col1, col2 = st.columns(2)
        with col1:
            concavity_mean = st.number_input('concavity_mean', value=0.1)
        with col2:
            concave_points_mean = st.number_input('concave_points_mean', value=0.05)
    
    #5
    with st.container():
        col1, col2 = st.columns(2)
        with col1:
            symmetry_mean = st.number_input('symmetry_mean', value=0.2)
        with col2:
            fractal_dimension_mean = st.number_input('fractal_dimension_mean', value=0.06)
    
    #6
    with st.container():
        col1, col2 = st.columns(2)
        with col1:
            radius_se = st.number_input('radius_se', value=0.5)
        with col2:
            texture_se = st.number_input('texture_se', value=0.8)
    
    #7
    with st.container():
        col1, col2 = st.columns(2)
        with col1:
            perimeter_se = st.number_input('perimeter_se', value=6)
        with col2:
            area_se = st.number_input('area_se', value=100)
    
    #8
    with st.container():
        col1, col2 = st.columns(2)
        with col1:
            smoothness_se = st.number_input('smoothness_se', value=0.008)
        with col2:
            compactness_se = st.number_input('compactness_se', value=0.05)
    
    #9
    with st.container():
        col1, col2 = st.columns(2)
        with col1:
            concavity_se = st.number_input('concavity_se', value=0.07)
        with col2:
            concave_points_se = st.number_input('concave_points_se', value=0.02)
    
    #10
    with st.container():
        col1, col2 = st.columns(2)
        with col1:
            symmetry_se = st.number_input('symmetry_se', value=0.07)
        with col2:
            fractal_dimension_se = st.number_input('fractal_dimension_se', value=0.01)
    
    #11
    with st.container():
        col1, col2 = st.columns(2)
        with col1:
            radius_worst = st.number_input('radius_worst', value=20)
        with col2:
            texture_worst = st.number_input('texture_worst', value=27)
    
    #12
    with st.container():
        col1, col2 = st.columns(2)
        with col1:
            perimeter_worst = st.number_input('perimeter_worst', value=160)
        with col2:
            area_worst = st.number_input('area_worst', value=1700)
    
    #13
    with st.container():
        col1, col2 = st.columns(2)
        with col1:
            smoothness_worst = st.number_input('smoothness_worst', value=0.11)
        with col2:
            compactness_worst = st.number_input('compactness_worst', value=0.9)
    
    #14
    with st.container():
        col1, col2 = st.columns(2)
        with col1:
            concavity_worst = st.number_input('concavity_worst', value=0.8)
        with col2:
            concave_points_worst = st.number_input('concave_points_worst', value=0.3)
    
    #15
    with st.container():
        col1, col2 = st.columns(2)
        with col1:
            symmetry_worst = st.number_input('symmetry_worst', value=0.8)
        with col2:
            fractal_dimension_worst = st.number_input('fractal_dimension_worst', value=0.2)
        
    # Inference
    data = {
            'radius_mean': radius_mean,
            'texture_mean': texture_mean,
            'perimeter_mean': perimeter_mean,
            'area_mean': area_mean,
            'smoothness_mean': smoothness_mean,
            'compactness_mean': compactness_mean,
            'concavity_mean': concavity_mean,
            'concave_points_mean': concave_points_mean,
            'symmetry_mean': symmetry_mean,
            'fractal_dimension_mean': fractal_dimension_mean,
            'radius_se': radius_se,
            'texture_se': texture_se,
            'perimeter_se': perimeter_se,
            'area_se': area_se,
            'smoothness_se': smoothness_se,
            'compactness_se': compactness_se,
            'concavity_se': concavity_se,
            'concave_points_se': concave_points_se,
            'symmetry_se': symmetry_se,
            'fractal_dimension_se': fractal_dimension_se,
            'radius_worst': radius_worst,
            'texture_worst': texture_worst,
            'perimeter_worst': perimeter_worst,
            'area_worst': area_worst,
            'smoothness_worst': smoothness_worst,
            'compactness_worst': compactness_worst,
            'concavity_worst': concavity_worst,
            'concave_points_worst': concave_points_worst,
            'symmetry_worst': symmetry_worst,
            'fractal_dimension_worst': fractal_dimension_worst          
            }
    
    
    # Tabel data
    kolom = list(data.keys())
    df = pd.DataFrame([data.values()], columns=kolom)
    
    mystyle = '''
    <style>
        p {
            text-align: justify;
        }
    </style>
    '''
    
    # Memunculkan hasil di Web 
    st.write('***'*10)
    if st.button('Breast Cancer Classification'):
        prediksi = model_final.predict(df)
        if (prediksi[0] == 1):
            st.warning('Malignant')
        else:
            st.success('Benign')
    
