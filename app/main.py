# import dependecies
import streamlit as st
import pickle
import pandas as pd
import plotly.graph_objects as go
import numpy as np

# data cleaning function
def getClean():
    # uses pandas to get data into variable
    data = pd.read_csv("data/data.csv")
    
    # drop unnamed data column and id column
    data = data.drop(["Unnamed: 32", "id"], axis=1)

    # maps the M (malignant) disgnosis to 1 and the B(begnin) diagnosis to 0
    data['diagnosis'] = data['diagnosis'].map({'M':1, 'B':0})

    # returns data
    return data


# function to add a sidebar and adjust data
def addSidebar():
    st.sidebar.header("Cell Measurements:")

    # import clean data from csv
    data = getClean()

    # add labels for sliders 
    slider_labels = [
        ("Radius (mean)", "radius_mean"),
        ("Texture (mean)", "texture_mean"),
        ("Perimeter (mean)", "perimeter_mean"),
        ("Area (mean)", "area_mean"),
        ("Smoothness (mean)", "smoothness_mean"),
        ("Compactness (mean)", "compactness_mean"),
        ("Concavity (mean)", "concavity_mean"),
        ("Concave points (mean)", "concave points_mean"),
        ("Symmetry (mean)", "symmetry_mean"),
        ("Fractal dimension (mean)", "fractal_dimension_mean"),
        ("Radius (se)", "radius_se"),
        ("Texture (se)", "texture_se"),
        ("Perimeter (se)", "perimeter_se"),
        ("Area (se)", "area_se"),
        ("Smoothness (se)", "smoothness_se"),
        ("Compactness (se)", "compactness_se"),
        ("Concavity (se)", "concavity_se"),
        ("Concave points (se)", "concave points_se"),
        ("Symmetry (se)", "symmetry_se"),
        ("Fractal dimension (se)", "fractal_dimension_se"),
        ("Radius (worst)", "radius_worst"),
        ("Texture (worst)", "texture_worst"),
        ("Perimeter (worst)", "perimeter_worst"),
        ("Area (worst)", "area_worst"),
        ("Smoothness (worst)", "smoothness_worst"),
        ("Compactness (worst)", "compactness_worst"),
        ("Concavity (worst)", "concavity_worst"),
        ("Concave points (worst)", "concave points_worst"),
        ("Symmetry (worst)", "symmetry_worst"),
        ("Fractal dimension (worst)", "fractal_dimension_worst"),
    ]

    # input dict to return data
    inputDict = {}

    # for loop to add each label and slider to the sidebar and add input to input dict
    for label, key in slider_labels:
        inputDict[key] = st.sidebar.slider(
            label,
            min_value=float(0),
            max_value=float(data[key].max()),
            value=float(data[key].mean())
        )

    # return input dict
    return inputDict


# function to get scaled values
def getScaled(inputDict):

    # import clean data from csv
    data = getClean()

    # drop diagnosis
    X = data.drop(['diagnosis'], axis=1)

    # create scaled dict
    scaledDict = {}

    # for loop to get scaled values
    for key, value in inputDict.items():
        max_val = X[key].max()
        min_val = X[key].min()
        scaled_value = (value - min_val) / (max_val - min_val)
        scaledDict[key] = scaled_value

    # return scaled values
    return scaledDict

# generate radar chart (from plotly docs)
def getRadar(input_data):
    input_data = getScaled(input_data)
  
    categories = ['Radius', 'Texture', 'Perimeter', 'Area', 
                    'Smoothness', 'Compactness', 
                    'Concavity', 'Concave Points',
                    'Symmetry', 'Fractal Dimension']

    fig = go.Figure()

    fig.add_trace(go.Scatterpolar(
            r=[
            input_data['radius_mean'], input_data['texture_mean'], input_data['perimeter_mean'],
            input_data['area_mean'], input_data['smoothness_mean'], input_data['compactness_mean'],
            input_data['concavity_mean'], input_data['concave points_mean'], input_data['symmetry_mean'],
            input_data['fractal_dimension_mean']
            ],
            theta=categories,
            fill='toself',
            name='Mean Value'
    ))
    fig.add_trace(go.Scatterpolar(
            r=[
            input_data['radius_se'], input_data['texture_se'], input_data['perimeter_se'], input_data['area_se'],
            input_data['smoothness_se'], input_data['compactness_se'], input_data['concavity_se'],
            input_data['concave points_se'], input_data['symmetry_se'],input_data['fractal_dimension_se']
            ],
            theta=categories,
            fill='toself',
            name='Standard Error'
    ))
    fig.add_trace(go.Scatterpolar(
            r=[
            input_data['radius_worst'], input_data['texture_worst'], input_data['perimeter_worst'],
            input_data['area_worst'], input_data['smoothness_worst'], input_data['compactness_worst'],
            input_data['concavity_worst'], input_data['concave points_worst'], input_data['symmetry_worst'],
            input_data['fractal_dimension_worst']
            ],
            theta=categories,
            fill='toself',
            name='Worst Value'
    ))

    fig.update_layout(
        polar=dict(
        radialaxis=dict(
            visible=True,
            range=[0, 1]
        )),
        showlegend=True
    )
    
    return fig

# function to get predictions
def getPred(input_data):
    model = pickle.load(open("model/model.pkl", "rb"))
    scaler = pickle.load(open("model/scaler.pkl", "rb"))
    
    input_array = np.array(list(input_data.values())).reshape(1, -1)
    
    input_array_scaled = scaler.transform(input_array)
    
    prediction = model.predict(input_array_scaled)
    
    st.subheader("Cell cluster prediction")
    st.write("The cell cluster is:")
    
    if prediction[0] == 0:
        st.write("<span class='diagnosis benign'>Benign</span>", unsafe_allow_html=True)
    else:
        st.write("<span class='diagnosis malicious'>Malignant</span>", unsafe_allow_html=True)
        
    
    st.write("Probability of being benign (interpret as a percentage): ", model.predict_proba(input_array_scaled)[0][0])
    st.write("Probability of being malignant (interpret as a percentage): ", model.predict_proba(input_array_scaled)[0][1])
    
    st.write("This project is for EDUCATIONAL USE ONLY and should NOT be used for real cancer diagosis without the supervision of a doctor.")

# main function to run app
def main():

    # setting page configuration
    st.set_page_config(
        page_title="Malignancy Predictor For Breast Cancer", 
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # adds sidebar and stores value of sidebar
    inputSidebar = addSidebar()

    # new container
    with st.container():
        # title and description of the app
        st.title("Malignancy Predictor For Breast Cancer")
        st.write("This app uses machine learning to predict if a mass is benign or malignant based on breast cancer tissue measurements from a cytosis lab. These mesurments can be adjusted using the sidebar to the right and the model has an accuracy of 97%. This model uses logistic regression and was trained using the Breast Cancer Wisconsin (Diagnostic) Data Set from Kaggle provided by UCI MACHINE LEARNING. This project is for EDUCATIONAL USE ONLY and should NOT be used for real cancer diagosis without the supervision of a doctor.")

        # columns for cell measurement visualization and prediction probabilites
        c1, c2 = st.columns([4,1])

        # write the columns to the app for radar chart in c1 and predictions in c2
        with c1:
            radar_chart = getRadar(inputSidebar)
            st.plotly_chart(radar_chart)
        with c2:
            getPred(inputSidebar)

if __name__ == "__main__":
    main()