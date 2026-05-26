#import Libraries
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression

iris = load_iris()
x = iris.data
y = iris.target

model = LogisticRegression(max_iter=200)
model.fit(X, y)

st.title("Iris Flower Classification")
st.write("Enter the flower measurements:")

sepal_length = st.text_input("Sepal Length (cm)", "5.1")
sepal_width = st.text_input("Sepal Width (cm)", "3.5")

petal_length = st.slider("Petal Length (cm)", 1.0, 7.0, 1.4)
petal_width = st.selectbox("Petal Width (cm)", [0.1, 0.2, 0.5, 1.0, 1.5,2.0])

input_data = pd.DataFrame({
 "sepal length (cm)": [float(sepal_length)],
 "sepal width (cm)": [float(sepal_width)],
 "petal length (cm)": [petal_length],
 "petal width (cm)": [petal_width]
})

prediction = model.predict(input_data)
species = iris.target_names[prediction[0]]

st.subheader("Prediction Result")
st.success(f"The predicted flower species is: {species}")

