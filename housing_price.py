import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

data = pd.read_csv("dataset/Bengaluru_House_Data.csv")
data = data.dropna()

def convert_sqft(x):
   try:
     if '-' in str(x):
       nums = x.split('-')
       return (float(nums[0]) + float(nums[1])) / 2
     return float(x)
   except:
     return None

data['total_sqft'] = data['total_sqft'].apply(convert_sqft)
data = data.dropna()

data['bhk'] = data['size'].str.extract('(\d+)').astype(int)

data = data[['total_sqft', 'bath', 'bhk', 'price']]
data.columns = ['Area', 'Bathrooms', 'Bedrooms', 'Price']

X = data[['Area', 'Bedrooms', 'Bathrooms']]
y = data['Price']

model = LinearRegression()
model.fit(X, y)

st.title(" Bengaluru Housing Price Prediction")
st.write("Enter property details to estimate price.")

area = st.number_input("Area (sq. ft)", 500, 10000, 1000)
bedrooms = st.selectbox("Bedrooms (BHK)", [1, 2, 3, 4, 5])
bathrooms = st.selectbox("Bathrooms", [1, 2, 3, 4])
input_data = pd.DataFrame([[area, bedrooms, bathrooms]],
 columns=['Area', 'Bedrooms', 'Bathrooms'])
prediction = model.predict(input_data)[0]

st.subheader("Predicted Price")
st.success(f"Estimated Price: ₹ {prediction:.2f} Lakhs")

st.subheader("Prediction Visualization")
fig, ax = plt.subplots()
ax.bar(["Predicted Price"], [prediction])
ax.set_ylabel("Price (Lakhs)")
st.pyplot(fig)
