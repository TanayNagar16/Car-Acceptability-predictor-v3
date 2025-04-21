import streamlit as st
import pickle
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Load the trained model
try:
    with open('car_evaluation_model.pkl', 'rb') as file:
        model = pickle.load(file)
except FileNotFoundError:
    st.error("Error: Trained model file 'car_evaluation_model.pkl' not found. Please run car_evaluation.py first.")
    st.stop()

# Load the label encoders
try:
    with open('car_evaluation_encoders.pkl', 'rb') as file:
        encoders = pickle.load(file)
except FileNotFoundError:
    st.error("Error: Encoders file 'car_evaluation_encoders.pkl' not found. Please run car_evaluation.py first.")
    st.stop()

st.title('Car Acceptability Prediction')

buying_options = ['vhigh', 'high', 'med', 'low']
maint_options = ['vhigh', 'high', 'med', 'low']
doors_options = ['2', '3', '4', '5more']
persons_options = ['2', '4', 'more']
lug_boot_options = ['small', 'med', 'big']
safety_options = ['low', 'med', 'high']

buying = st.selectbox('Buying Price', buying_options)
maint = st.selectbox('Maintenance Price', maint_options)
doors = st.selectbox('Number of Doors', doors_options)
persons = st.selectbox('Number of Persons', persons_options)
lug_boot = st.selectbox('Luggage Boot Size', lug_boot_options)
safety = st.selectbox('Safety', safety_options)

if st.button('Predict Acceptability'):
    input_data = pd.DataFrame({
        'buying': [buying],
        'maint': [maint],
        'doors': [doors],
        'persons': [persons],
        'lug_boot': [lug_boot],
        'safety': [safety]
    })

    encoded_data = {}
    for col in input_data.columns:
        # Ensure the encoder exists for the column
        if col in encoders:
            encoded_data[col] = encoders[col].transform(input_data[col])
        else:
            st.error(f"Error: Encoder not found for column '{col}'.")
            st.stop()

    input_array = pd.DataFrame(encoded_data).values

    prediction_encoded = model.predict(input_array)[0]
    # Ensure the 'class' encoder exists
    if 'class' in encoders:
        prediction = encoders['class'].inverse_transform([prediction_encoded])[0]
        st.write(f'The predicted car acceptability is: *{prediction}*')
    else:
        st.error("Error: Encoder not found for the 'class' column.")