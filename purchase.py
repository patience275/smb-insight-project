
import streamlit as st
import numpy as np
import pickle

def load_model():
    with open('purchase_steps.pkl', 'rb') as file:
        data = pickle.load(file)
    return data

data = load_model()
model = data['model']
encoder = data['encoder']
scaler = data['scaler']

st.title('Will a customer come back? Fill in the details to find out')

total_orders = st.number_input('How many orders did your customer make?', min_value=0, max_value=100)
avg_order_value = st.slider('Average order value?', min_value=1000, max_value=10000)
purchase_frequency = st.number_input('How many times does your customer shop?', min_value=0, max_value=100)
num_categories_bought = st.number_input('Number of categories bought?', min_value=0, max_value=100)
review = st.number_input('What review did your customer leave?', min_value=0, max_value=5)
state = ['SP', 'MG', 'ES', 'RJ', 'RS', 'BA', 'CE', 'PR', 'MS', 'PB', 'SC',
         'MT', 'PA', 'RN', 'PI', 'DF', 'GO', 'PE', 'RO', 'MA', 'SE', 'AM',
         'AL', 'TO', 'AC', 'AP', 'RR']
location = st.radio('Where does your customer reside?', state)

# Encode location
location_encoded = encoder.transform([location])[0]

# Prepare features in the correct order
features = np.array([[total_orders, avg_order_value, purchase_frequency,
                      num_categories_bought, review, location_encoded]])

# Scale features
features_encoded = scaler.transform(features)

if st.button('Predict'):
    prediction = model.predict(features_encoded)
    label = "Yes" if prediction[0] == 0 else "No"
    st.success(f"Will my customer come back?: {label}")