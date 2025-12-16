import streamlit as st
import pandas as pd
import numpy as np
import joblib

# 1. Load the trained model
@st.cache_resource
def load_model():
    return joblib.load('car_price_model.pkl')

pipeline = load_model()

# 2. App Title and Description
st.title("🚗 US Used Car Price Predictor")
st.write("Enter the vehicle details below to estimate its market price.")

# 3. Create Input Form
with st.form("prediction_form"):
    st.header("Vehicle Details")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Numeric Inputs
        horsepower = st.number_input("Horsepower", min_value=50, max_value=1000, value=200)
        mileage = st.number_input("Mileage (miles)", min_value=0, max_value=300000, value=50000)
        vehicle_age = st.number_input("Vehicle Age (years)", min_value=0, max_value=30, value=5)
        daysonmarket = st.number_input("Days on Market", min_value=0, value=30)
    
    with col2:
        # Categorical Inputs (You must match the options in your training data)
        # Note: These lists should be populated based on your actual data 
        # For this demo, we use common examples.
        make_name = st.selectbox("Make", ['Jeep' 'Land Rover' 'Mazda' 'Alfa Romeo' 'BMW' 'Hyundai' 'Chevrolet'
        'Lexus' 'Subaru' 'Cadillac' 'Chrysler' 'Dodge' 'Mercedes-Benz' 'Nissan'
        'Honda' 'Kia' 'Ford' 'Lincoln' 'Audi' 'Volkswagen' 'RAM' 'Porsche'
        'Jaguar' 'Toyota' 'INFINITI' 'GMC' 'Acura' 'Maserati' 'FIAT' 'Volvo'
        'Mitsubishi' 'Buick' 'Mercury' 'Scion' 'Saab' 'MINI' 'Ferrari' 'Genesis'
        'Saturn' 'Bentley' 'Suzuki' 'Tesla' 'Fisker' 'Pontiac' 'Lamborghini'
        'smart' 'Hummer' 'Rolls-Royce' 'Lotus' 'Spyker' 'McLaren' 'Aston Martin'
        'Maybach' 'Freightliner' 'Isuzu' 'Oldsmobile' 'Plymouth' 'Pagani' 'Karma'
        'AM General' 'Geo' 'SRT' 'VPG' 'Eagle' 'Bugatti' 'Daewoo' 'Ariel'
        'Shelby' 'Mobility Ventures' 'Saleen' 'Koenigsegg' 'Rover'])
        body_type = st.selectbox("Body Type", ["Sedan", "SUV / Crossover", "Truck", "Coupe", "Hatchback", "Van"])
        wheel_system = st.selectbox("Drivetrain", ["FWD", "AWD", "RWD", "4WD"])
        fuel_type = st.selectbox("Fuel Type", ["Gasoline", "Diesel", "Hybrid", "Electric"])

    # Hidden/Default Inputs 
    # (If your model expects these but you don't want the user to type them, set defaults)
    transmission_simple = "automatic" 
    engine_cylinders_num = 4.0
    city_fuel_economy = 25.0
    highway_fuel_economy = 32.0
    engine_displacement = 2500.0
    is_new = False
    
    # Submit Button
    submit_val = st.form_submit_button("Predict Price")

# 4. Make Prediction
if submit_val:
    # Create a dataframe matching the model's training input exactly
    input_data = pd.DataFrame({
        'horsepower': [horsepower],
        'mileage': [mileage],
        'vehicle_age': [vehicle_age],
        'daysonmarket': [daysonmarket],
        'make_name': [make_name],
        'body_type': [body_type],
        'wheel_system': [wheel_system],
        'fuel_type': [fuel_type],
        # Include the features we defaulted (or add inputs for them above)
        'transmission_simple': [transmission_simple],
        'engine_cylinders_num': [engine_cylinders_num],
        'city_fuel_economy': [city_fuel_economy],
        'highway_fuel_economy': [highway_fuel_economy],
        'engine_displacement': [engine_displacement],
        'is_new': [is_new],
        # Add any other missing columns your model expects here with dummy values
        'model_name': ['Unknown'], 
        'trim_name': ['Unknown'],
        'engine_type': ['Unknown'],
        'owner_count': [1.0],
        'torque': [200.0]
    })

    # Predict log_price
    pred_log = pipeline.predict(input_data)[0]
    
    # Convert back to dollars (exp(x) - 1)
    pred_price = np.expm1(pred_log)
    
    st.success(f"### Estimated Price: ${pred_price:,.2f}")
    
    # Optional: Show input data for debugging
    with st.expander("See input features"):
        st.write(input_data)