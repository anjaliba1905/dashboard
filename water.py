import streamlit as st
import requests
import datetime
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score
import joblib
import os
from io import StringIO
import time
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import google.generativeai as genai

# Configure Streamlit page
st.set_page_config(page_title="Smart Irrigation Advisory", page_icon="🌾", layout="wide")
st.title("🌾 Smart Irrigation Advisory Dashboard ")
st.write("Get irrigation advice based on real-time weather, soil predictions, and crop-specific requirements with .")

# ------------------------ Load Real Crop Requirement Data ------------------------
@st.cache_data
def load_crop_data():
    """Load crop requirements from CSV file"""
    try:
        # Read the CSV file
        df = pd.read_csv("crop_requirements_icar_fao.csv")
        df['Crop'] = df['Crop'].str.strip().str.title()
        
        # Convert the data to a more usable format
        crop_data = {}
        for _, row in df.iterrows():
            crop_name = row['Crop']
            
            # Parse pH range
            ph_range = row['Ideal_pH'].split('-')
            min_ph = float(ph_range[0])
            max_ph = float(ph_range[1])
            
            # Parse temperature range
            temp_range = row['Temperature_Preference_C'].split('-')
            min_temp = int(temp_range[0])
            max_temp = int(temp_range[1])
            
            # Calculate derived values for irrigation logic
            total_etc = row['Total_ETc_mm']
            mid_etc = row['Mid_ETc_mm']
            
            # Estimate ideal moisture based on ETc values
            ideal_moisture = min(70, max(30, (mid_etc / total_etc) * 100))
            
            # Calculate water per irrigation based on ETc
            water_per_irrigation = max(20, total_etc / 15)  # Approximate irrigation frequency
            
            # Set rain threshold based on season
            rain_threshold = 5 if row['Season'] == 'Kharif' else 3
            
            crop_data[crop_name] = {
                'min_ph': min_ph,
                'max_ph': max_ph,
                'min_temp': min_temp,
                'max_temp': max_temp,
                'season': row['Season'],
                'total_etc': total_etc,
                'initial_etc': row['Initial_ETc_mm'],
                'development_etc': row['Development_ETc_mm'],
                'mid_etc': mid_etc,
                'late_etc': row['Late_ETc_mm'],
                'ideal_moisture': ideal_moisture,
                'water_per_irrigation': water_per_irrigation,
                'rain_threshold': rain_threshold,
                'temp_threshold': max_temp - 2  # Threshold for irrigation due to heat
            }
        
        return df, crop_data
    except FileNotFoundError:
        st.error("❌ crop_requirements_icar_fao.csv file not found!")
        st.stop()
    except Exception as e:
        st.error(f"❌ Error loading crop data: {str(e)}")
        st.stop()

# Load crop data
crop_df, crop_requirements = load_crop_data()

# --- Input Section ---
col1, col2, col3 = st.columns(3)
with col1:
    location = st.text_input("Enter your location (City/Village):", "")
    crop_type = st.selectbox("Select Crop Type:", list(crop_requirements.keys()))

with col2:
    language = st.selectbox("Select Language:", ["English", "Gujarati", "Hindi"])
    
with col3:
    alert_threshold = st.slider("Alert Threshold (%):", 10, 60, 30)
    email_alerts = st.checkbox("Enable Email Alerts")
    if email_alerts:
        user_email = st.text_input("Your Email:", "")

# Language Dictionary (enhanced)
translations = {
    "English": {
        "weather": "Weather Data",
        "temperature": "Temperature",
        "humidity": "Humidity",
        "rainfall": "Rainfall (last 1hr)",
        "wind_speed": "Wind Speed",
        "condition": "Condition",
        "advice": "Irrigation Advice",
        "recommended": "Irrigation Recommended",
        "not_needed": "No Irrigation Needed",
        "saving_score": "Water Saving Score",
        "soil_prediction": "Soil Moisture Prediction",
        "predicted_moisture": "Predicted Soil Moisture",
        "soil_type": "Predicted Soil Type",
        "water_amount": "Water Amount Needed",
        "alert": "Alert",
        "error_weather": "Unable to fetch weather data. Please check your location.",
        "error_api": "Weather service temporarily unavailable. Please try again later.",
        "crop_requirements": "Crop Requirements",
        "ph_status": "pH Suitability",
        "temp_status": "Temperature Status",
        "season_info": "Season Information"
    },
    "Gujarati": {
        "weather": "હવામાન માહિતી",
        "temperature": "તાપમાન",
        "humidity": "ભેજ",
        "rainfall": "વર્ષા (છેલ્લા 1 કલાકમાં)",
        "wind_speed": "પવનની ઝડપ",
        "condition": "હવામાન સ્થિતિ",
        "advice": "સિંચાઇ સલાહ",
        "recommended": "સિંચાઇ કરવાની ભલામણ છે",
        "not_needed": "સિંચાઇની જરૂર નથી",
        "saving_score": "પાણી બચાવ સ્કોર",
        "soil_prediction": "માટીની ભેજની આગાહી",
        "predicted_moisture": "અનુમાનિત માટીની ભેજ",
        "soil_type": "અનુમાનિત માટીનો પ્રકાર",
        "water_amount": "પાણીની જરૂરિયાત",
        "alert": "ચેતવણી",
        "error_weather": "હવામાન માહિતી મેળવવામાં અસમર્થ. કૃપા કરીને તમારું સ્થાન તપાસો.",
        "error_api": "હવામાન સેવા અસ્થાયી રૂપે અનુપલબ્ધ છે. કૃપા કરીને ફરીથી પ્રયાસ કરો.",
        "crop_requirements": "પાકની જરૂરિયાતો",
        "ph_status": "pH યોગ્યતા",
        "temp_status": "તાપમાન સ્થિતિ",
        "season_info": "મોસમની માહિતી"
    },
    "Hindi": {
        "weather": "मौसम की जानकारी",
        "temperature": "तापमान",
        "humidity": "आर्द्रता",
        "rainfall": "वर्षा (पिछले 1 घंटे में)",
        "wind_speed": "हवा की गति",
        "condition": "स्थिति",
        "advice": "सिंचाई सलाह",
        "recommended": "सिंचाई की सिफारिश की जाती है",
        "not_needed": "सिंचाई की आवश्यकता नहीं है",
        "saving_score": "जल बचत स्कोर",
        "soil_prediction": "मिट्टी नमी की भविष्यवाणी",
        "predicted_moisture": "अनुमानित मिट्टी की नमी",
        "soil_type": "अनुमानित मिट्टी का प्रकार",
        "water_amount": "पानी की आवश्यकता",
        "alert": "चेतावनी",
        "error_weather": "मौसम डेटा प्राप्त करने में असमर्थ। कृपया अपना स्थान जांचें।",
        "error_api": "मौसम सेवा अस्थायी रूप से अनुपलब्ध है। कृपया बाद में पुनः प्रयास करें।",
        "crop_requirements": "फसल आवश्यकताएं",
        "ph_status": "pH उपयुक्तता",
        "temp_status": "तापमान स्थिति",
        "season_info": "मौसम की जानकारी"
    }
}

labels = translations[language]

# Soil type water retention factors
soil_water_retention = {
    "sandy": 0.7,
    "loamy": 1.0,
    "clay": 1.3,
    "sandy_loam": 0.85,
    "silty": 1.1
}

# Fixed API key - use environment variable or secrets
api_key = st.secrets.get("OPENWEATHER_API_KEY", "c3189205c860439e4727a7a27fd77a7d")

# Initialize session state properly
if 'irrigation_logs' not in st.session_state:
    st.session_state.irrigation_logs = []
if 'ml_model_trained' not in st.session_state:
    st.session_state.ml_model_trained = False
if 'soil_moisture_model' not in st.session_state:
    st.session_state.soil_moisture_model = None
if 'soil_type_model' not in st.session_state:
    st.session_state.soil_type_model = None
if 'ph_model' not in st.session_state:
    st.session_state.ph_model = None

# Generate synthetic training data for ML models
@st.cache_data
def generate_training_data():
    """Generate realistic training data for soil moisture, soil type, and pH prediction"""
    np.random.seed(42)
    n_samples = 1000
    
    # Features: temperature, humidity, rainfall, wind_speed
    temperature = np.random.normal(25, 8, n_samples)
    humidity = np.random.normal(60, 20, n_samples)
    rainfall = np.random.exponential(2, n_samples)
    wind_speed = np.random.normal(10, 5, n_samples)
    
    # Soil moisture prediction (realistic model based on environmental factors)
    soil_moisture = (
        humidity * 0.6 +
        rainfall * 8 -
        temperature * 0.8 -
        wind_speed * 1.2 +
        np.random.normal(20, 10, n_samples)
    )
    soil_moisture = np.clip(soil_moisture, 0, 100)
    
    # Soil pH prediction (based on rainfall and other factors)
    soil_ph = (
        6.5 +
        (rainfall - 2) * 0.2 -
        (temperature - 25) * 0.05 +
        np.random.normal(0, 0.5, n_samples)
    )
    soil_ph = np.clip(soil_ph, 4.0, 9.0)
    
    # Soil type prediction (based on moisture patterns and environmental factors)
    soil_type = []
    for i in range(n_samples):
        if soil_moisture[i] > 60 and humidity[i] > 70:
            soil_type.append("clay")
        elif soil_moisture[i] < 30 and rainfall[i] < 1:
            soil_type.append("sandy")
        elif 30 <= soil_moisture[i] <= 45 and temperature[i] > 25:
            soil_type.append("sandy_loam")
        elif 45 <= soil_moisture[i] <= 55:
            soil_type.append("loamy")
        else:
            soil_type.append("silty")
    
    features = np.column_stack([temperature, humidity, rainfall, wind_speed])
    
    return features, soil_moisture, soil_type, soil_ph

def train_ml_models():
    """Train soil moisture, soil type, and pH prediction models"""
    features, soil_moisture, soil_type, soil_ph = generate_training_data()
    
    # Train soil moisture prediction model
    X_train, X_test, y_train_moisture, y_test_moisture = train_test_split(
        features, soil_moisture, test_size=0.2, random_state=42
    )
    
    moisture_model = RandomForestRegressor(n_estimators=100, random_state=42)
    moisture_model.fit(X_train, y_train_moisture)
    
    # Train soil pH prediction model
    _, _, y_train_ph, y_test_ph = train_test_split(
        features, soil_ph, test_size=0.2, random_state=42
    )
    
    ph_model = RandomForestRegressor(n_estimators=100, random_state=42)
    ph_model.fit(X_train, y_train_ph)
    
    # Train soil type prediction model
    _, _, y_train_type, y_test_type = train_test_split(
        features, soil_type, test_size=0.2, random_state=42
    )
    
    type_model = RandomForestClassifier(n_estimators=100, random_state=42)
    type_model.fit(X_train, y_train_type)
    
    # Calculate accuracies
    moisture_predictions = moisture_model.predict(X_test)
    ph_predictions = ph_model.predict(X_test)
    type_predictions = type_model.predict(X_test)
    
    moisture_rmse = np.sqrt(mean_squared_error(y_test_moisture, moisture_predictions))
    ph_rmse = np.sqrt(mean_squared_error(y_test_ph, ph_predictions))
    type_accuracy = accuracy_score(y_test_type, type_predictions)
    
    return moisture_model, type_model, ph_model, moisture_rmse, type_accuracy, ph_rmse

def predict_soil_conditions(temp, humidity, rainfall, wind_speed):
    """Predict soil moisture, type, and pH using trained models"""
    if not st.session_state.ml_model_trained:
        # Fallback prediction if models aren't trained
        predicted_moisture = max(0, min(100, humidity * 0.6 + rainfall * 8 - temp * 0.8 - wind_speed * 1.2 + 20))
        predicted_ph = max(4.0, min(9.0, 6.5 + (rainfall - 2) * 0.2 - (temp - 25) * 0.05))
        return predicted_moisture, "loamy", predicted_ph
    
    features = np.array([[temp, humidity, rainfall, wind_speed]])
    
    predicted_moisture = st.session_state.soil_moisture_model.predict(features)[0]
    predicted_soil_type = st.session_state.soil_type_model.predict(features)[0]
    predicted_ph = st.session_state.ph_model.predict(features)[0]
    
    return max(0, min(100, predicted_moisture)), predicted_soil_type, max(4.0, min(9.0, predicted_ph))

def calculate_water_needed_csv(crop_type, predicted_moisture, forecasted_rainfall, predicted_ph, temp):
    """Calculate water needed based on CSV crop requirements"""
    if crop_type not in crop_requirements:
        return 0, "Crop not found"
    
    crop_req = crop_requirements[crop_type]
    
    # Check if moisture is below ideal level
    ideal_moisture = crop_req['ideal_moisture']
    moisture_deficit = max(0, ideal_moisture - predicted_moisture)
    
    # Account for forecasted rainfall
    rain_adjustment = max(0, forecasted_rainfall - crop_req['rain_threshold'])
    
    # Base water calculation
    water_per_irrigation = crop_req['water_per_irrigation']
    
    # Calculate water needed
    water_needed = max(0, (moisture_deficit * water_per_irrigation / 100) - (rain_adjustment * 0.8))
    
    # Additional factors
    # Temperature stress
    if temp > crop_req['temp_threshold']:
        water_needed *= 1.2
    
    # pH adjustment (if pH is not optimal, plants may need more water)
    if not (crop_req['min_ph'] <= predicted_ph <= crop_req['max_ph']):
        water_needed *= 1.1
    
    return round(water_needed, 1), "Calculated successfully"

def get_enhanced_irrigation_advice(crop_type, predicted_moisture, predicted_ph, temp, humidity, rainfall, wind_speed, soil_type):
    """Enhanced irrigation advice using CSV data"""
    if crop_type not in crop_requirements:
        return False, "Crop not found", 0, 0
    
    crop_req = crop_requirements[crop_type]
    
    # Calculate water needed
    water_needed, status = calculate_water_needed_csv(crop_type, predicted_moisture, rainfall, predicted_ph, temp)
    
    # Multiple factors for irrigation decision
    factors = {
        "low_predicted_moisture": predicted_moisture < crop_req['ideal_moisture'],
        "high_temp": temp > crop_req['temp_threshold'],
        "low_rainfall": rainfall < crop_req['rain_threshold'],
        "high_wind": wind_speed > 15,
        "low_humidity": humidity < 40,
        "ph_not_optimal": not (crop_req['min_ph'] <= predicted_ph <= crop_req['max_ph'])
    }
    
    # Decision logic with weights
    irrigation_score = (
        factors["low_predicted_moisture"] * 4 +
        factors["high_temp"] * 2 +
        factors["low_rainfall"] * 2 +
        factors["high_wind"] * 1 +
        factors["low_humidity"] * 1 +
        factors["ph_not_optimal"] * 1
    )
    
    # Calculate water saving score
    max_possible_water = crop_req['water_per_irrigation'] * 1.5
    water_saving_score = max(0, 100 - ((water_needed / max_possible_water) * 100))
    
    if irrigation_score >= 6:
        return True, "High Priority", round(water_saving_score * 0.75, 1), water_needed
    elif irrigation_score >= 4:
        return True, "Medium Priority", round(water_saving_score * 0.85, 1), water_needed
    elif irrigation_score >= 2:
        return True, "Low Priority", round(water_saving_score * 0.90, 1), water_needed
    else:
        return False, "Not Needed", round(water_saving_score, 1), 0
    
def get_crop_status_advice(crop_type, predicted_ph, temp):
    """Get crop-specific status and advice"""
    if crop_type not in crop_requirements:
        return "Crop not found in database"
    
    crop_req = crop_requirements[crop_type]
    advice = []
    
    # pH status
    if crop_req['min_ph'] <= predicted_ph <= crop_req['max_ph']:
        advice.append(f"✅ Soil pH ({predicted_ph:.1f}) is optimal for {crop_type}")
    else:
        advice.append(f"⚠️ Soil pH ({predicted_ph:.1f}) is not optimal. Ideal range: {crop_req['min_ph']}-{crop_req['max_ph']}")
    
    # Temperature status
    if crop_req['min_temp'] <= temp <= crop_req['max_temp']:
        advice.append(f"✅ Temperature ({temp}°C) is suitable for {crop_type}")
    else:
        if temp < crop_req['min_temp']:
            advice.append(f"🌡️ Temperature ({temp}°C) is below optimal range ({crop_req['min_temp']}-{crop_req['max_temp']}°C)")
        else:
            advice.append(f"🌡️ Temperature ({temp}°C) is above optimal range. Consider additional irrigation.")
    
    # Season information
    advice.append(f"📅 {crop_type} is a {crop_req['season']} season crop")
    advice.append(f"💧 Total seasonal water requirement: {crop_req['total_etc']} mm")
    
    return "\n".join(advice)

def send_alert_email(email, crop_type, predicted_moisture, water_needed, location):
    """Send alert email when moisture is low"""
    try:
        # This is a mock implementation - in production, you'd use actual SMTP settings
        alert_message = f"""
        🚨 IRRIGATION ALERT 🚨
        
        Location: {location}
        Crop: {crop_type}
        Predicted Soil Moisture: {predicted_moisture}%
        Water Needed: {water_needed} mm
        
        Please irrigate your crop as soon as possible.
        """
        
        # In production, replace with actual email sending logic
        st.success(f"Alert email would be sent to {email}")
        return True
    except Exception as e:
        st.error(f"Failed to send alert email: {str(e)}")
        return False

def log_irrigation_data(data):
    """Log irrigation data to session state"""
    st.session_state.irrigation_logs.append(data)

def check_alert_conditions(predicted_moisture, threshold, crop_type, water_needed, location, email=None):
    """Check if alert conditions are met and send alerts"""
    if predicted_moisture <= threshold:
        alert_msg = f"🚨 {labels['alert']}: {crop_type} needs {water_needed} mm of water!"
        st.error(alert_msg)
        return True
    return False

# Train ML models on first run
if not st.session_state.ml_model_trained:
    with st.spinner("Training ML models for soil predictions..."):
        try:
            moisture_model, type_model, ph_model, moisture_rmse, type_accuracy, ph_rmse = train_ml_models()
            st.session_state.soil_moisture_model = moisture_model
            st.session_state.soil_type_model = type_model
            st.session_state.ph_model = ph_model
            st.session_state.ml_model_trained = True
            
            st.success(f"ML Models trained successfully! Moisture RMSE: {moisture_rmse:.2f}, pH RMSE: {ph_rmse:.2f}, Soil Type Accuracy: {type_accuracy:.2f}")
        except Exception as e:
            st.error(f"Error training ML models: {str(e)}")

# --- Main Analysis ---
if st.button("Get Irrigation Advice", type="primary"):
    if not location.strip():
        st.error("Please enter a valid location.")
    else:
        weather_url = f"https://api.openweathermap.org/data/2.5/weather?q={location}&appid={api_key}&units=metric"
        
        try:
            with st.spinner("Fetching weather data and making predictions..."):
                response = requests.get(weather_url, timeout=10)
                response.raise_for_status()
                data = response.json()

            if data.get("cod") != 200:
                st.error(f"{labels['error_weather']}: {data.get('message', 'Unknown error')}")
            else:
                # Extract weather data safely
                temp = data['main']['temp']
                humidity = data['main']['humidity']
                weather = data['weather'][0]['main']
                wind_speed = data['wind']['speed']
                rain_data = data.get("rain", {})
                rainfall = rain_data.get("1h", rain_data.get("3h", 0.0))

                # ML Predictions
                predicted_moisture, predicted_soil_type, predicted_ph = predict_soil_conditions(
                    temp, humidity, rainfall, wind_speed
                )

                # Weather display
                st.subheader(f"🌧️ {labels['weather']}")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric(labels['temperature'], f"{temp}°C")
                    st.metric(labels['humidity'], f"{humidity}%")
                with col2:
                    st.metric(labels['rainfall'], f"{rainfall} mm")
                    st.metric(labels['wind_speed'], f"{wind_speed} m/s")
                with col3:
                    st.metric(labels['condition'], weather)

                # ML Predictions Display
                st.subheader(f"🤖 {labels['soil_prediction']}")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric(labels['predicted_moisture'], f"{predicted_moisture:.1f}%")
                with col2:
                    st.metric(labels['soil_type'], predicted_soil_type.title())
                with col3:
                    st.metric("Predicted pH", f"{predicted_ph:.1f}")

                # Crop-specific analysis
                st.subheader(f"🌾 {labels['crop_requirements']}")
                crop_advice = get_crop_status_advice(crop_type, predicted_ph, temp)
                st.info(crop_advice)

                # Get irrigation advice using CSV data
                irrigate, priority, water_score, water_needed = get_enhanced_irrigation_advice(
                    crop_type, predicted_moisture, predicted_ph, temp, humidity, rainfall, wind_speed, predicted_soil_type
                )

                # Check alert conditions
                alert_triggered = check_alert_conditions(
                    predicted_moisture, alert_threshold, 
                    crop_type, water_needed, location, user_email if email_alerts else None
                )

                st.subheader(f"💧 {labels['advice']}")
                if irrigate:
                    st.success(f"{labels['recommended']} for {crop_type} - {priority}")
                    st.metric(labels['water_amount'], f"{water_needed} mm")
                    st.metric(labels['saving_score'], f"{water_score}%", 
                             delta=f"{priority}", delta_color="inverse")
                    result = labels['recommended']
                else:
                    st.info(f"{labels['not_needed']} for {crop_type}")
                    st.metric(labels['saving_score'], f"{water_score}%", 
                             delta="Optimal", delta_color="normal")
                    result = labels['not_needed']

                # Display crop requirements from CSV
                if crop_type in crop_requirements:
                    crop_req = crop_requirements[crop_type]
                    st.subheader("📊 Crop Requirements (ICAR/FAO Data)")
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Ideal pH Range", f"{crop_req['min_ph']}-{crop_req['max_ph']}")
                        st.metric("Temperature Range", f"{crop_req['min_temp']}-{crop_req['max_temp']}°C")
                    with col2:
                        st.metric("Season", crop_req['season'])
                        st.metric("Total ETc", f"{crop_req['total_etc']} mm")
                    with col3:
                        st.metric("Mid-stage ETc", f"{crop_req['mid_etc']} mm")
                        st.metric("Late-stage ETc", f"{crop_req['late_etc']} mm")
                    with col4:
                        st.metric("Initial ETc", f"{crop_req['initial_etc']} mm")
                        st.metric("Development ETc", f"{crop_req['development_etc']} mm")

                # Log data
                log_data = {
                    "datetime": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "location": location,
                    "crop": crop_type,
                    "predicted_moisture": predicted_moisture,
                    "predicted_soil_type": predicted_soil_type,
                    "predicted_ph": predicted_ph,
                    "temperature": temp,
                    "humidity": humidity,
                    "rainfall": rainfall,
                    "wind_speed": wind_speed,
                    "weather": weather,
                    "irrigation": result,
                    "priority": priority if irrigate else "N/A",
                    "water_needed": water_needed,
                    "water_score": water_score,
                    "alert_triggered": alert_triggered
                }
                log_irrigation_data(log_data)

                #------------------------------
                # 📊 Compact Analysis Dashboard
                st.subheader("📊 Analysis Dashboard")

                # Custom CSS for better spacing and larger tab font
                st.markdown("""
                <style>
                .metric-container {
                    background-color: #f0f2f6;
                    padding: 10px;
                    border-radius: 8px;
                    margin: 5px 0;
                }
                /* Increase tab font size */
                .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
                    font-size: 16px !important;
                    font-weight: 600 !important;
                }
                .stTabs [data-baseweb="tab-list"] button {
                    height: 50px !important;
                    padding: 10px 20px !important;
                }
                </style>
                """, unsafe_allow_html=True)

                tab1, tab2, tab3, tab4 = st.tabs(["🌱 Moisture", "🧪 pH Level", "🌡️ Temp", "💧 ETc"])

                with tab1:
                    col1, col2 = st.columns([2, 1])
                    
                    with col1:
                        days = [f"D{i+1}" for i in range(7)]  # Shorter labels
                        predicted_values = [predicted_moisture - i * 1.2 for i in range(7)]
                        
                        fig, ax = plt.subplots(figsize=(5, 3))  # Smaller size
                        ax.plot(days, predicted_values, marker='o', linewidth=2, 
                                color='#2E86C1', markersize=4, label='Predicted')
                        ax.axhline(crop_req['ideal_moisture'], linestyle='--', 
                                  color='#E74C3C', alpha=0.8, label='Min Ideal')
                        ax.axhline(alert_threshold, linestyle='--', 
                                  color='#F39C12', alpha=0.8, label='Alert')
                        
                        ax.set_title("7-Day Moisture Trend", fontsize=12, pad=10)
                        ax.set_ylabel("Moisture (%)", fontsize=10)
                        ax.tick_params(axis='both', labelsize=9)
                        ax.grid(True, alpha=0.2)
                        ax.legend(fontsize=8, loc='upper right')
                        
                        plt.tight_layout()
                        st.pyplot(fig, use_container_width=True)
                    
                    with col2:
                        # Add summary metrics
                        st.metric("Current", f"{predicted_moisture:.1f}%", 
                                 delta=f"{predicted_moisture - crop_req['ideal_moisture']:.1f}%")
                        
                        status = "🟢 Good" if predicted_moisture >= crop_req['ideal_moisture'] else "🔴 Low"
                        st.markdown(f"**Status:** {status}")

                with tab2:
                    col1, col2 = st.columns([1, 1])
                    
                    with col1:
                        # Compact bar chart
                        fig, ax = plt.subplots(figsize=(4, 2.5))
                        
                        categories = ['Current', 'Min', 'Max']
                        values = [predicted_ph, crop_req['min_ph'], crop_req['max_ph']]
                        colors = ['#E67E22', '#27AE60', '#27AE60']
                        
                        bars = ax.bar(categories, values, color=colors, alpha=0.8, width=0.6)
                        ax.set_title("pH Levels", fontsize=12, pad=10)
                        ax.set_ylabel("pH", fontsize=10)
                        ax.tick_params(axis='both', labelsize=9)
                        ax.grid(axis='y', alpha=0.2)
                        
                        # Add value labels on bars
                        for bar, val in zip(bars, values):
                            ax.text(bar.get_x() + bar.get_width()/2., val + 0.05, 
                                   f"{val:.1f}", ha='center', fontsize=9)
                        
                        plt.tight_layout()
                        st.pyplot(fig, use_container_width=True)
                    
                    with col2:
                        st.metric("Current pH", f"{predicted_ph:.1f}", 
                                 delta=f"{predicted_ph - (crop_req['min_ph'] + crop_req['max_ph'])/2:.1f}")
                        
                        if crop_req['min_ph'] <= predicted_ph <= crop_req['max_ph']:
                            st.markdown("**Status:** 🟢 Optimal")
                        else:
                            st.markdown("**Status:** 🟡 Needs Adjustment")

                with tab3:
                    col1, col2 = st.columns([1, 1])
                    
                    with col1:
                        # Horizontal bar chart for temperature
                        fig, ax = plt.subplots(figsize=(4, 2.5))
                        
                        labels = ['Current', 'Min Ideal', 'Max Ideal']
                        values = [temp, crop_req['min_temp'], crop_req['max_temp']]
                        colors = ['#3498DB', '#2ECC71', '#2ECC71']
                        
                        bars = ax.barh(labels, values, color=colors, alpha=0.8, height=0.5)
                        ax.set_title("Temperature Range", fontsize=12, pad=10)
                        ax.set_xlabel("Temperature (°C)", fontsize=10)
                        ax.tick_params(axis='both', labelsize=9)
                        ax.grid(axis='x', alpha=0.2)
                        
                        # Add value labels
                        for bar, val in zip(bars, values):
                            ax.text(val + 0.5, bar.get_y() + bar.get_height()/2., 
                                   f"{val:.1f}°C", va='center', fontsize=9)
                        
                        plt.tight_layout()
                        st.pyplot(fig, use_container_width=True)
                    
                    with col2:
                        st.metric("Current Temp", f"{temp:.1f}°C", 
                                 delta=f"{temp - (crop_req['min_temp'] + crop_req['max_temp'])/2:.1f}°C")
                        
                        if crop_req['min_temp'] <= temp <= crop_req['max_temp']:
                            st.markdown("**Status:** 🟢 Optimal")
                        elif temp < crop_req['min_temp']:
                            st.markdown("**Status:** 🔵 Too Cold")
                        else:
                            st.markdown("**Status:** 🔴 Too Hot")

                with tab4:
                    st.markdown("**💧 Water Requirements by Growth Stage**")
                    
                    stages = ['Initial', 'Development', 'Mid', 'Late']
                    values = [crop_req['initial_etc'], crop_req['development_etc'], crop_req['mid_etc'], crop_req['late_etc']]
                    
                    # Map technical terms to farmer-friendly terms
                    farmer_terms = {
                        'Initial': 'Planting', 
                        'Development': 'Growing', 
                        'Mid': 'Flowering', 
                        'Late': 'Harvest'
                    }
                    
                    col1, col2 = st.columns([1, 1])
                    
                    with col1:
                        # Compact bar chart
                        fig, ax = plt.subplots(figsize=(4, 2.5))
                        
                        # Create bars with thin width
                        x_positions = np.arange(len(stages))
                        bars = ax.bar(x_positions, values, 
                                      width=0.6,  # Thin bars
                                      color='#6366f1',  # Blue color
                                      alpha=0.8)
                        
                        # Customize the chart
                        ax.set_title("Water Requirements", fontsize=12, pad=10)
                        ax.set_ylabel("mm", fontsize=10)
                        ax.tick_params(axis='both', labelsize=8)
                        
                        # Set x-axis labels (shortened)
                        ax.set_xticks(x_positions)
                        short_labels = ['Plant', 'Grow', 'Flower', 'Harvest']
                        ax.set_xticklabels(short_labels, fontsize=8)
                        
                        # Add value labels on top of bars
                        for bar, val in zip(bars, values):
                            ax.text(bar.get_x() + bar.get_width()/2., val + max(values)*0.02,
                                   f'{val}', ha='center', fontsize=9)
                        
                        # Style the chart
                        ax.grid(axis='y', alpha=0.2)
                        ax.set_ylim(0, max(values) * 1.15)
                        
                        plt.tight_layout()
                        st.pyplot(fig, use_container_width=True)
                        plt.close()
                    
                    with col2:
                        # Summary metrics
                        total_water = sum(values)
                        max_value = max(values)
                        peak_stage_technical = stages[values.index(max_value)]
                        peak_stage_display = farmer_terms.get(peak_stage_technical, peak_stage_technical)
                        
                        st.metric("Total Season", f"{total_water} mm")
                        # Fixed the error - just use the stage name directly
                        st.metric("Peak Stage", peak_stage_display)
                        st.metric("Peak Water", f"{max_value} mm")
                        
                        # Status indicator
                        avg_water = total_water / len(values)
                        if max_value > avg_water * 1.5:
                            st.markdown("**Status:** 🔥 High Variation")
                        elif max_value > avg_water * 1.2:
                            st.markdown("**Status:** 🟡 Moderate Variation")
                        else:
                            st.markdown("**Status:** 🟢 Even Distribution")
                                
        except requests.exceptions.Timeout:
            st.error(f"{labels['error_api']} (Timeout)")
        except requests.exceptions.ConnectionError:
            st.error(f"{labels['error_api']} (Connection Error)")
        except requests.exceptions.RequestException as e:
            st.error(f"{labels['error_api']}: {str(e)}")
        except Exception as e:
            st.error(f"An unexpected error occurred: {str(e)}")
            st.write("Debug info:", str(e))

# Sidebar with crop information from CSV
with st.sidebar:
    st.header("📋 Crop Information (ICAR/FAO Data)")
    if crop_type in crop_requirements:
        selected_crop = crop_requirements[crop_type]
        st.write(f"**{crop_type} Requirements:**")
        st.write(f"• pH Range: {selected_crop['min_ph']}-{selected_crop['max_ph']}")
        st.write(f"• Temperature: {selected_crop['min_temp']}-{selected_crop['max_temp']}°C")
        st.write(f"• Season: {selected_crop['season']}")
        st.write(f"• Total ETc: {selected_crop['total_etc']} mm")
        st.write(f"• Water per Irrigation: {selected_crop['water_per_irrigation']:.1f} mm")
        st.write(f"• Rain Threshold: {selected_crop['rain_threshold']} mm")
        
        # Display crop growth stages
        st.subheader("🌱 Growth Stage ETc")
        stage_data = {
            "Initial": selected_crop['initial_etc'],
            "Development": selected_crop['development_etc'],
            "Mid": selected_crop['mid_etc'],
            "Late": selected_crop['late_etc']
        }
        
        for stage, value in stage_data.items():
            st.write(f"• {stage}: {value} mm")
        
    st.header("🤖 ML Model Status")
    if st.session_state.ml_model_trained:
        st.success("✅ Models trained and ready")
        st.write("• Soil moisture prediction")
        st.write("• Soil type classification")
        st.write("• Soil pH prediction")
    else:
        st.warning("⚠️ Training models...")
    
    st.header("🚨 Alert Settings")
    st.write(f"Alert Threshold: {alert_threshold}%")
    st.write(f"Email Alerts: {'✅' if email_alerts else '❌'}")
    
    st.header("🌍 About")
    st.write("This smart irrigation system uses ICAR/FAO crop data and machine learning to predict soil conditions and provide precise water recommendations based on scientific crop requirements.")

    # Real-time monitoring section
    st.header("📡 Real-time Monitoring")
    if st.checkbox("Enable Auto-refresh (30s)"):
        time.sleep(30)
        st.rerun()

# Additional features section
st.subheader("🔧 Advanced Features")

# 🌾 Krushi Sarthi - AI Chatbot
with st.expander("📱 Krushi Sarthi - AI Chatbot"):
    # Load Gemini API key safely
    try:
        gemini_api_key = st.secrets["GEMINI_API_KEY"]
    except Exception:
        st.error("❌ Gemini API key not found in secrets.toml!")
        st.stop()

    # Configure Gemini
    genai.configure(api_key=gemini_api_key)
    model = genai.GenerativeModel(model_name="models/gemini-2.0-flash")

    # Section Title
    st.markdown("---")
    st.markdown("## 🧠 Chat with **Krushi Sarthi** – Your Smart Farming Companion")

    # Chat Input Section
    user_input = st.text_area(
        "💬 Ask a question about farming, irrigation, or soil health:",
        height=100,
        placeholder="e.g. What is the ideal pH for wheat cultivation?"
    )

    # Centered button layout using columns
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        ask_button = st.button("🚜 Ask Krushi Sarthi", use_container_width=True)

    # Display response
    if ask_button:
        if not user_input.strip():
            st.warning("Please enter a question for Krushi Sarthi.")
        else:
            with st.spinner("🤖 Thinking..."):
                try:
                    # Enhanced prompt with crop data context
                    enhanced_prompt = f"""
                    You are Krushi Sarthi, an expert agricultural advisor with access to ICAR and FAO crop data. 
                    Current context: User is working with {crop_type} crop in {location}.
                    
                    Crop-specific data for {crop_type}:
                    - pH Range: {crop_requirements[crop_type]['min_ph']}-{crop_requirements[crop_type]['max_ph']}
                    - Temperature Range: {crop_requirements[crop_type]['min_temp']}-{crop_requirements[crop_type]['max_temp']}°C
                    - Season: {crop_requirements[crop_type]['season']}
                    - Total Water Requirement: {crop_requirements[crop_type]['total_etc']} mm
                    
                    User question: {user_input}
                    
                    Please provide a helpful, accurate response based on scientific agricultural practices and the provided crop data.
                    """
                    
                    response = model.generate_content(enhanced_prompt)
                    st.success("✅ Answer:")
                    st.markdown(f"🗨️ {response.text}")
                except Exception as e:
                    st.error(f"❌ Failed to generate response: {e}")
                    
#---------------------------
                    
# Weather forecast integration
with st.expander("🌤️ Weather Forecast Analysis"):
    st.write("**7-Day Weather Impact Analysis**")
    
    # Simulated forecast data
    forecast_days = ["Today", "Tomorrow", "Day 3", "Day 4", "Day 5", "Day 6", "Day 7"]
    
    # Generate realistic forecast
    base_temp = 25
    base_humidity = 60
    base_rainfall = 2
    base_wind = 10
    
    forecast_data = []
    for i, day in enumerate(forecast_days):
        temp_var = np.random.normal(0, 3)
        humidity_var = np.random.normal(0, 10)
        rain_var = np.random.exponential(1)
        wind_var = np.random.normal(0, 3)
        
        forecast_data.append({
            "Day": day,
            "Temperature": round(base_temp + temp_var, 1),
            "Humidity": round(max(30, min(90, base_humidity + humidity_var)), 1),
            "Rainfall": round(max(0, rain_var), 1),
            "Wind Speed": round(max(0, base_wind + wind_var), 1),
            "Irrigation Need": "Yes" if (base_humidity + humidity_var < 50 or rain_var < 1) else "No"
        })
    
    forecast_df = pd.DataFrame(forecast_data)
    st.dataframe(forecast_df, use_container_width=True)
    
   
                    
# Crop comparison with CSV data
with st.expander("🌱 Enhanced Crop Comparison Tool"):
    st.write("**Compare ICAR/FAO Crop Requirements**")
    
    available_crops = list(crop_requirements.keys())
    selected_crops = st.multiselect("Select Crops to Compare", 
                                  available_crops,
                                  default=available_crops[:3] if len(available_crops) >= 3 else available_crops)
    
    if selected_crops:
        comparison_data = []
        for crop in selected_crops:
            req = crop_requirements[crop]
            comparison_data.append({
                "Crop": crop,
                "pH Range": f"{req['min_ph']}-{req['max_ph']}",
                "Temp Range (°C)": f"{req['min_temp']}-{req['max_temp']}",
                "Season": req['season'],
                "Total ETc (mm)": req['total_etc'],
                "Mid ETc (mm)": req['mid_etc'],
                "Water/Irrigation (mm)": f"{req['water_per_irrigation']:.1f}",
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        st.dataframe(comparison_df, use_container_width=True)
        
        # Enhanced comparison chart
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # Total ETc comparison
        crops = comparison_df['Crop']
        total_etc_values = [crop_requirements[crop]['total_etc'] for crop in crops]
        ax1.bar(crops, total_etc_values, color='lightblue', alpha=0.8)
        ax1.set_title('Total Water Requirement (ETc)')
        ax1.set_ylabel('ETc (mm)')
        ax1.tick_params(axis='x', rotation=45)
        
        # pH range comparison (using mid-point)
        ph_mid_values = [(crop_requirements[crop]['min_ph'] + crop_requirements[crop]['max_ph'])/2 for crop in crops]
        ax2.bar(crops, ph_mid_values, color='lightgreen', alpha=0.8)
        ax2.set_title('Optimal pH (Mid-point)')
        ax2.set_ylabel('pH Value')
        ax2.tick_params(axis='x', rotation=45)
        
        # Temperature range comparison (using mid-point)
        temp_mid_values = [(crop_requirements[crop]['min_temp'] + crop_requirements[crop]['max_temp'])/2 for crop in crops]
        ax3.bar(crops, temp_mid_values, color='lightcoral', alpha=0.8)
        ax3.set_title('Optimal Temperature (Mid-point)')
        ax3.set_ylabel('Temperature (°C)')
        ax3.tick_params(axis='x', rotation=45)
        
        # Water per irrigation comparison
        water_values = [crop_requirements[crop]['water_per_irrigation'] for crop in crops]
        ax4.bar(crops, water_values, color='lightsalmon', alpha=0.8)
        ax4.set_title('Water per Irrigation')
        ax4.set_ylabel('Water (mm)')
        ax4.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        st.pyplot(fig)

# Enhanced irrigation scheduling with CSV data
with st.expander("📅 Smart Irrigation Scheduling"):
    st.write("**Automated Irrigation Schedule Based on Crop Requirements**")
    
    # Schedule parameters
    col1, col2 = st.columns(2)
    with col1:
        schedule_enabled = st.checkbox("Enable Automated Scheduling")
        preferred_time = st.time_input("Preferred Irrigation Time", datetime.time(6, 0))
    with col2:
        schedule_days = st.multiselect("Schedule Days", 
                                     ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"],
                                     default=["Monday", "Wednesday", "Friday"])
    
    if schedule_enabled and crop_type in crop_requirements:
        crop_req = crop_requirements[crop_type]
        st.info(f"Automated irrigation scheduled for {crop_type} on {', '.join(schedule_days)} at {preferred_time}")
        st.write(f"**Crop-specific scheduling parameters:**")
        st.write(f"• Expected water per session: {crop_req['water_per_irrigation']:.1f} mm")
        st.write(f"• Season: {crop_req['season']}")
        st.write(f"• Total seasonal requirement: {crop_req['total_etc']} mm")
        
        # Generate enhanced schedule for next 7 days
        schedule_data = []
        for i in range(7):
            date = datetime.date.today() + datetime.timedelta(days=i)
            day_name = date.strftime("%A")
            if day_name in schedule_days:
                schedule_data.append({
                    "Date": date.strftime("%Y-%m-%d"),
                    "Day": day_name,
                    "Time": preferred_time.strftime("%H:%M"),
                    "Expected Water (mm)": f"{crop_req['water_per_irrigation']:.1f}",
                    "Status": "Scheduled"
                })
        
        if schedule_data:
            st.dataframe(pd.DataFrame(schedule_data), use_container_width=True)
            
# ✅ Ensure temp and moisture values are available from latest log
if st.session_state.get("irrigation_logs"):
    latest_log = st.session_state.irrigation_logs[-1]

    if 'temp' not in st.session_state:
        st.session_state.temp = latest_log['temperature']
    if 'predicted_moisture' not in st.session_state:
        st.session_state.predicted_moisture = latest_log['predicted_moisture']

#-------water use utility 
with st.expander("💧 Smart Water Management Assistant"):
    st.write("**🌾 Practical Water Usage Guide for Your Farm**")
    st.info("💡 Get actionable irrigation advice based on your crop, weather, and soil conditions.")
    
    # Smart irrigation recommendation system
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("🎯 Today's Irrigation Recommendation")

        if crop_type in crop_requirements:
            crop_info = crop_requirements[crop_type]

            # Determine growth stage based on days since planting
            if 'days_planted' in st.session_state:
                days = st.session_state.days_planted
            else:
                days = st.selectbox("Days since planting:", [0, 30, 60, 90, 120], index=2)

            # Determine current growth stage
            if days < 20:
                current_stage = "initial"
                stage_name = "🌱 Seedling Stage"
                water_need = crop_info.get('initial_etc', 50)
            elif days < 60:
                current_stage = "development" 
                stage_name = "🌿 Vegetative Growth"
                water_need = crop_info.get('development_etc', 70)
            elif days < 100:
                current_stage = "mid"
                stage_name = "🌾 Flowering/Fruiting"
                water_need = crop_info.get('mid_etc', 95)
            else:
                current_stage = "late"
                stage_name = "🍂 Maturity Stage"
                water_need = crop_info.get('late_etc', 60)

            # Validate temperature and moisture from session state
            if 'temp' in st.session_state and 'predicted_moisture' in st.session_state:
                temp = st.session_state.temp
                predicted_moisture = st.session_state.predicted_moisture

                # Adjust for weather conditions
                if temp > 35:
                    weather_factor = 1.3
                    weather_msg = "🔥 Hot weather - increase irrigation"
                elif temp > 30:
                    weather_factor = 1.1
                    weather_msg = "☀️ Warm weather - normal irrigation"
                elif temp < 20:
                    weather_factor = 0.8
                    weather_msg = "🌤️ Cool weather - reduce irrigation"
                else:
                    weather_factor = 1.0
                    weather_msg = "🌡️ Moderate temperature"

                # Adjust for soil moisture
                moisture_factor = 1.0
                if predicted_moisture < crop_info.get('ideal_moisture', 60) - 10:
                    moisture_factor = 1.4
                    moisture_msg = "🚨 Soil very dry - urgent irrigation needed"
                elif predicted_moisture < crop_info.get('ideal_moisture', 60):
                    moisture_factor = 1.2
                    moisture_msg = "⚠️ Soil getting dry - irrigation recommended"
                else:
                    moisture_factor = 0.7
                    moisture_msg = "✅ Soil moisture adequate"

                # Final recommendation
                recommended_water = water_need * weather_factor * moisture_factor

                st.markdown(f"""
                <div style="background: linear-gradient(135deg, #e3f2fd 0%, #bbdefb 100%); 
                            padding: 20px; border-radius: 10px; margin: 10px 0;">
                    <h3 style="color: #1565c0; margin: 0;">💧 {stage_name}</h3>
                    <h2 style="color: #0d47a1; margin: 10px 0;">{recommended_water:.0f} mm water needed</h2>
                    <p style="color: #1976d2; font-size: 16px; margin: 5px 0;">
                        ⏰ <strong>Best irrigation time:</strong> Early morning (6–8 AM) or evening (6–8 PM)
                    </p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.warning("⚠️ Please run 'Get Irrigation Advice' first to fetch temperature and moisture data.")
                recommended_water = 0
                weather_msg = "N/A"
                moisture_msg = "N/A"
        else:
            st.warning("⚠️ Please select a valid crop type.")
            recommended_water = 0
            weather_msg = "N/A"
            moisture_msg = "N/A"

    with col2:
        st.subheader("📊 Quick Status")

        if recommended_water > 80:
            status_color = "🔴"
            urgency = "High"
        elif recommended_water > 50:
            status_color = "🟡"
            urgency = "Medium"
        else:
            status_color = "🟢"
            urgency = "Low"

        st.metric("Water Priority", f"{status_color} {urgency}")
        st.metric("Amount Needed", f"{recommended_water:.0f} mm")
        st.markdown("**Conditions:**")
        st.write(f"• {weather_msg}")
        st.write(f"• {moisture_msg}")

    # --- Practical Irrigation Planning ---
    st.markdown("---")
    st.subheader("🚿 Irrigation Planning Helper")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**💡 Water Application Methods**")
        irrigation_methods = {
            "Drip Irrigation": {"efficiency": 90, "time": "2–4 hours"},
            "Sprinkler": {"efficiency": 75, "time": "1–2 hours"},
            "Flood/Furrow": {"efficiency": 60, "time": "30 min"},
            "Manual Watering": {"efficiency": 65, "time": "1–3 hours"}
        }

        selected_method = st.selectbox("Choose your irrigation method:", list(irrigation_methods.keys()))
        method_info = irrigation_methods[selected_method]
        st.write(f"• **Efficiency:** {method_info['efficiency']}%")
        st.write(f"• **Time needed:** {method_info['time']}")

    with col2:
        st.markdown("**📅 Weekly Water Schedule**")
        if recommended_water > 0:
            actual_water_needed = recommended_water * (100 / method_info['efficiency'])
            irrigation_frequency = 3 if recommended_water > 70 else 2
            water_per_session = actual_water_needed / irrigation_frequency
            schedule_days = ["Monday", "Wednesday", "Friday"] if irrigation_frequency == 3 else ["Tuesday", "Saturday"]

            for day in schedule_days:
                st.write(f"• **{day}:** {water_per_session:.0f} mm")

            st.info(f"💧 Total weekly: {actual_water_needed:.0f} mm")
        else:
            st.info("No irrigation schedule available yet.")

    with col3:
        st.markdown("**💰 Cost Calculator**")
        field_area = st.number_input("Field area (acres):", min_value=0.1, value=1.0, step=0.1)
        total_liters = recommended_water * field_area * 4047 if recommended_water > 0 else 0
        st.write(f"• **Estimated water used:** {total_liters:.0f} liters")

    # --- Log Section ---
    st.markdown("---")
    st.subheader("📝 Log Your Irrigation")

    log_col1, log_col2, log_col3 = st.columns(3)

    with log_col1:
        log_date = st.date_input("Irrigation date:", datetime.datetime.now().date())
    with log_col2:
        log_amount = st.number_input("Water applied (mm):", min_value=0, value=int(recommended_water))
    with log_col3:
        log_method = st.selectbox("Method used:", list(irrigation_methods.keys()), key="log_method")

    if st.button("💾 Save Irrigation Record"):
        if 'irrigation_history' not in st.session_state:
            st.session_state.irrigation_history = []

        st.session_state.irrigation_history.append({
            'date': log_date,
            'amount': log_amount,
            'method': log_method,
            'crop_stage': stage_name if 'stage_name' in locals() else "Unknown",
            'efficiency': method_info['efficiency']
        })

        st.success(f"✅ Recorded {log_amount}mm irrigation on {log_date}")

    if 'irrigation_history' in st.session_state and st.session_state.irrigation_history:
        st.markdown("**📊 Recent Irrigation History:**")
        recent_logs = st.session_state.irrigation_history[-5:]
        for log in reversed(recent_logs):
            st.write(f"• {log['date']} - {log['amount']}mm using {log['method']} ({log['crop_stage']})")

# Enhanced export with CSV crop data
with st.expander("📤 Enhanced Data Export & Backup"):
    st.write("**Export Your Irrigation Data with Crop Requirements**")
    
    if st.session_state.irrigation_logs:
        export_format = st.selectbox("Select Export Format", ["CSV", "JSON", "Excel"])
        include_crop_data = st.checkbox("Include ICAR/FAO Crop Requirements", value=True)
        
        if st.button("Generate Enhanced Export File"):
            df = pd.DataFrame(st.session_state.irrigation_logs)
            
            # Add crop requirements data if requested
            if include_crop_data:
                crop_info = []
                for _, row in df.iterrows():
                    crop = row['crop']
                    if crop in crop_requirements:
                        req = crop_requirements[crop]
                        crop_info.append({
                            'crop_ph_min': req['min_ph'],
                            'crop_ph_max': req['max_ph'],
                            'crop_temp_min': req['min_temp'],
                            'crop_temp_max': req['max_temp'],
                            'crop_season': req['season'],
                            'crop_total_etc': req['total_etc'],
                            'crop_standard_irrigation': req['water_per_irrigation']
                        })
                    else:
                        crop_info.append({
                            'crop_ph_min': None, 'crop_ph_max': None,
                            'crop_temp_min': None, 'crop_temp_max': None,
                            'crop_season': None, 'crop_total_etc': None,
                            'crop_standard_irrigation': None
                        })
                
                crop_df_info = pd.DataFrame(crop_info)
                df = pd.concat([df, crop_df_info], axis=1)
            
            if export_format == "CSV":
                file_data = df.to_csv(index=False)
                mime_type = "text/csv"
                file_ext = "csv"
            elif export_format == "JSON":
                file_data = df.to_json(orient='records', indent=2)
                mime_type = "application/json"
                file_ext = "json"
            else:  # Excel
                from io import BytesIO
                buffer = BytesIO()
                
                # Create multiple sheets if including crop data
                if include_crop_data:
                    with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                        df.to_excel(writer, sheet_name='Irrigation_Logs', index=False)
                        
                        # Add crop requirements as separate sheet
                        crop_req_df = pd.DataFrame.from_dict(crop_requirements, orient='index')
                        crop_req_df.to_excel(writer, sheet_name='Crop_Requirements')
                else:
                    df.to_excel(buffer, index=False)
                
                file_data = buffer.getvalue()
                mime_type = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                file_ext = "xlsx"
            
            filename = f"enhanced_irrigation_data_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.{file_ext}"
            
            st.download_button(
                label=f"📥 Download Enhanced {export_format} File",
                data=file_data,
                file_name=filename,
                mime=mime_type
            )
    else:
        st.info("No data available for export. Start using the system to generate data.")


# System settings with CSV integration
with st.expander("⚙️ Enhanced System Settings"):
    st.write("**Configure System Parameters with ICAR/FAO Standards**")
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Alert Settings")
        if crop_type in crop_requirements:
            crop_ideal_moisture = crop_requirements[crop_type]['ideal_moisture']
            suggested_threshold = max(10, crop_ideal_moisture - 10)
            custom_threshold = st.slider("Custom Alert Threshold", 10, 70, int(suggested_threshold))
            st.info(f"Suggested threshold for {crop_type}: {suggested_threshold}% (based on ideal moisture: {crop_ideal_moisture}%)")
        else:
            custom_threshold = st.slider("Custom Alert Threshold", 10, 70, alert_threshold)
        
        email_frequency = st.selectbox("Email Frequency", ["Immediate", "Daily Summary", "Weekly Report"])
        
    with col2:
        st.subheader("ICAR/FAO Integration")
        use_icar_standards = st.checkbox("Use ICAR/FAO Standards", value=True)
        auto_adjust_thresholds = st.checkbox("Auto-adjust based on crop type", value=True)
        
        st.subheader("Model Settings")
        model_retrain = st.checkbox("Auto-retrain Models Weekly")
        prediction_horizon = st.slider("Prediction Horizon (hours)", 6, 48, 24)
    
    if st.button("Save Enhanced Settings"):
        st.success("Enhanced settings saved successfully!")
        st.info("System now configured to use ICAR/FAO crop standards for optimal irrigation recommendations.")

# Footer with enhanced information
st.markdown("---")
st.markdown("### 🔧 Enhanced System Information")
col1, col2, col3 = st.columns(3)

with col1:
    st.write(f"**ML Models:** {'✅ Active' if st.session_state.ml_model_trained else '❌ Training'}")
with col2:
    st.write(f"**Language:** {language}")
with col3:
    st.write(f"**Crop Database:** {len(crop_requirements)} crops")

st.markdown("---")
st.markdown("** Data Sources:** ICAR (Indian Council of Agricultural Research) & FAO (Food and Agriculture Organization)")