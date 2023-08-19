import base64
from io import BytesIO
import pandas as pd
import pickle
import streamlit as st
from gnewsclient import gnewsclient
from PIL import Image
import os

import cv2
import numpy as np
from keras.models import load_model


def preprocess_image1(img_path, img_size=(128, 128)):
    img = cv2.imread(img_path)
    if img is not None:
        img = cv2.resize(img, img_size)
        img = img.astype(np.float32) / 255.0
        return img
    return None


def make_prediction1(model, img):
    img = np.expand_dims(img, axis=0)
    prediction = model.predict(img)
    return prediction[0][0]

# Load the trained model
model = load_model('MALARIA2.h5')

def final_malaria_prediction(blood_sample, model):

    # Example of using the model for prediction
    image_path = blood_sample
    image = preprocess_image1(image_path)
    if image is not None:
        prediction = make_prediction1(model, image)
        if prediction[0][0] >= 0.5:
            result = 'Malaria Detected'
        else:
            result = 'No Malaria Detected'
    else:
        st.write("Error loading the image.")


def detect_malaria_cells(uploaded_image, model):

    # Load and preprocess the uploaded image
    img = cv2.imread(uploaded_image)
    img = cv2.resize(img, (128, 128))  # Resize to match the model's input size
    img = img / 255.0  # Normalize pixel values

    # Perform prediction using the model
    prediction = model.predict(np.expand_dims(img, axis=0))

    # Reverse the class labels: 0 -> 'Malaria Detected', 1 -> 'No Malaria Detected'
    if prediction[0][0] >= 0.5:
        result = 'No Malaria Detected'
    else:
        result = 'Malaria Detected'
    return result

def image_to_base64(image):
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()

# Load the saved model
#with open("logistic_regression_model.pkl", "rb") as file1:
#    lg_model = pickle.load(file1)


import pandas as pd
import pickle
from scipy.stats import mode

# Load the saved models and encoder using pickle
with open("svm_model.pkl", "rb") as f:
    svm_model = pickle.load(f)

with open("nb_model.pkl", "rb") as f:
    nb_model = pickle.load(f)

with open("rf_model.pkl", "rb") as f:
    rf_model = pickle.load(f)

with open("encoder.pkl", "rb") as f:
    encoder = pickle.load(f)


# Function to preprocess the input data
def preprocess_input_42(input_data):
    # Preprocessing steps
    return input_data

# Create a dictionary that maps disease numbers to disease names
disease_mapping = {
    2: "Drug Reaction",
    3: "Malaria",
    4: "Allergy",
    5: "Hypothyroidism",
    6: "Psoriasis",
    7: "GERD",
    8: "Chronic Cholestasis",
    9: "Hepatitis A",
    10: "Osteoarthritis",
    11: "(Vertigo) Paroxysmal Positional Vertigo",
    12: "Hypoglycemia",
    13: "Acne",
    14: "Diabetes",
    15: "Impetigo",
    16: "Hypertension",
    17: "Peptic Ulcer Disease",
    18: "Dimorphic Hemorrhoids (Piles)",
    19: "Common Cold",
    20: "Chickenpox",
    21: "Cervical Spondylosis",
    22: "Hyperthyroidism",
    23: "Urinary Tract Infection",
    24: "Varicose Veins",
    25: "AIDS",
    26: "Paralysis (Brain Hemorrhage)",
    27: "Typhoid",
    28: "Hepatitis B",
    29: "Fungal Infection",
    30: "Hepatitis C",
    31: "Migraine",
    32: "Bronchial Asthma",
    33: "Alcoholic Hepatitis",
    34: "Jaundice",
    35: "Hepatitis E",
    36: "Dengue",
    37: "Hepatitis D",
    38: "Heart Attack",
    39: "Pneumonia",
    40: "Arthritis",
    41: "Gastroenteritis",
    42: "Tuberculosis"
}

# Function to make disease predictions

from scipy.stats import mode

def predict_disease(input_data, encoder):
    # Preprocess the input data
    input_data = preprocess_input_42(input_data)

    # Make predictions using the loaded models
    svm_prediction = svm_model.predict(input_data)
    nb_prediction = nb_model.predict(input_data)
    rf_prediction = rf_model.predict(input_data)

    # Combine the predictions (use mode, voting, or any other desired method)
    final_prediction = mode([svm_prediction, nb_prediction, rf_prediction]).mode[0][0]

    # Convert the numerical prediction to disease name using the label encoder
    disease_name = encoder.inverse_transform([final_prediction])[0]

    return disease_name

# Helper function to save uploaded image and return its path
def save_uploaded_image(uploaded_image):
    try:
        # Create a temporary directory if it doesn't exist
        temp_dir = "temp_images"
        os.makedirs(temp_dir, exist_ok=True)

        # Save the uploaded image to the temporary directory
        img_path = os.path.join(temp_dir, uploaded_image.name)
        with open(img_path, "wb") as f:
            f.write(uploaded_image.getbuffer())

        return img_path
    except Exception as e:
        st.error(f"Error saving image: {e}")
        return None


def malaria_detection_page():
    # Take input from the user using text_area widget
    with st.form("malaria_detection_form"):
        st.subheader("Enter Detection Details")
        blood_sample = st.file_uploader("Upload Blood Sample Image", type=["png", "jpg", "jpeg"])
        detect_button = st.form_submit_button("Detect Malaria Cells")

    # Perform malaria cell detection logic here
    if detect_button and blood_sample:
        st.subheader("Blood Sample Image Preview")
        st.image(blood_sample, use_column_width=True)
        st.subheader("Malaria Cell Detection Results")

        # Convert the uploaded file to an image and get the path
        image_path = save_uploaded_image(blood_sample)
        if image_path is not None:
            detection_result = detect_malaria_cells(image_path, model)  # Call detect_malaria_cells function
            st.write(detection_result)  # Display the detection result
    elif detect_button and not blood_sample:
        st.warning("Please upload a blood sample image.")


def disease_42_prediction():

    #st.markdown(content, unsafe_allow_html=True)
    symptoms = {
            'itching': 1,'skin_rash': 2,'nodal_skin_eruptions': 3,'continuous_sneezing': 4,'shivering': 5,'chills': 6,
            'joint_pain': 7,'stomach_pain': 8,'acidity': 9,'ulcers_on_tongue': 10,'muscle_wasting': 11,
            'vomiting': 12,'burning_micturition': 13,'spotting_ urination': 14,'fatigue': 15,'weight_gain': 16,
            'anxiety': 17,'cold_hands_and_feets': 18,'mood_swings': 19,'weight_loss': 20,'restlessness': 21,
            'lethargy': 22,'patches_in_throat': 23,'irregular_sugar_level': 24,'cough': 25,'high_fever': 26,
            'sunken_eyes': 27,'breathlessness': 28,'sweating': 29,'dehydration': 30,'indigestion': 31,'headache': 32,
            'yellowish_skin': 33,'dark_urine': 34,'nausea': 35,'loss_of_appetite': 36,'pain_behind_the_eyes': 37,
            'back_pain': 38,'constipation': 39,'abdominal_pain': 40,'diarrhoea': 41,'mild_fever': 42,'yellow_urine': 43,
            'yellowing_of_eyes': 44,'acute_liver_failure': 45,'fluid_overload': 46,'swelling_of_stomach': 47,'swelled_lymph_nodes': 48,
            'malaise': 49,'blurred_and_distorted_vision': 50,'phlegm': 51,'throat_irritation': 52,'redness_of_eyes': 53,
            'sinus_pressure': 54,'runny_nose': 55,'congestion': 56,'chest_pain': 57,'weakness_in_limbs': 58,
            'fast_heart_rate': 59,'pain_during_bowel_movements': 60,'pain_in_anal_region': 61,'bloody_stool': 62,
            'irritation_in_anus': 63,'neck_pain': 64,'dizziness': 65,'cramps': 66,'bruising': 67,'obesity': 68,
            'swollen_legs': 69,'swollen_blood_vessels': 70,'puffy_face_and_eyes': 71,'enlarged_thyroid': 72,
            'brittle_nails': 73,'swollen_extremeties': 74,'excessive_hunger': 75,'extra_marital_contacts': 76,
            'drying_and_tingling_lips': 77,'slurred_speech': 78,'knee_pain': 79,'hip_joint_pain': 80,'muscle_weakness': 81,
            'stiff_neck': 82,'swelling_joints': 83,'movement_stiffness': 84,'spinning_movements': 85,'loss_of_balance': 86,
            'unsteadiness': 87,'weakness_of_one_body_side': 88,'loss_of_smell': 89,'bladder_discomfort': 90,
            'foul_smell_of urine': 91,'continuous_feel_of_urine': 92,'passage_of_gases': 93,'internal_itching': 94,
            'toxic_look_(typhos)': 95,'depression': 96,'irritability': 97,'muscle_pain': 98,'altered_sensorium': 99,
            'red_spots_over_body': 100,'belly_pain': 101,'abnormal_menstruation': 102,'dischromic _patches': 103,
            'watering_from_eyes': 104,'increased_appetite': 105,'polyuria': 106,'family_history': 107,'mucoid_sputum': 108,
            'rusty_sputum': 109,'lack_of_concentration': 110,'visual_disturbances': 111,'receiving_blood_transfusion': 112,
            'receiving_unsterile_injections': 113,'coma': 114,'stomach_bleeding': 115,'distention_of_abdomen': 116,
            'history_of_alcohol_consumption': 117,'fluid_overload.1': 118,'blood_in_sputum': 119,'prominent_veins_on_calf': 120,
            'palpitations': 121,'painful_walking': 122,'pus_filled_pimples': 123,'blackheads': 124,'scurring': 125,
            'skin_peeling': 126,'silver_like_dusting': 127,'small_dents_in_nails': 128,'inflammatory_nails': 129,'blister': 130,
            'red_sore_around_nose': 131,'yellow_crust_ooze': 132
        }


    with st.form(key="prediction_form1"):
        st.subheader("Enter Symptoms")
        symptom_inputs = {}
        for symptom, code in symptoms.items():
            symptom_inputs[symptom] = int(st.checkbox(symptom, value=False))

        # Create the "Predict" button
        predict_button = st.form_submit_button("Predict")

        # Perform the prediction only if the "Predict" button is pressed
        if predict_button:
            # Prepare the input data for prediction
            input_data = pd.DataFrame(symptom_inputs, index=[0])
            # Perform your prediction logic here (replace this with your actual prediction function)
            predictions = predict_disease(input_data,encoder)#.iloc[0].values.reshape(1, -1))

            st.write(predictions)


def fetch_and_display_news(selected_topics):
    location = 'United States'
    max_results = 1

    for topic in selected_topics:
        client = gnewsclient.NewsClient(language='english', location=location, topic=topic, max_results=max_results)
        news_list = client.get_news()

        st.header(f"News for {topic}:")
        for item in news_list:
            st.write("Title:", item['title'])
            st.write("Link:", item['link'])
            st.write("")


def main():
    # Set page layout
    #st.set_page_config(layout="wide")

    # Load and display the image
    img1 = Image.open("Pulse Prophet.png")
    st.image(img1, width=400)

    # Open the image
    image_path = "Widget.png"
    image = Image.open(image_path)

    # Convert the image to RGBA mode (if not already)
    image = image.convert("RGBA")

    # Adjust the transparency level (alpha channel)
    transparency = 0.5  # Set the desired transparency level (0.0 to 1.0)
    image_with_transparency = image.copy()
    alpha = image_with_transparency.split()[3]  # Get the alpha channel
    alpha = alpha.point(lambda p: p * transparency)  # Adjust the alpha values
    image_with_transparency.putalpha(alpha)  # Apply the new alpha channel

    # Set the desired coordinates for the image
    image_top = 0  # Vertical position from the top
    image_left = -10  # Horizontal position from the left



    # Position the transparent image at the specified coordinates
    st.markdown(
    f'<div style="position: absolute; top: {image_top}px; left: {image_left}px;">'
    f'<img src="data:image/png;base64,{image_to_base64(image_with_transparency)}" alt="Image" style="width: 300px;">'
    '</div>',
    unsafe_allow_html=True)


    # Define the options for the selection box
    options = ["Home", "42 Disease Prediction", "Malaria Cells Detection", "Latest Health Articles and News"]

    # Create a selection box in the sidebar
    selected_option = st.sidebar.selectbox("Select an option:", options)

    # Determine which option was selected and display the corresponding content
    if selected_option == "Home":
        st.markdown(get_home_page_content(), unsafe_allow_html=True)
    elif selected_option == "42 Disease Prediction":
        st.markdown(get_disease_prediction_content(), unsafe_allow_html=True)
        disease_42_prediction()
    elif selected_option == "Malaria Cells Detection":
        st.markdown(get_malaria_detection_content(), unsafe_allow_html=True)
        malaria_detection_page()
    elif selected_option == "Latest Health Articles and News":
        st.markdown(get_health_articles_content(), unsafe_allow_html=True)
        health_topic=['Science','Health','Technology']
        fetch_and_display_news(health_topic)

def get_home_page_content():
    return """
    # Home Page
    Welcome to the Home page of the Pulse-Prophet App.
    This page provides general information about the app and its features.
    Please select a specific option from the sidebar to explore different functionalities.

    1. **42 Types Disease Prediction**
    Predict the presence of various diseases based on symptoms and risk factors.

    2. **Malaria Cell Detection**
    For medical professionals only. Analyze cell images to detect malaria cells.

    4. **General Health Information**
    Access the latest health, science and technology articles to stay informed about trends and breakthroughs.
    
    
    **-App by Auhona Chakraborty**
    """

def get_disease_prediction_content():
    return """
    # 42 Disease Prediction
    This page allows you to predict the 42 types of diseases based on symptoms and recommend precautions need to be taken.
    Please select the symptoms to see prediction and click on "Predict" button to get the prediction results.
    """

def get_malaria_detection_content():
    return """
    # Malaria Cells Detection
    This page is specifically designed for medical professionals to detect malaria cells from blood samples.
    Please upload the blood sample image and click the "Detect" button to analyze and identify malaria cells.
    """

def get_health_articles_content():
    return """
    # Latest Health Articles and News
    This page provides access to the latest health articles and news.
    Stay updated with the latest health information, medical breakthroughs, and healthcare trends.
    """


if __name__ == "__main__":
    main()
