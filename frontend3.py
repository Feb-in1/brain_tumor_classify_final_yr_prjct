import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go
from tensorflow.keras.models import load_model
from io import BytesIO
from PIL import Image
import random
import cv2
import matplotlib.cm as cm
import base64
from fpdf import FPDF
from io import BytesIO
import pandas as pd
import numpy as np
from datetime import datetime
import os


# GPU Configuration
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

# Force CPU usage if needed
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# Model Prediction Labels
labels = ['glioma_tumor', 'meningioma_tumor', 'no_tumor', 'pituitary_tumor']

def model_prediction(image_data):
    """
    Process an MRI image and return tumor prediction results
    Returns: predicted class, confidence score, GradCAM heatmap, and original image
    """
    try:
        # Load the model
        model = load_model('bestresnet4.h5') 
        
        # Load and preprocess the image
        img = Image.open(BytesIO(image_data))
        img = img.resize((200, 200))
        
        # Convert to numpy array and preprocess
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        img_array = img_array / 255.0  # Normalize pixel values
        
        # Save original image for display
        original_img = img
        
        # Add batch dimension
        input_tensor = np.expand_dims(img_array, axis=0)
        
        # Make prediction
        predictions = model.predict(input_tensor)
        predicted_class_index = np.argmax(predictions[0])
        predicted_class = labels[predicted_class_index]
        confidence = round(100 * np.max(predictions[0]), 2)
        
        # Generate GradCAM visualization
        gradcam_heatmap = get_gradcam(model, input_tensor, predicted_class_index)
        
        return predicted_class, confidence, gradcam_heatmap, original_img
    
    except Exception as e:
        import traceback
        st.error(f"Prediction error: {str(e)}")
        st.error(traceback.format_exc())
        return None, None, None, None

def get_gradcam(model, img_array, predicted_class_index):
    """
    Completely redesigned GradCAM implementation using TensorFlow's gradient API
    """
    import tensorflow as tf
    import numpy as np
    import cv2
    
    try:
        # Ensure we have a batch dimension
        if len(img_array.shape) == 3:
            img_array = np.expand_dims(img_array, axis=0)
        
        # Convert to TensorFlow tensor
        img_tensor = tf.cast(img_array, tf.float32)
        
        # We'll use the final convolutional layer before the classification head
        # This requires knowing your model architecture - let's try a more robust approach
        
        # Find all convolutional layers
        conv_layers = []
        for layer in model.layers:
            # Check if it's a convolutional layer by name or class
            if isinstance(layer, tf.keras.layers.Conv2D) or 'conv' in layer.name.lower():
                conv_layers.append(layer.name)
        
        if not conv_layers:
            print("No convolutional layers found in the model")
            return np.zeros((img_array.shape[1], img_array.shape[2], 3), dtype=np.uint8)
        
        # Use the last convolutional layer
        target_layer_name = conv_layers[-1]
        
        # Create a GradCAM class from scratch
        # Get the output of the last conv layer
        target_layer = model.get_layer(target_layer_name)
        
        # Create a model that goes from input to both target layer and prediction
        gradcam_model = tf.keras.models.Model(
            inputs=[model.inputs],
            outputs=[target_layer.output, model.output]
        )
        
        # Prediction label - get from labels list
        labels = ['glioma_tumor', 'meningioma_tumor', 'no_tumor', 'pituitary_tumor']
        label = labels[predicted_class_index]

        # Set up the gradient tape
        with tf.GradientTape() as tape:
            conv_output, predictions = gradcam_model(img_tensor)
            
            # For "no_tumor" cases, approach differently
            if label == "no_tumor":
                # Look at what makes it NOT the other classes
                # Create a loss based on the average of other class scores
                loss = tf.reduce_mean([predictions[0][i] for i in range(len(labels)) if i != predicted_class_index])
                # We'll invert this gradient later
                inverted = True
            else:
                # For actual tumors, use that class's score
                loss = predictions[0][predicted_class_index]
                inverted = False
                
        # Calculate gradients
        grads = tape.gradient(loss, conv_output)
        
        # Invert gradients for no_tumor cases
        if inverted:
            grads = -grads
        
        # Global average pooling
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        
        # Get the output from the last convolutional layer
        last_conv_layer_output = conv_output[0]
        
        # Multiply each channel by its importance
        heatmap = tf.reduce_sum(
            tf.multiply(pooled_grads, last_conv_layer_output),
            axis=-1
        )
        
        # ReLU to focus on positive contributions
        heatmap = tf.maximum(heatmap, 0) / (tf.reduce_max(heatmap) + tf.keras.backend.epsilon())
        heatmap = heatmap.numpy()
        
        # Resize to match original image size
        heatmap = cv2.resize(heatmap, (img_array.shape[2], img_array.shape[1]))
        
        # Apply Gaussian blur for smoother visualization
        heatmap = cv2.GaussianBlur(heatmap, (5, 5), 0)
        
        # Normalize between 0-1 if there are non-zero values
        if np.max(heatmap) > 0:
            heatmap = heatmap / np.max(heatmap)
            
        # Special treatment for no_tumor cases to make visualization more intuitive
        if label == "no_tumor":
            # For no_tumor, we show a more diffuse, lower-intensity heatmap
            heatmap = heatmap * 0.7  # Reduce intensity
        
        # Convert to RGB heatmap
        heatmap_colored = np.uint8(255 * heatmap)
        heatmap_colored = cv2.applyColorMap(heatmap_colored, cv2.COLORMAP_JET)
        
        return heatmap_colored
        
    except Exception as e:
        import traceback
        print(f"Error in GradCAM generation: {str(e)}")
        print(traceback.format_exc())
        # Return a blank red image on error for debugging
        blank = np.zeros((img_array.shape[1], img_array.shape[2], 3), dtype=np.uint8)
        blank[:,:,2] = 128  # Some red to indicate error
        return blank

def overlay_gradcam(original_img, heatmap, alpha=0.6):
    """
    Improved overlay function with better blending
    """
    import numpy as np
    import cv2
    from PIL import Image
    
    try:
        # Convert PIL Image to numpy array if needed
        if isinstance(original_img, Image.Image):
            original_np = np.array(original_img)
        else:
            original_np = original_img.copy()
        
        # Ensure original image is RGB
        if len(original_np.shape) == 2:  # Grayscale
            original_rgb = cv2.cvtColor(original_np, cv2.COLOR_GRAY2RGB)
        elif original_np.shape[2] == 1:  # Single channel
            original_rgb = cv2.cvtColor(original_np, cv2.COLOR_GRAY2RGB)
        elif original_np.shape[2] == 4:  # RGBA
            original_rgb = cv2.cvtColor(original_np, cv2.COLOR_RGBA2RGB)
        else:
            original_rgb = original_np
            
        # Resize heatmap to match original image dimensions
        heatmap_resized = cv2.resize(heatmap, (original_rgb.shape[1], original_rgb.shape[0]))
        
        # Use addWeighted for proper alpha blending
        superimposed = cv2.addWeighted(
            original_rgb, 
            1.0 - alpha,  # Original image weight
            heatmap_resized,
            alpha,        # Heatmap weight
            0             # Scalar added to each sum
        )
        
        # Return as PIL image
        return Image.fromarray(cv2.cvtColor(superimposed, cv2.COLOR_BGR2RGB))
        
    except Exception as e:
        import traceback
        print(f"Error generating overlay: {str(e)}")
        print(traceback.format_exc())
        # Return original image if overlay fails
        if isinstance(original_img, Image.Image):
            return original_img
        else:
            return Image.fromarray(original_np)
        

def create_doctor_database():
    return [
        {
            "name": "Dr. Dilip Panikar",
            "specialty": "Neurosurgery",
            "expertise": ["glioma_tumor", "meningioma_tumor", "pituitary_tumor"],
            "hospital": "Amrita Institute of Medical Sciences",
            "city": "Kochi",
            "state": "Kerala",
            "experience": 25,
            "contact": "+91-484-2801234",
            "email": "appointments@aims.amrita.edu",
            "appointment_link": "https://www.amritahospitals.org/appointments",
            "description": "Specialized in complex brain tumors and skull base surgery with over 25 years of experience."
        },
        {
            "name": "Dr. P. Radhakrishnan",
            "specialty": "Neurosurgery",
            "expertise": ["glioma_tumor", "meningioma_tumor", "pituitary_tumor"],
            "hospital": "Sree Chitra Tirunal Institute for Medical Sciences",
            "city": "Thiruvananthapuram",
            "state": "Kerala",
            "experience": 30,
            "contact": "+91-471-2524690",
            "email": "neurosurgery@sctimst.ac.in",
            "appointment_link": "https://www.sctimst.ac.in/",
            "description": "Renowned for treating complex neuro-oncology cases and pediatric brain tumors."
        },
        {
            "name": "Dr. Mathew Abraham",
            "specialty": "Neurosurgery",
            "expertise": ["glioma_tumor", "meningioma_tumor"],
            "hospital": "KIMS Hospital",
            "city": "Thiruvananthapuram",
            "state": "Kerala",
            "experience": 20,
            "contact": "+91-471-2447575",
            "email": "neurosurgery@kimsglobal.com",
            "appointment_link": "https://www.kimsglobal.com/appointment",
            "description": "Expert in gliomas, meningiomas, and awake craniotomy procedures."
        },
        {
            "name": "Dr. K.V. Viswanathan",
            "specialty": "Neurosurgery",
            "expertise": ["pituitary_tumor", "meningioma_tumor"],
            "hospital": "Aster Medcity",
            "city": "Kochi",
            "state": "Kerala",
            "experience": 25,
            "contact": "+91-484-6699999",
            "email": "appointments@asterhospital.com",
            "appointment_link": "https://www.astermedcity.com/",
            "description": "Specialized in minimally invasive neurosurgery and pituitary tumors."
        },
        # Other Kerala doctors
        {
            "name": "Dr. Hari Subramoniam",
            "specialty": "Neurosurgery",
            "expertise": ["glioma_tumor", "meningioma_tumor", "pituitary_tumor"],
            "hospital": "Kerala Institute of Medical Sciences (KIMS)",
            "city": "Thiruvananthapuram",
            "state": "Kerala",
            "experience": 18,
            "contact": "+91-471-2447575",
            "email": "appointments@kimshealth.org",
            "appointment_link": "https://www.kimshealth.org/",
            "description": "Expert in complex brain tumors and stereotactic surgery."
        },
        # All-India renowned doctors
        {
            "name": "Dr. Basant K. Misra",
            "specialty": "Neurosurgery",
            "expertise": ["glioma_tumor", "meningioma_tumor", "pituitary_tumor"],
            "hospital": "P.D. Hinduja National Hospital",
            "city": "Mumbai",
            "state": "Maharashtra",
            "experience": 35,
            "contact": "+91-22-24452222",
            "email": "contact@hindujahospital.com",
            "appointment_link": "https://www.hindujahospital.com/",
            "description": "Padma Shri awardee and globally renowned neurosurgeon specialized in complex brain tumors."
        },
        {
            "name": "Dr. Suresh Sankhla",
            "specialty": "Neurosurgery",
            "expertise": ["glioma_tumor", "meningioma_tumor", "pituitary_tumor"],
            "hospital": "Global Hospital",
            "city": "Mumbai",
            "state": "Maharashtra",
            "experience": 30,
            "contact": "+91-22-67676767",
            "email": "appointments@globalhospitalsindia.com",
            "appointment_link": "https://www.globalhospitalsindia.com/",
            "description": "Pioneer in keyhole and minimally invasive neurosurgery for brain tumors."
        },
        {
            "name": "Dr. Vipul Gupta",
            "specialty": "Neuro-Intervention",
            "expertise": ["glioma_tumor", "meningioma_tumor"],
            "hospital": "Artemis Hospital",
            "city": "Gurugram",
            "state": "Haryana",
            "experience": 20,
            "contact": "+91-124-4511111",
            "email": "info@artemishospitals.com",
            "appointment_link": "https://www.artemishospitals.com/",
            "description": "Expert in neuro-interventional procedures for brain tumors."
        },
        {
            "name": "Dr. Ashish Suri",
            "specialty": "Neurosurgery",
            "expertise": ["glioma_tumor", "pituitary_tumor"],
            "hospital": "All India Institute of Medical Sciences (AIIMS)",
            "city": "New Delhi",
            "state": "Delhi",
            "experience": 25,
            "contact": "+91-11-26588500",
            "email": "neurosurgery@aiims.edu",
            "appointment_link": "https://www.aiims.edu/",
            "description": "Renowned for complex brain tumor surgeries and neuro-oncology research."
        },
        {
            "name": "Dr. V.P. Singh",
            "specialty": "Neurosurgery",
            "expertise": ["glioma_tumor", "meningioma_tumor", "pituitary_tumor"],
            "hospital": "Medanta - The Medicity",
            "city": "Gurugram",
            "state": "Haryana",
            "experience": 28,
            "contact": "+91-124-4141414",
            "email": "info@medanta.org",
            "appointment_link": "https://www.medanta.org/",
            "description": "Specializes in complex brain tumors and skull base surgery."
        },
        {
            "name": "Dr. Rana Patir",
            "specialty": "Neurosurgery",
            "expertise": ["glioma_tumor", "meningioma_tumor", "pituitary_tumor"],
            "hospital": "Fortis Memorial Research Institute",
            "city": "Gurugram",
            "state": "Haryana",
            "experience": 30,
            "contact": "+91-124-4996222",
            "email": "info@fortishealthcare.com",
            "appointment_link": "https://www.fortishealthcare.com/",
            "description": "Pioneer in using advanced techniques for brain tumor surgery."
        },
        {
            "name": "Dr. Sanjay Behari",
            "specialty": "Neurosurgery",
            "expertise": ["glioma_tumor", "meningioma_tumor"],
            "hospital": "Sanjay Gandhi Postgraduate Institute of Medical Sciences",
            "city": "Lucknow",
            "state": "Uttar Pradesh",
            "experience": 25,
            "contact": "+91-522-2668800",
            "email": "neurosurgery@sgpgi.ac.in",
            "appointment_link": "https://www.sgpgi.ac.in/",
            "description": "Expert in surgical management of brain tumors with extensive research background."
        },
        {
            "name": "Dr. B.A. Krishna",
            "specialty": "Neurosurgery",
            "expertise": ["pituitary_tumor", "meningioma_tumor"],
            "hospital": "Apollo Hospitals",
            "city": "Chennai",
            "state": "Tamil Nadu",
            "experience": 25,
            "contact": "+91-44-28290200",
            "email": "enquiry@apollohospitals.com",
            "appointment_link": "https://www.apollohospitals.com/",
            "description": "Pioneered advanced surgical techniques for pituitary tumors."
        }
    ]

def recommend_doctors(tumor_type, location_preference="All India"):
    """
    Returns a list of doctors specialized in treating the detected tumor type
    
    Parameters:
    tumor_type (str): The detected tumor type
    location_preference (str): 'Kerala', 'All India', or specific city/state
    
    Returns:
    list: A filtered and sorted list of recommended doctors
    """
    doctors = create_doctor_database()
    matching_doctors = []
    
    for doctor in doctors:
        # Skip if the doctor doesn't have expertise in this tumor type
        if tumor_type not in doctor["expertise"]:
            continue
            
        # Filter by location preference
        if location_preference == "All India":
            matching_doctors.append(doctor)
        elif location_preference == "Kerala" and doctor["state"] == "Kerala":
            matching_doctors.append(doctor)
        elif (location_preference == doctor["city"] or 
              location_preference == doctor["state"]):
            matching_doctors.append(doctor)
    
    # Sort by experience (you could use other criteria)
    matching_doctors.sort(key=lambda x: x["experience"], reverse=True)
    
    return matching_doctors



def doctor_recommendation_section(tumor_type):
    """
    Creates the doctor recommendation section in the UI
    
    Parameters:
    tumor_type (str): The detected tumor type
    """
    st.markdown("---")
    st.subheader("👨‍⚕️ Recommended Specialists")
    
    # Location preference
    location_options = ["Kerala", "All India"]
    location_preference = st.radio(
        "Filter doctors by location:",
        location_options,
        index=0  # Default to Kerala
    )
    
    # Get doctor recommendations
    recommended_doctors = recommend_doctors(tumor_type, location_preference)
    
    if not recommended_doctors:
        st.warning(f"No specialists found for {tumor_type.replace('_', ' ')} in {location_preference}. Please consult with a general neurologist.")
        return
    
    # Display recommendations
    st.write(f"Based on the analysis, we recommend the following specialists who treat {tumor_type.replace('_', ' ')}:")
    
    # Use columns for better layout
    for doctor in recommended_doctors:
        col1, col2 = st.columns([1, 2])
        
        with col1:
            # Doctor image (using a placeholder)
            st.markdown(f"""
            <div style="width:100px; height:100px; border-radius:50%; background-color:#3498db; 
                    color:white; display:flex; align-items:center; justify-content:center; 
                    font-size:36px; margin:10px auto;">
                {doctor['name'][0]}
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown(f"<div style='text-align:center; font-weight:bold;'>{doctor['experience']} Years</div>", 
                       unsafe_allow_html=True)
            
        with col2:
            st.markdown(f"""
            <div style="background-color: #f8f9fa; padding: 15px; border-radius: 5px; 
                      border-left: 4px solid #3498db;">
                <h3 style="margin-top:0;">{doctor['name']}</h3>
                <p><strong>Specialty:</strong> {doctor['specialty']}</p>
                <p><strong>Hospital:</strong> {doctor['hospital']}, {doctor['city']}, {doctor['state']}</p>
                <p><strong>Experience:</strong> {doctor['experience']} years</p>
                <p>{doctor['description']}</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Contact and appointment section
                # <a href="tel:{doctor['contact']}" target="_blank" style="text-decoration:none;">
                #     <button style="background-color:#27ae60; color:white; padding:8px 15px; 
                #              border:none; border-radius:5px; cursor:pointer;">
                #         📞 Call
                #     </button>
                # </a>
                # <a href="mailto:{doctor['email']}" target="_blank" style="text-decoration:none;">
                #     <button style="background-color:#2980b9; color:white; padding:8px 15px; 
                #              border:none; border-radius:5px; cursor:pointer;">
                #         ✉️ Email
                #     </button>
                # </a>
            st.markdown(f"""
            <div style="display:flex; gap:10px; margin-top:10px;">
                <a href="{doctor['appointment_link']}" target="_blank" style="text-decoration:none;">
                    <button style="background-color:#8e44ad; color:white; padding:8px 15px; 
                             border:none; border-radius:5px; cursor:pointer;">
                        🗓️ Book Appointment
                    </button>
                </a>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("<hr>", unsafe_allow_html=True)
    
    # Add disclaimer
    st.markdown("""
    <div style="background-color:#f8f9fa; padding:15px; border-radius:5px; font-size:0.9em; 
              border-left:4px solid #e74c3c; margin-top:20px;">
        <strong>Disclaimer:</strong> This doctor recommendation is based on general information
        about specialists in the field. Always consult with your primary care physician before 
        making medical decisions. The final choice of doctor should be based on your specific 
        medical needs, insurance coverage, and personal preferences.
    </div>
    """, unsafe_allow_html=True)

# Helper Functions for Data

def create_medical_terminology_glossary():
    return {
        'Tumor': 'An abnormal growth of cells that can be benign or malignant.',
        'Glioma': 'A type of tumor that starts in the glial cells of the brain.',
        'Meningioma': 'A tumor that arises from the membranes surrounding the brain and spinal cord.',
        'Pituitary Tumor': 'A growth that develops in the pituitary gland at the base of the brain.',
        'Metastasis': 'The spread of cancer cells from one part of the body to another.',
        'Biopsy': 'A medical procedure to remove cells or tissues for examination.',
        'Radiation Therapy': 'A treatment that uses high-energy radiation to kill cancer cells.',
        'Chemotherapy': 'A type of cancer treatment that uses drugs to destroy cancer cells.',
        'Neuroplasticity': 'The brain\'s ability to reorganize itself by forming new neural connections.',
        'Genetic Marker': 'A specific genetic variation associated with a particular disease or condition.',
        'MRI (Magnetic Resonance Imaging)': 'A non-invasive imaging technique that uses magnetic fields and radio waves to create detailed images of organs and tissues.',
        'CT Scan': 'Computed Tomography scan that uses X-rays and computer processing to create cross-sectional images.',
        'Benign': 'Not cancerous; does not invade nearby tissue or spread to other parts of the body.',
        'Malignant': 'Cancerous; can invade nearby tissue and spread to other parts of the body.',
        'Prognosis': 'The likely outcome or course of a disease; the chance of recovery.',
        'Neurosurgeon': 'A physician who specializes in the surgical treatment of disorders of the brain, spine, and nervous system.'
    }

def create_tumor_progression_stages():
    return {
        'Glioma': [
            {'Stage': 'I', 'Description': 'Slow-growing, well-differentiated cells', 'Survival Rate': '90-95%'},
            {'Stage': 'II', 'Description': 'Slightly abnormal cells, potential for recurrence', 'Survival Rate': '70-80%'},
            {'Stage': 'III', 'Description': 'Actively reproducing abnormal cells', 'Survival Rate': '50-60%'},
            {'Stage': 'IV', 'Description': 'Highly aggressive, rapidly spreading cells', 'Survival Rate': '10-15%'}
        ],
        'Meningioma': [
            {'Stage': 'I', 'Description': 'Benign, slow-growing', 'Survival Rate': '95-100%'},
            {'Stage': 'II', 'Description': 'Atypical cells, potential for recurrence', 'Survival Rate': '80-90%'},
            {'Stage': 'III', 'Description': 'Malignant, aggressive growth', 'Survival Rate': '50-70%'}
        ],
         'Pituitary': [
            {'Stage': 'I', 'Description': 'Small, localized tumor with minimal growth', 'Survival Rate': '95-100%'},
            {'Stage': 'II', 'Description': 'Moderate-sized tumor with potential local expansion', 'Survival Rate': '85-95%'},
            {'Stage': 'III', 'Description': 'Larger tumor with potential compression of surrounding structures', 'Survival Rate': '70-85%'}
        ]
    }

def create_risk_assessment_questionnaire():
    return [
        {
            'question': 'Do you have a family history of brain tumors?',
            'type': 'radio',
            'options': ['Yes', 'No', 'Unsure']
        },
        {
            'question': 'Age Group',
            'type': 'radio',
            'options': ['0-20', '21-40', '41-60', '61+']
        },
        {
            'question': 'Have you been exposed to ionizing radiation?',
            'type': 'radio',
            'options': ['Yes', 'No', 'Unsure']
        },
        {
            'question': 'Frequency of mobile phone usage (hours per day)',
            'type': 'radio',
            'options': ['Less than 1', '1-3', '3-5', 'More than 5']
        },
        {
            'question': 'Do you have any genetic predispositions?',
            'type': 'multiselect',
            'options': ['None', 'Li-Fraumeni Syndrome', 'Neurofibromatosis', 'Other']
        },
        {
            'question': 'Have you experienced persistent headaches?',
            'type': 'radio',
            'options': ['Yes', 'No', 'Occasionally']
        },
        {
            'question': 'Have you noticed recent changes in vision or hearing?',
            'type': 'radio',
            'options': ['Yes', 'No', 'Unsure']
        }
    ]

def generate_global_tumor_statistics():
    """
    Generate global tumor statistics with multi-year data for trend analysis
    """
    # Define countries and regions
    countries = [
        'United States', 'Canada', 'United Kingdom', 'Germany', 'France',
        'China', 'India', 'Japan', 'Australia', 'Brazil', 'South Africa',
        'Russia', 'Mexico', 'Italy', 'Spain', 'South Korea'
    ]
    
    regions = [
        'North America', 'North America', 'Europe', 'Europe', 'Europe',
        'Asia', 'Asia', 'Asia', 'Oceania', 'South America', 'Africa',
        'Europe', 'North America', 'Europe', 'Europe', 'Asia'
    ]
    
    # Create multi-year data with appropriate growth and fluctuations
    data = []
    
    # Base cases for 2021
    base_cases_2021 = [
        23400, 8900, 8500, 7800, 6200,
        18900, 16500, 11300, 3500, 8300, 4200,
        9500, 5100, 5800, 5300, 7100
    ]
    
    # Mortality rates for 2021
    base_mortality_2021 = [
        33.5, 30.2, 30.0, 29.1, 30.5,
        45.2, 51.3, 35.6, 27.2, 42.1, 55.3,
        47.8, 45.9, 31.8, 31.0, 32.6
    ]
    
    # Average diagnosis age for 2021
    base_age_2021 = [
        60, 58, 62, 63, 61,
        54, 52, 59, 58, 55, 50,
        57, 53, 64, 61, 57
    ]
    
    # Years to include
    years = [2021, 2022, 2023]
    
    # Generate data for each year with realistic trends
    for year_idx, year in enumerate(years):
        for i, country in enumerate(countries):
            # Calculate cases with a general increasing trend (3-5% per year)
            # with some random fluctuation
            growth_factor = 1 + (0.03 + 0.02 * year_idx) * (1 + 0.1 * (random.random() - 0.5))
            cases = int(base_cases_2021[i] * growth_factor)
            
            # Mortality rates generally decreasing slightly (0.5-1.5% per year)
            # with some random fluctuation
            mortality_factor = 1 - (0.005 + 0.01 * year_idx) * (1 + 0.2 * (random.random() - 0.5))
            mortality = round(base_mortality_2021[i] * mortality_factor, 1)
            
            # Average age increasing slightly (0.5 years per period)
            # with some random fluctuation
            age_increase = year_idx * 0.5 * (1 + 0.2 * (random.random() - 0.5))
            avg_age = round(base_age_2021[i] + age_increase, 1)
            
            # Add to dataset
            data.append({
                'Country': country,
                'Region': regions[i],
                'Year': year,
                'New Cases': cases,
                'Mortality Rate (%)': mortality,
                'Average Age of Diagnosis': avg_age
            })
    
    return pd.DataFrame(data)

def generate_age_gender_distribution():
    # Simulated data for demonstration
    ages = list(range(0, 81, 10))
    data = []
    for age in ages:
        data.append({
            'Age Group': f'{age}-{age+9}',
            'Male Incidence': random.uniform(5, 20),
            'Female Incidence': random.uniform(4, 18)
        })
    return pd.DataFrame(data)

def generate_treatment_success_rates():
    return pd.DataFrame({
        'Treatment Method': ['Surgery', 'Radiation', 'Chemotherapy', 'Combined Therapy'],
        'Glioma Success Rate (%)': [45, 35, 30, 55],
        'Meningioma Success Rate (%)': [75, 40, 25, 80],
        'Pituitary Tumor Success Rate (%)': [85, 50, 35, 90]
    })

def create_survival_data():
    return pd.DataFrame({
        'Tumor Type': ['Glioma', 'Meningioma', 'Pituitary', 'No Tumor'],
        '2018 Survival Rate (%)': [45, 80, 85, 100],
        '2022 Survival Rate (%)': [55, 85, 90, 100],
        '2024 Survival Rate (%)': [65, 90, 95, 100]
    })

def create_risk_factors_data():
    return {
        'Genetic Factors': [
            "Family history of brain tumors",
            "Inherited genetic syndromes",
            "Specific gene mutations",
            "History of certain cancers"
        ],
        'Environmental Factors': [
            "Exposure to ionizing radiation",
            "Prolonged use of mobile phones",
            "Chemical exposure in workplace",
            "Living in industrial areas with high pollution"
        ],
        'Lifestyle Factors': [
            "Chronic stress",
            "Poor diet",
            "Lack of physical activity",
            "Smoking and alcohol consumption",
            "Inadequate sleep patterns"
        ]
    }

def create_tumor_details():
    return {
        'Glioma Tumor': {
            'Description': 'Originates from glial cells in the brain.',
            'Characteristics': [
                'Can be low-grade or high-grade',
                'Develops in cerebral hemispheres',
                'Varies in growth rate and aggressiveness',
                'May cause seizures, headaches, and cognitive changes'
            ],
            'Common Symptoms': [
                'Headaches that worsen in the morning',
                'Seizures',
                'Progressive weakness or paralysis',
                'Speech difficulties',
                'Cognitive decline'
            ],
            'Treatment Options': [
                'Surgery',
                'Radiation therapy',
                'Chemotherapy',
                'Targeted drug therapy',
                'Clinical trials'
            ]
        },
        'Meningioma Tumor': {
            'Description': 'Develops in the meninges covering the brain.',
            'Characteristics': [
                'Usually benign',
                'Slow-growing',
                'Can cause neurological symptoms',
                'More common in women than men'
            ],
            'Common Symptoms': [
                'Headaches',
                'Blurred vision',
                'Weakness in limbs',
                'Seizures',
                'Memory problems'
            ],
            'Treatment Options': [
                'Observation (for small, asymptomatic tumors)',
                'Surgery',
                'Radiation therapy',
                'Stereotactic radiosurgery'
            ]
        },
        'Pituitary Tumor': {
            'Description': 'Grows in the pituitary gland at the base of the brain.',
            'Characteristics': [
                'Often benign',
                'Can affect hormone production',
                'Treatable with surgery or medication',
                'May cause vision problems due to proximity to optic nerves'
            ],
            'Common Symptoms': [
                'Hormone imbalances',
                'Visual disturbances',
                'Headaches',
                'Fatigue',
                'Reproductive issues'
            ],
            'Treatment Options': [
                'Medication to control hormone production',
                'Transsphenoidal surgery',
                'Radiation therapy',
                'Hormone replacement therapy'
            ]
        }
    }

def create_symptoms_data():
    return {
        'Common': [
            'Headaches (especially in the morning)',
            'Seizures',
            'Nausea and vomiting',
            'Fatigue',
            'Vision problems'
        ],
        'Neurological': [
            'Speech difficulties',
            'Memory problems',
            'Balance issues',
            'Coordination problems',
            'Numbness or tingling in extremities'
        ],
        'Cognitive': [
            'Confusion',
            'Personality changes',
            'Difficulty concentrating',
            'Problems with reasoning',
            'Loss of initiative'
        ],
        'When to See a Doctor': [
            'Persistent, worsening headaches',
            'Unexplained nausea or vomiting',
            'Vision problems',
            'Gradual loss of sensation or movement in arms or legs',
            'Difficulty with balance',
            'Speech difficulties',
            'Confusion in everyday matters',
            'Seizures, especially if you dont have a history of seizures'
        ]
    }

def create_treatment_options():
    return {
        'Surgery': {
            'Description': 'Physical removal of tumor tissue',
            'When Used': 'When tumors are accessible and patient is healthy enough',
            'Benefits': 'Direct removal of tumor tissue',
            'Risks': 'Infection, bleeding, damage to surrounding tissue'
        },
        'Radiation Therapy': {
            'Description': 'Use of high-energy beams to kill tumor cells',
            'When Used': 'After surgery or for inoperable tumors',
            'Benefits': 'Non-invasive, can target specific areas',
            'Risks': 'Damage to healthy tissue, fatigue, skin problems'
        },
        'Chemotherapy': {
            'Description': 'Use of drugs to kill tumor cells',
            'When Used': 'Often in combination with surgery and radiation',
            'Benefits': 'Can reach cancer cells throughout the body',
            'Risks': 'Side effects, weakened immune system'
        },
        'Targeted Therapy': {
            'Description': 'Drugs that target specific abnormalities in tumor cells',
            'When Used': 'For specific tumor types with known genetic profiles',
            'Benefits': 'More precise than chemotherapy, fewer side effects',
            'Risks': 'Limited effectiveness, resistance development'
        },
        'Stereotactic Radiosurgery': {
            'Description': 'Precise delivery of radiation to tumor',
            'When Used': 'Small tumors or as follow-up to surgery',
            'Benefits': 'Highly targeted, minimal damage to surrounding tissue',
            'Risks': 'Limited to smaller tumors, potential radiation damage'
        }
    }

# Page Functions

def home_page():
    st.header('Brain Tumor Detection System')
    
    # Hero section with animation
    col1, col2 = st.columns([3, 2])
    
    with col1:
        st.markdown("""
        <div class="hero-section">
            <h2>Advanced Brain Tumor Detection & Analysis</h2>
            <p class="hero-text">
                Leveraging AI to provide accurate tumor detection, classification, and comprehensive information
                for patients, healthcare providers, and researchers.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="call-to-action">
            <p>Upload your MRI scan now to get an instant analysis</p>
        </div>
        """, unsafe_allow_html=True)
        
        button_col1, button_col2 = st.columns(2)
        
        with button_col1:
            if st.button("🔍 Start Detection", use_container_width=True):
                st.session_state['current_page'] = "🔍 Tumor Detection"
                st.rerun()
                
        with button_col2:
            if st.button("🚶‍♂️ Take the Tutorial", use_container_width=True):
                st.session_state['current_page'] = "🚶‍♂️ Tutorial"
                st.rerun()
    
    with col2:
        # Either display an image or a placeholder for the brain MRI
        try:
            st.image("https://healtharkinsights.com/wp-content/uploads/2020/12/img1-1958x1200.jpg")
        except:
            st.markdown("""
            <div class="brain-img-placeholder">
                <div class="brain-img-content">Brain MRI Analysis</div>
            </div>
            """, unsafe_allow_html=True)
    
    # Statistics counter section
    st.markdown("""
    <div class="stats-section">
        <div class="stat-item">
            <div class="stat-number">7,000+</div>
            <div class="stat-label">MRI Scans Analyzed</div>
        </div>
        <div class="stat-item">
            <div class="stat-number">95%</div>
            <div class="stat-label">Accuracy Rate</div>
        </div>
        <div class="stat-item">
            <div class="stat-number">4</div>
            <div class="stat-label">Tumor Types Classified</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    # Features section with hover effects
    st.markdown("""
    <h2 style="color: #2c3e50; margin: 40px 0 15px; text-align: center; font-weight: 700; font-size: 2.2rem;">Key Features</h2>
    <p style="text-align: center; color: #7f8c8d; max-width: 700px; margin: 0 auto 30px; font-size: 1.1rem;">
        Our comprehensive brain tumor detection system combines cutting-edge AI with user-friendly interfaces
    </p>
    """, unsafe_allow_html=True)
    
    # Features with improved cards and hover effects
    feature_cols = st.columns(3)
    
    features = [
        {
            "icon": "🔬",
            "title": "AI-Powered Detection",
            "desc": "Advanced machine learning algorithms trained on thousands of MRI scans provide accurate tumor classification with 95% accuracy."
        },
        {
            "icon": "📊",
            "title": "Comprehensive Analysis",
            "desc": "Detailed visualization with GradCAM technology to highlight tumor regions and provide confidence scores for greater transparency."
        },
        {
            "icon": "📚",
            "title": "Educational Resources",
            "desc": "Extensive information on brain tumors, symptoms, treatment options, and global statistics to support informed healthcare decisions."
        }
    ]
    
    for i, col in enumerate(feature_cols):
        with col:
            st.markdown(f"""
            <div style="background-color: white; border-radius: 12px; padding: 30px 25px; height: 100%;
                 box-shadow: 0 8px 24px rgba(0,0,0,0.07); border-top: 5px solid #3498db; transition: all 0.3s ease;"
                 class="feature-card" onmouseover="this.style.transform='translateY(-10px)';this.style.boxShadow='0 15px 30px rgba(0,0,0,0.1)';" 
                 onmouseout="this.style.transform='translateY(0px)';this.style.boxShadow='0 8px 24px rgba(0,0,0,0.07)';">
                <div style="font-size: 3.5rem; margin-bottom: 20px; text-align: center;">{features[i]['icon']}</div>
                <h3 style="margin-bottom: 15px; text-align: center; color: #2c3e50; font-size: 1.4rem; font-weight: 600;">{features[i]['title']}</h3>
                <p style="color: #555; line-height: 1.7; text-align: center; font-size: 1rem;">{features[i]['desc']}</p>
            </div>
            """, unsafe_allow_html=True)
    
    # Add CSS for hover effects
    # st.markdown("""
    # <style>
    # .feature-card:hover {
    #     transform: translateY(-10px);
    #     box-shadow: 0 15px 30px rgba(0,0,0,0.1);
    # }
    # </style>
    # """, unsafe_allow_html=True)
    
    # "At a Glance" process section
    st.markdown("""
    <div style="margin: 60px 0 40px;">
        <h2 style="color: #2c3e50; margin-bottom: 15px; text-align: center; font-weight: 700; font-size: 2.2rem;">At a Glance</h2>
        <p style="text-align: center; color: #7f8c8d; max-width: 700px; margin: 0 auto 40px; font-size: 1.1rem;">
            Experience a seamless workflow from upload to analysis in just four simple steps
        </p>
        <div style="display: flex; flex-wrap: wrap; gap: 15px; justify-content: center; max-width: 950px; margin: 0 auto;">
            <!-- Step 1 -->
            <div style="flex: 1; min-width: 200px; max-width: 220px; background: white; border-radius: 12px; padding: 25px; 
                 box-shadow: 0 8px 24px rgba(0,0,0,0.07); text-align: center; transition: all 0.3s ease;"
                 class="step-card">
                <div style="width: 60px; height: 60px; background-color: #3498db; border-radius: 50%; display: flex; 
                     justify-content: center; align-items: center; color: white; font-size: 1.8rem; font-weight: bold; 
                     margin: 0 auto 15px auto;">1</div>
                <h3 style="color: #3498db; margin-bottom: 10px; font-size: 1.2rem; font-weight: 600;">Upload Scan</h3>
                <p style="color: #555; font-size: 0.95rem;">Submit your brain MRI scan in common image formats</p>
            </div>
            <!-- Step 2 -->
            <div style="flex: 1; min-width: 200px; max-width: 220px; background: white; border-radius: 12px; padding: 25px; 
                 box-shadow: 0 8px 24px rgba(0,0,0,0.07); text-align: center; transition: all 0.3s ease;"
                 class="step-card">
                <div style="width: 60px; height: 60px; background-color: #3498db; border-radius: 50%; display: flex; 
                     justify-content: center; align-items: center; color: white; font-size: 1.8rem; font-weight: bold; 
                     margin: 0 auto 15px auto;">2</div>
                <h3 style="color: #3498db; margin-bottom: 10px; font-size: 1.2rem; font-weight: 600;">AI Analysis</h3>
                <p style="color: #555; font-size: 0.95rem;">Our deep learning model analyzes your scan with precision</p>
            </div>
            <!-- Step 3 -->
            <div style="flex: 1; min-width: 200px; max-width: 220px; background: white; border-radius: 12px; padding: 25px; 
                 box-shadow: 0 8px 24px rgba(0,0,0,0.07); text-align: center; transition: all 0.3s ease;"
                 class="step-card">
                <div style="width: 60px; height: 60px; background-color: #3498db; border-radius: 50%; display: flex; 
                     justify-content: center; align-items: center; color: white; font-size: 1.8rem; font-weight: bold; 
                     margin: 0 auto 15px auto;">3</div>
                <h3 style="color: #3498db; margin-bottom: 10px; font-size: 1.2rem; font-weight: 600;">Get Results</h3>
                <p style="color: #555; font-size: 0.95rem;">Receive diagnosis, confidence score and visual heatmap</p>
            </div>
            <!-- Step 4 -->
            <div style="flex: 1; min-width: 200px; max-width: 220px; background: white; border-radius: 12px; padding: 25px; 
                 box-shadow: 0 8px 24px rgba(0,0,0,0.07); text-align: center; transition: all 0.3s ease;"
                 class="step-card">
                <div style="width: 60px; height: 60px; background-color: #3498db; border-radius: 50%; display: flex; 
                     justify-content: center; align-items: center; color: white; font-size: 1.8rem; font-weight: bold; 
                     margin: 0 auto 15px auto;">4</div>
                <h3 style="color: #3498db; margin-bottom: 10px; font-size: 1.2rem; font-weight: 600;">Export Report</h3>
                <p style="color: #555; font-size: 0.95rem;">Download detailed PDF or CSV reports to share</p>
            </div>
        </div>    
        <div style="text-align: center; margin-top: 25px;">
            <p style="font-size: 0.9rem; color: #7f8c8d; font-style: italic;">
                Need more details? Take our <a href="#" style="color: #3498db; text-decoration: none; font-weight: 500;">comprehensive tutorial</a> for in-depth guidance.
            </p>
        </div>
    </div>
    
    <style>
    .step-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 15px 30px rgba(0,0,0,0.1);
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Tumor types overview
    st.markdown("""
    <div style="background: linear-gradient(135deg, #f5f7fa 0%, #e4ecf7 100%); 
         padding: 40px; border-radius: 15px; margin: 40px 0; box-shadow: 0 8px 32px rgba(0,0,0,0.05);">
        <h2 style="color: #2c3e50; margin-bottom: 25px; text-align: center; font-weight: 700; font-size: 2.2rem;">Tumor Types We Detect</h2>
        <div style="display: flex; flex-wrap: wrap; gap: 20px; justify-content: center; margin-top: 30px;">
            <div style="flex: 1; min-width: 250px; background-color: white; padding: 25px; border-radius: 12px; 
                 box-shadow: 0 8px 24px rgba(0,0,0,0.07); border-top: 4px solid #e74c3c; transition: all 0.3s ease;"
                 class="tumor-card">
                <h3 style="color: #e74c3c; margin-bottom: 15px; font-size: 1.3rem; font-weight: 600;">Glioma Tumors</h3>
                <p style="color: #555; font-size: 1rem; line-height: 1.6;">
                    Originate from glial cells that support and nourish neurons in the brain. These can range from 
                    slow-growing to highly aggressive variants.
                </p>
            </div>
            <div style="flex: 1; min-width: 250px; background-color: white; padding: 25px; border-radius: 12px; 
                 box-shadow: 0 8px 24px rgba(0,0,0,0.07); border-top: 4px solid #3498db; transition: all 0.3s ease;"
                 class="tumor-card">
                <h3 style="color: #3498db; margin-bottom: 15px; font-size: 1.3rem; font-weight: 600;">Meningioma Tumors</h3>
                <p style="color: #555; font-size: 1rem; line-height: 1.6;">
                    Develop in the meninges, the membranes that surround the brain and spinal cord. Most are benign 
                    and slow-growing.
                </p>
            </div>
            <div style="flex: 1; min-width: 250px; background-color: white; padding: 25px; border-radius: 12px; 
                 box-shadow: 0 8px 24px rgba(0,0,0,0.07); border-top: 4px solid #9b59b6; transition: all 0.3s ease;"
                 class="tumor-card">
                <h3 style="color: #9b59b6; margin-bottom: 15px; font-size: 1.3rem; font-weight: 600;">Pituitary Tumors</h3>
                <p style="color: #555; font-size: 1rem; line-height: 1.6;">
                    Form in the pituitary gland at the base of the brain. Can affect hormone production and vision 
                    if they grow large.
                </p>
            </div>
        </div>
    </div>
    
    <style>
    .tumor-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 15px 30px rgba(0,0,0,0.1);
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Testimonials with modern design
    st.markdown("""
    <h2 style="color: #2c3e50; margin: 50px 0 30px; text-align: center; font-weight: 700; font-size: 2.2rem;">Healthcare Professionals Trust Us</h2>
    """, unsafe_allow_html=True)
    
    testimonial_cols = st.columns(2)
    
    testimonials = [
        {
            "quote": "This tool has significantly improved our diagnostic workflow. The GradCAM visualizations are particularly helpful in explaining findings to patients.",
            "author": "Dr. Sarah Johnson",
            "title": "Neurologist",
            "avatar": "👩‍⚕️",
            "color": "#3498db"
        },
        {
            "quote": "The educational resources combined with the AI detection make this a valuable tool for both clinical practice and medical education.",
            "author": "Dr. Michael Chen",
            "title": "Radiologist",
            "avatar": "👨‍⚕️",
            "color": "#9b59b6"
        }
    ]
    
    for i, col in enumerate(testimonial_cols):
        with col:
            st.markdown(f"""
            <div style="background-color: white; border-radius: 12px; padding: 30px 25px; height: 100%;
                 box-shadow: 0 8px 24px rgba(0,0,0,0.07); position: relative; margin-top: 20px; transition: all 0.3s ease;"
                 class="testimonial-card">
                <div style="position: absolute; top: -20px; left: 25px; width: 50px; height: 50px; 
                     background-color: {testimonials[i]['color']}; border-radius: 50%; display: flex; 
                     justify-content: center; align-items: center; color: white; font-size: 1.5rem; box-shadow: 0 4px 8px rgba(0,0,0,0.1);">
                    {testimonials[i]['avatar']}
                </div>
                <div style="position: absolute; top: 25px; right: 25px; font-size: 4rem; opacity: 0.1; color: {testimonials[i]['color']};">"</div>
                <div style="color: #555; font-style: italic; line-height: 1.8; margin: 15px 0 20px; font-size: 1.05rem; position: relative; z-index: 1;">
                    "{testimonials[i]['quote']}"
                </div>
                <div style="display: flex; align-items: center; border-top: 1px solid #eee; padding-top: 15px; margin-top: 15px;">
                    <div style="margin-left: auto;">
                        <div style="font-weight: 600; color: #2c3e50; font-size: 1.1rem;">{testimonials[i]['author']}</div>
                        <div style="color: {testimonials[i]['color']}; font-size: 0.9rem; font-weight: 500;">{testimonials[i]['title']}</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown("""
    <style>
    .testimonial-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 15px 30px rgba(0,0,0,0.1);
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Final call to action with gradient background
    st.markdown("""
    <div style="background: linear-gradient(135deg, #2980b9 0%, #3498db 100%); padding: 50px 30px; border-radius: 15px; 
         margin: 60px 0 30px; text-align: center; box-shadow: 0 10px 30px rgba(52, 152, 219, 0.3);">
        <h2 style="color: white; margin-bottom: 20px; font-weight: 700; font-size: 2.3rem; text-shadow: 1px 1px 2px rgba(0,0,0,0.1);">
            Ready to Experience Advanced Brain Tumor Detection?
        </h2>
        <p style="color: rgba(255,255,255,0.9); font-size: 1.2rem; margin-bottom: 30px; max-width: 700px; 
              margin-left: auto; margin-right: auto; line-height: 1.6;">
            Start using our AI-powered system now and gain valuable insights from your MRI scans.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Final action button in larger size
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("""
        <div style="margin-bottom: 15px; text-align: center;">
            <p style="font-size: 1rem; color: #7f8c8d; margin-bottom: 10px;">No account or sign-up required</p>
        </div>
        """, unsafe_allow_html=True)
        if st.button("🚀 Start Your First Analysis", key="final_cta", use_container_width=True):
            st.session_state['current_page'] = "🔍 Tumor Detection"
            st.rerun()
    

def tumor_detection_page():
    # Make sure we stay on this page during file upload and prediction
    st.session_state['current_page'] = "🔍 Tumor Detection"
    
    # Initialize session state variables if they don't exist
    if 'analysis_complete' not in st.session_state:
        st.session_state['analysis_complete'] = False
    if 'predicted_class' not in st.session_state:
        st.session_state['predicted_class'] = None
    if 'confidence' not in st.session_state:
        st.session_state['confidence'] = None
    if 'original_img' not in st.session_state:
        st.session_state['original_img'] = None
    if 'gradcam_heatmap' not in st.session_state:
        st.session_state['gradcam_heatmap'] = None
    if 'overlayed_img' not in st.session_state:
        st.session_state['overlayed_img'] = None
    
    st.header("Brain Tumor Detection & Analysis")
    
    # Two columns layout for initial content - explanation and upload
    col1, col2 = st.columns([1, 1], gap="large")
    
    with col1:
        # Your existing explanation code
        st.subheader("How It Works")
        st.write("Our AI-powered system analyzes brain MRI scans to detect and classify tumors with high accuracy.")
        
        st.write("**The system can identify:**")
        st.write("• Glioma tumors")
        st.write("• Meningioma tumors")
        st.write("• Pituitary tumors")
        st.write("• Absence of tumors")
        
        st.write("**The Analysis Provides:**")
        st.write("• Tumor classification")
        st.write("• Confidence score")
        st.write("• GradCAM visualization highlighting tumor regions")
    
    with col2:
        st.markdown("""
        <div class="upload-card">
            <h3>Upload MRI Scan</h3>
            <p>Please upload a brain MRI scan image in JPG, JPEG, or PNG format.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Improve the file uploader appearance
        test_image = st.file_uploader("Choose an MRI scan image:", type=["jpg", "jpeg", "png"])
        
        # Check if demo mode is active
        if 'show_demo' in st.session_state and st.session_state['show_demo']:
            test_image = handle_demo_mode()
        
        if test_image is not None:
            # Show the uploaded image
            st.image(test_image, caption="Uploaded Brain MRI Scan", width=300)
            
            # Predict button with improved styling
            if not st.session_state['analysis_complete']:
                predict_button = st.button("🔍 Analyze Image", use_container_width=True)
                
                if predict_button:
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    # Simulate analysis steps
                    status_text.text("Loading model...")
                    progress_bar.progress(20)
                    
                    status_text.text("Preprocessing image...")
                    progress_bar.progress(40)
                    
                    status_text.text("Running AI analysis...")
                    progress_bar.progress(60)
                    
                    status_text.text("Generating visualizations...")
                    progress_bar.progress(80)
                    
                    # Actual prediction
                    predicted_class, confidence, gradcam_heatmap, original_img = model_prediction(test_image.read())
                    
                    progress_bar.progress(100)
                    status_text.text("Analysis complete!")
                    
                    # Store results in session state
                    if predicted_class is not None:
                        st.session_state['predicted_class'] = predicted_class
                        st.session_state['confidence'] = confidence
                        st.session_state['original_img'] = original_img
                        st.session_state['gradcam_heatmap'] = gradcam_heatmap
                        
                        # Try to compute overlay if both components are available
                        if gradcam_heatmap is not None and original_img is not None:
                            try:
                                overlayed_img = overlay_gradcam(original_img, gradcam_heatmap)
                                st.session_state['overlayed_img'] = overlayed_img
                            except Exception as e:
                                st.error(f"Error generating overlay: {str(e)}")
                                st.session_state['overlayed_img'] = None
                        else:
                            st.session_state['overlayed_img'] = None
                            
                        # Mark analysis as complete to display results
                        st.session_state['analysis_complete'] = True
                        st.rerun()  # Rerun to display results
                    else:
                        st.error("Analysis failed. Please try uploading a different MRI image.")
    
    # Show results if analysis is complete - NOW FULLWIDTH CENTERED LAYOUT
    if st.session_state['analysis_complete'] and test_image is not None:
        predicted_class = st.session_state['predicted_class']
        confidence = st.session_state['confidence']
        original_img = st.session_state['original_img']
        gradcam_heatmap = st.session_state['gradcam_heatmap']
        overlayed_img = st.session_state['overlayed_img']
        
        if predicted_class is None:
            st.error("Analysis results are not available. Please try again.")
            # Reset analysis state if results are invalid
            st.session_state['analysis_complete'] = False
            st.rerun()
        
        # Clear separation between input and results
        st.markdown("<hr/>", unsafe_allow_html=True)
        st.subheader("📊 Analysis Results")
        
        # Results overview - now using full width with columns
        result_cols = st.columns(2)
        
        with result_cols[0]:
            st.markdown(f"""
            <div class="result-card">
                <h3>Diagnosis</h3>
                <p class="result-tumor-type">{predicted_class.replace('_', ' ').title()}</p>
                <div class="confidence-bar">
                    <div class="confidence-value" style="width: {confidence}%"></div>
                </div>
                <p>Confidence: {confidence}%</p>
            </div>
            """, unsafe_allow_html=True)
        
        with result_cols[1]:
            # Determine risk level based on tumor type
            risk_level = "Low" if predicted_class == "no_tumor" else "High" if predicted_class == "glioma_tumor" else "Moderate"
            
            # Get info about next steps based on tumor type
            next_steps = {
                "no_tumor": "No immediate action required. Regular check-ups recommended.",
                "glioma_tumor": "Urgent consultation with a neurologist. Further diagnostic tests needed.",
                "meningioma_tumor": "Follow-up with a neurologist. Additional imaging may be required.",
                "pituitary_tumor": "Consult with an endocrinologist and neurosurgeon. Hormone tests recommended."
            }
            
            st.markdown(f"""
            <div class="result-card">
                <h3>Recommended Action</h3>
                <p class="risk-level risk-{risk_level.lower()}">{risk_level} Risk</p>
                <p>{next_steps.get(predicted_class, "Consult with a healthcare professional.")}</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Detailed visualization section
        st.markdown("<hr/>", unsafe_allow_html=True)
        st.subheader("🔬 Detailed Visualization")
        
        # Display visualizations in 3 equal columns with container for size control
        st.markdown("""
        <div style="max-width: 90%; margin: 0 auto;">
            <p style="text-align: center; font-style: italic; margin-bottom: 15px;">
                The visualizations below show the original MRI, areas of interest identified by our AI (heatmap), and an overlay of both.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Create columns with spacing
        visual_cols = st.columns([1, 1, 1])

        with visual_cols[0]:
            st.markdown("<h4 style='text-align: center;'>Original MRI</h4>", unsafe_allow_html=True)
            if original_img is not None:
                # Create a container with fixed width to control image size
                col1, col2, col3 = st.columns([1, 3, 1])
                with col2:
                    st.image(original_img, use_container_width=True)
            else:
                st.warning("Original image not available.")

        with visual_cols[1]:
            st.markdown("<h4 style='text-align: center;'>GradCAM Heatmap</h4>", unsafe_allow_html=True)
            if gradcam_heatmap is not None:
                from PIL import Image
                heatmap_pil = Image.fromarray(gradcam_heatmap)
                # Ensure the heatmap has the same dimensions as original image
                heatmap_pil = heatmap_pil.resize(original_img.size)
                # Create a container with fixed width to control image size
                col1, col2, col3 = st.columns([1, 3, 1])
                with col2:
                    st.image(heatmap_pil, use_container_width=True)
            else:
                st.warning("GradCAM heatmap could not be generated.")

        with visual_cols[2]:
            st.markdown("<h4 style='text-align: center;'>Overlay</h4>", unsafe_allow_html=True)
            if overlayed_img is not None:
                # Ensure overlay has the same dimensions as original image
                overlayed_img = overlayed_img.resize(original_img.size)
                # Create a container with fixed width to control image size
                col1, col2, col3 = st.columns([1, 3, 1])
                with col2:
                    st.image(overlayed_img, use_container_width=True)
            else:
                st.warning("Overlay could not be generated.")
        
        # Display legend for GradCAM in a centered container
        st.markdown("""
        <div style="background-color: #f8f9fa; padding: 15px; border-radius: 5px; margin: 20px auto; max-width: 90%; border-left: 4px solid #3498db;">
            <h4 style="margin-top: 0;">GradCAM Interpretation</h4>
            <p>The heatmap highlights areas that influenced the AI's decision:</p>
            <div style="display: flex; flex-direction: column; gap: 10px;">
                <div style="display: flex; align-items: center;">
                    <div style="display: flex; width: 100%; height: 20px; border-radius: 3px; overflow: hidden;">
                        <div style="background-color: blue; width: 20%;"></div>
                        <div style="background-color: cyan; width: 20%;"></div>
                        <div style="background-color: green; width: 20%;"></div>
                        <div style="background-color: yellow; width: 20%;"></div>
                        <div style="background-color: red; width: 20%;"></div>
                    </div>
                </div>
                <div style="display: flex; justify-content: space-between;">
                    <span>Less important</span>
                    <span>More important</span>
                </div>
            </div>
            <p style="margin-top: 10px; font-style: italic;">Red areas indicate regions that strongly influenced the model's prediction.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Tumor information section if a tumor was detected - centered design
        if predicted_class != "no_tumor":
            doctor_recommendation_section(predicted_class)
            st.markdown("---")
            st.subheader(f"ℹ️ About {predicted_class.replace('_', ' ').title()}")
            
            # Get the tumor details
            tumor_key = predicted_class.replace('_', ' ').title()
            tumor_info = create_tumor_details().get(tumor_key, {})
            
            if tumor_info:
                # Two columns for better layout
                info_cols = st.columns([3, 2])
                
                with info_cols[0]:
                    # Description
                    st.write(f"**Description:** {tumor_info.get('Description', 'No description available')}")
                    
                    # Characteristics
                    st.write("**Characteristics:**")
                    for char in tumor_info.get('Characteristics', ['Information not available']):
                        st.write(f"• {char}")
                    
                    # Common Symptoms
                    st.write("**Common Symptoms:**")
                    for symptom in tumor_info.get('Common Symptoms', ['Information not available']):
                        st.write(f"• {symptom}")
                
                with info_cols[1]:
                    # Treatment Options
                    st.write("**Treatment Options:**")
                    for treatment in tumor_info.get('Treatment Options', ['Consult with a healthcare professional']):
                        st.write(f"• {treatment}")
                    
                    # Add a visual element - info card for the specific tumor
                    tumor_colors = {
                        "Glioma Tumor": "#f8d7da",
                        "Meningioma Tumor": "#d1ecf1",
                        "Pituitary Tumor": "#e2e3f9"
                    }
                    border_colors = {
                        "Glioma Tumor": "#f5c6cb",
                        "Meningioma Tumor": "#bee5eb",
                        "Pituitary Tumor": "#d6d8f0"
                    }
                    
                    st.markdown(f"""
                    <div style="background-color: {tumor_colors.get(tumor_key, '#f8f9fa')}; 
                                border: 1px solid {border_colors.get(tumor_key, '#dee2e6')}; 
                                border-radius: 10px; padding: 15px; margin-top: 20px;">
                        <div style="text-align: center; margin-bottom: 10px;">
                            <span style="font-size: 48px;">🧠</span>
                        </div>
                        <h4 style="text-align: center; margin-bottom: 10px;">{tumor_key} Visualization</h4>
                        <p style="font-style: italic; text-align: center;">
                            {tumor_info.get('Description', 'No description available')}
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.warning("Detailed information about this tumor type is not available.")



        
        # Export section - using two columns for better layout
        st.markdown("---")
        st.subheader("📊 Export Analysis Results")
        
        export_col1, export_col2 = st.columns(2)
        
        with export_col1:
            st.info("Export a detailed PDF report with images and recommendations")
            if st.button("📄 Generate PDF Report", key="pdf_button", use_container_width=True):
                try:
                    # Generate PDF using data from session state
                    pdf_bytes = export_pdf_report(
                        predicted_class, 
                        confidence, 
                        original_img, 
                        gradcam_heatmap, 
                        overlayed_img
                    )
                    
                    # Create a download link
                    from datetime import datetime
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    download_filename = f"brain_tumor_analysis_{timestamp}.pdf"
                    
                    # Display download link
                    st.markdown(
                        create_download_link(pdf_bytes, download_filename),
                        unsafe_allow_html=True
                    )
                except Exception as e:
                    st.error(f"Error generating PDF: {str(e)}")
        
        with export_col2:
            st.info("Export a simple CSV file with analysis results")
            if st.button("📊 Generate CSV Report", key="csv_button", use_container_width=True):
                try:
                    # Generate CSV using data from session state
                    csv_bytes = export_csv_report(predicted_class, confidence)
                    
                    # Create a download link
                    from datetime import datetime
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    download_filename = f"brain_tumor_analysis_{timestamp}.csv"
                    
                    # Display download link
                    st.markdown(
                        create_download_link(csv_bytes, download_filename),
                        unsafe_allow_html=True
                    )
                except Exception as e:
                    st.error(f"Error generating CSV: {str(e)}")
        
        # Add a button to reset the analysis - centered for better visibility
        st.markdown("---")
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("🔄 Analyze Another Image", use_container_width=True):
                # Reset the analysis state
                st.session_state['analysis_complete'] = False
                # Keep the current page
                st.rerun()
    else:
        # Reset analysis state when no image is uploaded
        if st.session_state['analysis_complete']:
            st.session_state['analysis_complete'] = False
def tutorial_page():
    st.header("🚶‍♂️ Tutorial Walkthrough")
    
    st.markdown("""
    <div class="info-card">
        <p>Welcome to the BrainScan AI tutorial! This guide will walk you through the main features of our brain tumor detection system
        and how to use them effectively.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Tutorial Sections
    st.subheader("How to Use the Brain Tumor Detection System")
    
    # Step 1
    st.markdown("""
    <div class="tutorial-step">
        <div class="step-number">1</div>
        <div class="step-content">
            <h3>Navigate to the Tumor Detection Page</h3>
            <p>From the home page, either:</p>
            <ul>
                <li>Click the "🔍 Start Detection" button in the center of the home page, or</li>
                <li>Select "🔍 Tumor Detection" from the sidebar navigation menu</li>
            </ul>
            <div style="background-color: #f0f2f6; padding: 10px; border-radius: 5px; text-align: center;">
                <p style="margin: 0; font-style: italic;">Navigation to the Tumor Detection page</p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Step 2
    st.markdown("""
    <div class="tutorial-step">
        <div class="step-number">2</div>
        <div class="step-content">
            <h3>Upload Your MRI Scan</h3>
            <p>On the Tumor Detection page:</p>
            <ul>
                <li>Click the "Browse files" button in the upload section</li>
                <li>Select a brain MRI scan image from your computer (supported formats: JPG, JPEG, PNG)</li>
                <li>The selected image will appear on the page once uploaded</li>
            </ul>
            <p><em>Note: For best results, use clear, high-quality MRI scans taken in the axial plane.</em></p>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Step 3
    st.markdown("""
    <div class="tutorial-step">
        <div class="step-number">3</div>
        <div class="step-content">
            <h3>Analyze the Image</h3>
            <p>After your image is uploaded:</p>
            <ul>
                <li>Click the "🔍 Analyze Image" button</li>
                <li>The system will process your image (this may take a few seconds)</li>
                <li>A progress bar will show the status of the analysis</li>
            </ul>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Step 4
    st.markdown("""
    <div class="tutorial-step">
        <div class="step-number">4</div>
        <div class="step-content">
            <h3>Interpret the Results</h3>
            <p>Once analysis is complete, you'll see several sections:</p>
            <ul>
                <li><strong>Diagnosis:</strong> Shows the detected tumor type (or no tumor) and confidence score</li>
                <li><strong>Recommended Action:</strong> Suggests next steps based on the diagnosis</li>
                <li><strong>Detailed Visualization:</strong> Displays the original MRI, GradCAM heatmap, and overlay</li>
            </ul>
            <p>The GradCAM heatmap highlights areas that influenced the AI's decision, with red areas indicating regions most strongly associated with the detected tumor type.</p>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Step 5 - UPDATED to highlight the export feature
    st.markdown("""
    <div class="tutorial-step">
        <div class="step-number">5</div>
        <div class="step-content">
            <h3>Export Your Results</h3>
            <p>To save or share your analysis:</p>
            <ul>
                <li>Scroll down to the "📊 Export Analysis Results" section</li>
                <li>Choose between:</li>
                <ul>
                    <li><strong>PDF Report:</strong> Comprehensive report with images, diagnosis, and recommendations</li>
                    <li><strong>CSV Report:</strong> Simple data export with key analysis metrics</li>
                </ul>
                <li>Click the corresponding button to generate your report</li>
                <li>Click the download link that appears to save the file to your computer</li>
            </ul>
            <div style="background-color: #e8f4f8; border-left: 4px solid #3498db; padding: 10px; margin-top: 10px;">
                <p style="margin: 0;"><strong>Tip:</strong> The PDF report includes comprehensive information about the detected tumor type and is ideal for sharing with healthcare providers.</p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Educational Resources - UPDATED to include the FAQ page
    st.subheader("Exploring Educational Resources")
    
    st.markdown("""
    <div class="tutorial-resources">
        <p>Our system offers extensive educational resources to help you understand brain tumors:</p>
        <div class="resource-card">
            <h4>📊 Survival Statistics</h4>
            <p>Access detailed survival rates and trends for different tumor types, along with treatment success data.</p>
        </div>
        <div class="resource-card">
            <h4>🩺 Tumor Types</h4>
            <p>Learn about different brain tumor types, their characteristics, symptoms, and treatment options.</p>
        </div>
        <div class="resource-card">
            <h4>⚠️ Prevention & Care</h4>
            <p>Find information on risk factors, warning signs, lifestyle recommendations, and care guidelines.</p>
        </div>
        <div class="resource-card">
            <h4>📝 Risk Assessment</h4>
            <p>Complete a questionnaire to evaluate personal risk factors related to brain tumors.</p>
        </div>
        <div class="resource-card" style="border-left: 5px solid #3498db;">
            <h4>❓ FAQ</h4>
            <p>Get answers to common questions about brain tumors, MRI scans, diagnosis, treatment options, and our AI system.</p>
            <p><em>Our newly added FAQ section covers questions about brain tumor types, symptoms, diagnosis processes, treatment options, and how our AI detection system works.</em></p>
        </div>
    </div>
    """,unsafe_allow_html=True)
    
    # Best Practices
    st.subheader("Best Practices & Tips")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="tips-card">
            <h4>For Accurate Results:</h4>
            <ul>
                <li>Use high-quality MRI scans with good contrast</li>
                <li>Ensure the scan shows a complete cross-section of the brain</li>
                <li>Use recent scans for the most relevant results</li>
                <li>Check the confidence score - higher scores indicate greater certainty</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="tips-card">
            <h4>Important Reminders:</h4>
            <ul>
                <li>This system is a supportive tool, not a replacement for professional medical advice</li>
                <li>Always consult healthcare professionals for diagnosis and treatment decisions</li>
                <li>The results should be reviewed by qualified medical personnel</li>
                <li>Use the educational resources to better understand the context of the results</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    # Interactive Demo Button
    st.markdown("### Try a Demo")
    
    if st.button("🔬 Try with a Sample MRI", use_container_width=True):
        st.session_state['current_page'] = "🔍 Tumor Detection"
        st.session_state['show_demo'] = True
        st.rerun()

def handle_demo_mode():
    if 'show_demo' in st.session_state and st.session_state['show_demo']:
        # Clear the flag so it only happens once
        st.session_state['show_demo'] = False
        
        # Info message about demo mode
        st.info("🔬 Demo Mode: A sample MRI has been loaded for demonstration purposes. Click 'Analyze Image' to proceed.")
        
        # Instead of relying on external image files, we'll create a demo image
        # approach 1: Use one of the sample images you've already included in your dataset
        try:
            # Try to use one of your sample MRI images from the dataset page
            sample_img_path = "sample_mri_glioma.jpg"
            from PIL import Image
            sample_img = Image.open(sample_img_path)
            return sample_img
        except:
            pass
            
        # approach 2: Generate a placeholder if no sample image is available
        try:
            import numpy as np
            from PIL import Image, ImageDraw, ImageFont
            
            # Create a blank image
            img = Image.new('RGB', (400, 400), color = (0, 0, 0))
            draw = ImageDraw.Draw(img)
            
            # Draw a simple circular shape to represent a brain
            draw.ellipse((50, 50, 350, 350), outline=(255, 255, 255), width=2)
            
            # Draw a smaller shape to represent a potential tumor
            draw.ellipse((250, 150, 300, 200), fill=(200, 200, 200), outline=(255, 255, 255))
            
            # Add some text
            draw.text((120, 180), "Sample MRI", fill=(255, 255, 255))
            
            return img
        except Exception as e:
            st.warning(f"Could not create demo image. Please upload your own MRI scan. Error: {str(e)}")
            return None
    
    return None


def medical_terminology_page():
    st.header("🔬 Medical Terminology Glossary")
    
    # Add a search box with improved styling
    st.markdown("""
    <div class="search-container">
        <p>Search for medical terms related to brain tumors and neuroimaging</p>
    </div>
    """, unsafe_allow_html=True)
    
    search_term = st.text_input("", placeholder="Type to search...", label_visibility="collapsed")
    
    # Get glossary data
    glossary = create_medical_terminology_glossary()
    
    # Filter based on search term
    if search_term:
        filtered_glossary = {
            k: v for k, v in glossary.items() 
            if search_term.lower() in k.lower() or search_term.lower() in v.lower()
        }
    else:
        filtered_glossary = glossary
    
    # Display glossary in an attractive format
    if filtered_glossary:
        # Use columns for better layout
        col1, col2 = st.columns(2)
        
        # Split the glossary into two parts for the columns
        glossary_items = list(filtered_glossary.items())
        mid_point = len(glossary_items) // 2
        
        # First column
        with col1:
            for term, definition in glossary_items[:mid_point]:
                st.markdown(f"""
                <div class="glossary-item">
                    <h3>{term}</h3>
                    <p>{definition}</p>
                </div>
                """, unsafe_allow_html=True)
        
        # Second column
        with col2:
            for term, definition in glossary_items[mid_point:]:
                st.markdown(f"""
                <div class="glossary-item">
                    <h3>{term}</h3>
                    <p>{definition}</p>
                </div>
                """, unsafe_allow_html=True)
    else:
        st.warning("No matching terms found. Please try a different search.")
    
    # Add a helpful note at the bottom
    st.markdown("""
    <div class="info-note">
        <p><strong>Note:</strong> This glossary provides general information and should not replace professional medical advice.</p>
    </div>
    """, unsafe_allow_html=True)

def global_statistics_page():
    st.header("🌍 Global Brain Tumor Statistics")
    
    # Get global statistics data
    global_stats = generate_global_tumor_statistics()
    
    # Tabs for different statistics views
    stat_tabs = st.tabs(["Regional Analysis", "Age & Gender Data", "Treatment Success Rates"])
    
    with stat_tabs[0]:
        st.subheader("Regional Brain Tumor Distribution")
        
        # Year selector
        years = sorted(global_stats['Year'].unique())
        selected_year = st.select_slider(
            "Select year to view data:",
            options=years,
            value=years[-1]  # Default to most recent year
        )
        
        # Filter data for selected year
        year_data = global_stats[global_stats['Year'] == selected_year]
        
        # Display a metric showing overall cases for selected year
        total_cases = year_data['New Cases'].sum()
        prev_year_data = global_stats[global_stats['Year'] == selected_year - 1] if selected_year > years[0] else None
        
        if prev_year_data is not None:
            prev_total = prev_year_data['New Cases'].sum()
            percent_change = ((total_cases - prev_total) / prev_total) * 100
            st.metric(
                f"Total Global Brain Tumor Cases ({selected_year})", 
                f"{total_cases:,}",
                f"{percent_change:.1f}% from {selected_year-1}"
            )
        else:
            st.metric(f"Total Global Brain Tumor Cases ({selected_year})", f"{total_cases:,}")
        
        # Create a visually appealing choropleth map visualization for countries
        fig_map = px.choropleth(
            year_data,
            locations="Country",
            locationmode="country names",
            color="New Cases",
            hover_name="Country",
            hover_data=["New Cases", "Mortality Rate (%)", "Average Age of Diagnosis"],
            color_continuous_scale="Viridis",
            projection="natural earth",
            title=f"Global Distribution of Brain Tumor Cases ({selected_year})",
            labels={"New Cases": "New Cases"},
        )
        
        # Make the map responsive
        fig_map.update_layout(
            autosize=True,
            margin=dict(l=0, r=0, t=50, b=0),
            coloraxis_colorbar=dict(title="Number of Cases")
        )
        
        st.plotly_chart(fig_map, use_container_width=True)
        
        # Add a regional summary for clarity
        st.subheader(f"Regional Summary ({selected_year})")
        
        # Aggregate data by region
        region_summary = year_data.groupby('Region').agg({
            'New Cases': 'sum',
            'Mortality Rate (%)': 'mean',
            'Average Age of Diagnosis': 'mean'
        }).reset_index()
        
        # Round the averages to 1 decimal place
        region_summary['Mortality Rate (%)'] = region_summary['Mortality Rate (%)'].round(1)
        region_summary['Average Age of Diagnosis'] = region_summary['Average Age of Diagnosis'].round(1)
        
        # Create a bar chart for regional data
        fig_region = px.bar(
            region_summary,
            x="Region",
            y="New Cases",
            color="Mortality Rate (%)",
            text="New Cases",
            title=f"Brain Tumor Cases by Region ({selected_year})",
            labels={"New Cases": "Number of New Cases", "Mortality Rate (%)": "Mortality Rate (%)"},
            color_continuous_scale="Reds"
        )
        
        fig_region.update_layout(
            xaxis_title="Region",
            yaxis_title="Number of New Cases",
            coloraxis_colorbar=dict(title="Mortality Rate (%)")
        )
        
        fig_region.update_traces(texttemplate='%{text:,}', textposition='outside')
        
        st.plotly_chart(fig_region, use_container_width=True)
        
        # Create a treemap for a different visualization of the same data
        fig_treemap = px.treemap(
            year_data,
            path=['Region', 'Country'],
            values='New Cases',
            color='Mortality Rate (%)',
            hover_data=['Average Age of Diagnosis'],
            color_continuous_scale='RdBu_r',
            title=f'Brain Tumor Cases Distribution by Region and Country ({selected_year})'
        )
        
        fig_treemap.update_layout(
            margin=dict(l=0, r=0, t=50, b=0)
        )
        
        st.plotly_chart(fig_treemap, use_container_width=True)
        
        # Add a year-over-year comparison visualization
        st.subheader("Year-over-Year Comparison")
        
        # Select regions for comparison
        all_regions = sorted(global_stats['Region'].unique())
        selected_regions = st.multiselect(
            "Select regions to compare:",
            options=all_regions,
            default=all_regions[:3]  # Default to first 3 regions
        )
        
        if not selected_regions:
            st.warning("Please select at least one region to view the comparison.")
        else:
            # Filter for selected regions
            region_comparison = global_stats[global_stats['Region'].isin(selected_regions)]
            
            # Group by Year and Region
            comparison_data = region_comparison.groupby(['Year', 'Region']).agg({
                'New Cases': 'sum',
                'Mortality Rate (%)': 'mean'
            }).reset_index()
            
            # Line chart for new cases
            fig_trend = px.line(
                comparison_data,
                x="Year", 
                y="New Cases",
                color="Region",
                markers=True,
                title="Brain Tumor Cases Trend by Region (2021-2023)",
                labels={"New Cases": "Number of New Cases", "Year": "Year"},
            )
            
            fig_trend.update_layout(
                xaxis_title="Year",
                yaxis_title="Number of New Cases",
                legend_title="Region"
            )
            
            st.plotly_chart(fig_trend, use_container_width=True)
            
            # Line chart for mortality rates
            fig_mortality = px.line(
                comparison_data,
                x="Year", 
                y="Mortality Rate (%)",
                color="Region",
                markers=True,
                title="Mortality Rate Trend by Region (2021-2023)",
                labels={"Mortality Rate (%)": "Mortality Rate (%)", "Year": "Year"},
            )
            
            fig_mortality.update_layout(
                xaxis_title="Year",
                yaxis_title="Mortality Rate (%)",
                legend_title="Region"
            )
            
            st.plotly_chart(fig_mortality, use_container_width=True)
        
        # Display data table
        st.subheader(f"Country-Level Data Table ({selected_year})")
        st.dataframe(
            year_data[['Country', 'Region', 'New Cases', 'Mortality Rate (%)', 'Average Age of Diagnosis']], 
            use_container_width=True, 
            hide_index=True
        )
        
        # Add trend analysis button
        if st.button("Show Complete Multi-Year Data Table"):
            st.subheader("Complete Multi-Year Data (2021-2023)")
            st.dataframe(
                global_stats.sort_values(['Country', 'Year']),
                use_container_width=True,
                hide_index=True
            )
        
        # Add data source disclaimer
        st.caption("""
        Data Note: These statistics represent data compiled from multiple international cancer registries. 
        While efforts have been made to ensure accuracy, reporting methods vary by country. 
        Trends show an overall increase in detection rates due to improved diagnostic technologies, 
        while mortality rates have generally declined due to advancements in treatment.
        """)

    with stat_tabs[1]:
        # Rest of your age & gender data code remains unchanged
        st.subheader("Age & Gender Distribution")
        
        # Get age and gender data
        age_gender_data = generate_age_gender_distribution()
        
        # Create a grouped bar chart
        fig_age = px.bar(
            age_gender_data,
            x="Age Group",
            y=["Male Incidence", "Female Incidence"],
            title="Brain Tumor Incidence by Age and Gender",
            labels={"value": "Incidence Rate (per 100,000)", "variable": "Gender"},
            barmode="group",
            color_discrete_map={"Male Incidence": "#2C3E50", "Female Incidence": "#E74C3C"}
        )
        
        fig_age.update_layout(
            xaxis_title="Age Group",
            yaxis_title="Incidence Rate (per 100,000 population)",
            legend_title="Gender"
        )
        
        st.plotly_chart(fig_age, use_container_width=True)
        
        # Add information about age and gender trends
        st.markdown("""
        <div class="info-box">
            <h4>Key Insights on Age & Gender Distribution:</h4>
            <ul>
                <li>Brain tumor incidence generally increases with age, with peak rates in the 60-79 age group.</li>
                <li>Certain tumor types show gender preferences: meningiomas are more common in women, while gliomas are slightly more common in men.</li>
                <li>Children under 14 represent about 15% of all brain tumor cases, with different histological profiles than adults.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    with stat_tabs[2]:
        st.subheader("Treatment Success Rates")
        
        # Get treatment success data
        treatment_data = generate_treatment_success_rates()
        
        # Create a grouped bar chart for treatment success rates
        fig_treatment = px.bar(
            treatment_data,
            x="Treatment Method",
            y=["Glioma Success Rate (%)", "Meningioma Success Rate (%)", "Pituitary Tumor Success Rate (%)"],
            title="Treatment Success Rates by Tumor Type",
            labels={"value": "Success Rate (%)", "variable": "Tumor Type"},
            barmode="group",
            color_discrete_map={
                "Glioma Success Rate (%)": "#3498DB", 
                "Meningioma Success Rate (%)": "#2ECC71",
                "Pituitary Tumor Success Rate (%)": "#9B59B6"
            }
        )
        
        fig_treatment.update_layout(
            xaxis_title="Treatment Method",
            yaxis_title="Success Rate (%)",
            legend_title="Tumor Type"
        )
        
        st.plotly_chart(fig_treatment, use_container_width=True)
        
        # Display data table
        st.subheader("Treatment Data Table")
        st.dataframe(treatment_data, use_container_width=True, hide_index=True)

def tumor_progression_page():
    st.header("📈 Tumor Progression Stages")
    
    # Introduction
    st.markdown("""
    <div class="info-card">
        <p>Understanding the stages of brain tumor progression is crucial for treatment planning and prognosis assessment. 
        Each tumor type follows slightly different staging systems based on its biological behavior.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Get tumor progression data
    tumor_stages = create_tumor_progression_stages()
    
    # Create tabs for each tumor type
    tumor_tabs = st.tabs(list(tumor_stages.keys()))
    
    # For each tumor type
    for i, (tumor_type, tab) in enumerate(zip(tumor_stages.keys(), tumor_tabs)):
        with tab:
            stages = tumor_stages[tumor_type]
            
            # Convert to DataFrame for visualization
            stage_df = pd.DataFrame(stages)
            
            # Create two columns layout
            col1, col2 = st.columns([1, 1])
            
            with col1:
                # Display data in a formatted table
                st.markdown(f"### {tumor_type} Tumor Stages")
                st.dataframe(stage_df, use_container_width=True, hide_index=True)
                
                # Add additional information
                if tumor_type == "Glioma":
                    st.markdown("""
                    <div class="info-note">
                        <p><strong>Note:</strong> Gliomas are graded from I to IV based on their cell appearance and growth rate. 
                        Grade IV (Glioblastoma) is the most aggressive form with the poorest prognosis.</p>
                    </div>
                    """, unsafe_allow_html=True)
                elif tumor_type == "Meningioma":
                    st.markdown("""
                    <div class="info-note">
                        <p><strong>Note:</strong> Most meningiomas are benign (WHO Grade I) and slow-growing. 
                        Higher grades indicate more aggressive behavior.</p>
                    </div>
                    """, unsafe_allow_html=True)
                elif tumor_type == "Pituitary":
                    st.markdown("""
                    <div class="info-note">
                        <p><strong>Note:</strong> Pituitary tumors are often classified by their size (microadenomas vs. macroadenomas) 
                        and hormone secretion status rather than traditional staging.</p>
                    </div>
                    """, unsafe_allow_html=True)
            
            with col2:
                # Survival Rate Visualization
                survival_data = []
                for stage in stages:
                    # Extract the lower bound of the survival rate range
                    rate = stage['Survival Rate'].split('-')[0]
                    try:
                        rate_value = float(rate.rstrip('%'))
                    except:
                        rate_value = 0
                    
                    survival_data.append({
                        'Stage': stage['Stage'],
                        'Survival Rate': rate_value,
                        'Description': stage['Description']
                    })
                
                survival_df = pd.DataFrame(survival_data)
                
                # Create a visually appealing bar chart
                colors = ["#2ECC71" if rate >= 80 else "#F1C40F" if rate >= 50 else "#E74C3C" for rate in survival_df['Survival Rate']]
                
                fig = px.bar(
                    survival_df,
                    x="Stage",
                    y="Survival Rate",
                    text="Survival Rate",
                    title=f"{tumor_type} Tumor - Survival Rates by Stage",
                    labels={"Survival Rate": "Survival Rate (%)"},
                    hover_data=["Description"]
                )
                
                fig.update_traces(marker_color=colors, texttemplate='%{y}%', textposition='outside')
                
                fig.update_layout(
                    xaxis_title="Stage",
                    yaxis_title="Survival Rate (%)",
                    yaxis_range=[0, 100]
                )
                
                st.plotly_chart(fig, use_container_width=True)

def risk_assessment_page():
    st.header("🩺 Personalized Risk Assessment")
    
    # Introduction
    st.markdown("""
    <div class="info-card">
        <p>This tool provides a preliminary assessment of brain tumor risk factors based on personal and family history. 
        It is not a diagnostic tool and should not replace professional medical advice.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Get questionnaire data
    questionnaire = create_risk_assessment_questionnaire()
    
    # Create a more appealing form layout
    st.markdown("### Personal Risk Factors Questionnaire")
    
    # Use columns to improve the layout
    col1, col2 = st.columns(2)
    
    # Track responses
    responses = {}
    
    # First column of questions
    with col1:
        for q in questionnaire[:4]:  # First half of questions
            if q['type'] == 'radio':
                responses[q['question']] = st.radio(
                    q['question'], 
                    q['options'],
                    key=f"radio_{q['question']}"
                )
            elif q['type'] == 'multiselect':
                responses[q['question']] = st.multiselect(
                    q['question'], 
                    q['options'],
                    key=f"multiselect_{q['question']}"
                )
    
    # Second column of questions
    with col2:
        for q in questionnaire[4:]:  # Second half of questions
            if q['type'] == 'radio':
                responses[q['question']] = st.radio(
                    q['question'], 
                    q['options'],
                    key=f"radio_{q['question']}"
                )
            elif q['type'] == 'multiselect':
                responses[q['question']] = st.multiselect(
                    q['question'], 
                    q['options'],
                    key=f"multiselect_{q['question']}"
                )
    
    # Calculate Risk button with improved styling
    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("Calculate Risk Assessment", use_container_width=True):
        # Simple risk calculation (for demonstration)
        risk_score = 0
        
        # Family history factor
        if responses.get('Do you have a family history of brain tumors?') == 'Yes':
            risk_score += 2
        
        # Age factor
        age_risk = {
            '0-20': 1,
            '21-40': 2,
            '41-60': 3,
            '61+': 4
        }
        risk_score += age_risk.get(responses.get('Age Group', '0-20'), 1)
        
        # Radiation exposure factor
        if responses.get('Have you been exposed to ionizing radiation?') == 'Yes':
            risk_score += 2
        
        # Mobile phone usage factor
        mobile_risk = {
            'Less than 1': 1,
            '1-3': 2,
            '3-5': 3,
            'More than 5': 4
        }
        risk_score += mobile_risk.get(responses.get('Frequency of mobile phone usage (hours per day)', 'Less than 1'), 1)
        
        # Genetic predispositions factor
        genetic_responses = responses.get('Do you have any genetic predispositions?', [])
        if 'None' not in genetic_responses and len(genetic_responses) > 0:
            risk_score += 2
        
        # Headaches factor
        if responses.get('Have you experienced persistent headaches?') == 'Yes':
            risk_score += 1
        
        # Vision or hearing changes factor
        if responses.get('Have you noticed recent changes in vision or hearing?') == 'Yes':
            risk_score += 1
        
        # Risk categorization
        if risk_score < 5:
            risk_level = "Low"
            advice = "Your risk appears to be low. Continue regular check-ups."
            color = "#2ECC71"  # Green
        elif risk_score < 8:
            risk_level = "Moderate"
            advice = "Consider consulting a healthcare professional for personalized guidance."
            color = "#F1C40F"  # Yellow
        else:
            risk_level = "High"
            advice = "We recommend a comprehensive medical evaluation and consultation."
            color = "#E74C3C"  # Red
        
        # Display results in an attractive format
        st.markdown("<hr/>", unsafe_allow_html=True)
        st.subheader("Risk Assessment Results")
        
        # Create a gauge chart for risk score
        fig = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = risk_score,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Risk Score"},
            gauge = {
                'axis': {'range': [0, 15], 'tickwidth': 1},
                'bar': {'color': color},
                'steps': [
                    {'range': [0, 5], 'color': "#EAFAF1"},  # Light green
                    {'range': [5, 8], 'color': "#FEF9E7"},  # Light yellow
                    {'range': [8, 15], 'color': "#FADBD8"}  # Light red
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': risk_score
                }
            }
        ))
        
        fig.update_layout(
            height=300,
            margin=dict(l=20, r=20, t=50, b=20)
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Display risk level and advice
        st.markdown(f"""
        <div class="result-container">
            <div class="risk-level-card" style="background-color: {color}20; border-left: 5px solid {color};">
                <h3>Risk Level: {risk_level}</h3>
                <p>{advice}</p>
            </div>
            <div class="disclaimer">
                <p><strong>Disclaimer:</strong> This is a simplified risk assessment tool and not a diagnostic test. 
                The results should be interpreted with caution and discussed with healthcare professionals.</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Recommended next steps based on risk level
        st.subheader("Recommended Next Steps")
        
        if risk_level == "Low":
            st.markdown("""
            <ul class="recommendations">
                <li>Continue with regular health check-ups</li>
                <li>Maintain a healthy lifestyle</li>
                <li>Be aware of any new neurological symptoms</li>
                <li>Repeat this assessment if risk factors change</li>
            </ul>
            """, unsafe_allow_html=True)
        elif risk_level == "Moderate":
            st.markdown("""
            <ul class="recommendations">
                <li>Consult with your primary care physician</li>
                <li>Discuss your risk factors and concerns</li>
                <li>Consider a neurological evaluation</li>
                <li>Monitor symptoms closely</li>
                <li>Follow preventive health measures</li>
            </ul>
            """, unsafe_allow_html=True)
        else:  # High
            st.markdown("""
            <ul class="recommendations">
                <li>Seek prompt medical evaluation from a neurologist</li>
                <li>Discuss appropriate screening tests with your doctor</li>
                <li>Be vigilant about monitoring symptoms</li>
                <li>Consider genetic counseling if family history is significant</li>
                <li>Develop a regular monitoring plan with your healthcare provider</li>
            </ul>
            """, unsafe_allow_html=True)

def prevention_page():
    st.header("⚠️ Prevention & Care Guidelines")
    
    # Introduction
    st.markdown("""
    <div class="info-card">
        <p>While many brain tumors cannot be prevented, understanding risk factors and adopting healthy 
        lifestyle choices may help reduce risks. Early detection and proper care are essential for 
        better outcomes.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Create tabs for different sections
    prevention_tabs = st.tabs(["Risk Factors", "Warning Signs", "Lifestyle Recommendations", "Care Guidelines"])
    
    # Risk Factors Tab
    with prevention_tabs[0]:
        # Get risk factors data
        risk_factors = create_risk_factors_data()
        
        st.subheader("Understanding Brain Tumor Risk Factors")
        
        # Create three columns for risk factor categories
        cols = st.columns(3)
        
        # Genetic factors
        with cols[0]:
            st.markdown("""
            <div class="factor-card genetic">
                <h3>🧬 Genetic Factors</h3>
                <ul>
            """, unsafe_allow_html=True)
            
            for factor in risk_factors['Genetic Factors']:
                st.markdown(f"<li>{factor}</li>", unsafe_allow_html=True)
            
            st.markdown("</ul></div>", unsafe_allow_html=True)
        
        # Environmental factors
        with cols[1]:
            st.markdown("""
            <div class="factor-card environmental">
                <h3>🌍 Environmental Factors</h3>
                <ul>
            """, unsafe_allow_html=True)
            
            for factor in risk_factors['Environmental Factors']:
                st.markdown(f"<li>{factor}</li>", unsafe_allow_html=True)
            
            st.markdown("</ul></div>", unsafe_allow_html=True)
        
        # Lifestyle factors
        with cols[2]:
            st.markdown("""
            <div class="factor-card lifestyle">
                <h3>🏃 Lifestyle Factors</h3>
                <ul>
            """, unsafe_allow_html=True)
            
            for factor in risk_factors['Lifestyle Factors']:
                st.markdown(f"<li>{factor}</li>", unsafe_allow_html=True)
            
            st.markdown("</ul></div>", unsafe_allow_html=True)
        
        # Additional information
        st.markdown("""
        <div class="note-card">
            <h4>Important to Remember:</h4>
            <p>The presence of risk factors does not guarantee that a person will develop a brain tumor.
            Many people with multiple risk factors never develop tumors, while others with no known
            risk factors do. Research into the causes of brain tumors is ongoing.</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Warning Signs Tab
    with prevention_tabs[1]:
        # Get symptoms data
        symptoms = create_symptoms_data()
        
        st.subheader("Recognizing Warning Signs")
        
        # Introduction
        st.markdown("""
        <p>Brain tumor symptoms vary greatly depending on tumor type, size, and location. 
        Many symptoms develop gradually and may be subtle at first. Being aware of potential 
        warning signs can lead to earlier detection.</p>
        """, unsafe_allow_html=True)
        
        # Display symptoms by category
        col1, col2 = st.columns(2)
        
        # Common symptoms
        with col1:
            st.markdown("""
            <div class="symptoms-card">
                <h3>Common Symptoms</h3>
                <ul>
            """, unsafe_allow_html=True)
            
            for symptom in symptoms['Common']:
                st.markdown(f"<li>{symptom}</li>", unsafe_allow_html=True)
            
            st.markdown("</ul></div>", unsafe_allow_html=True)
            
            # Cognitive symptoms
            st.markdown("""
            <div class="symptoms-card">
                <h3>Cognitive Symptoms</h3>
                <ul>
            """, unsafe_allow_html=True)
            
            for symptom in symptoms['Cognitive']:
                st.markdown(f"<li>{symptom}</li>", unsafe_allow_html=True)
            
            st.markdown("</ul></div>", unsafe_allow_html=True)
        
        # Neurological symptoms
        with col2:
            st.markdown("""
            <div class="symptoms-card">
                <h3>Neurological Symptoms</h3>
                <ul>
            """, unsafe_allow_html=True)
            
            for symptom in symptoms['Neurological']:
                st.markdown(f"<li>{symptom}</li>", unsafe_allow_html=True)
            
            st.markdown("</ul></div>", unsafe_allow_html=True)
            
            # When to see a doctor
            st.markdown("""
            <div class="symptoms-card warning">
                <h3>When to See a Doctor</h3>
                <ul>
            """, unsafe_allow_html=True)
            
            for warning in symptoms['When to See a Doctor']:
                st.markdown(f"<li>{warning}</li>", unsafe_allow_html=True)
            
            st.markdown("</ul></div>", unsafe_allow_html=True)
    
    # Lifestyle Recommendations Tab
    with prevention_tabs[2]:
        st.subheader("Healthy Lifestyle Recommendations")
        
        # Create a clean layout with icons
        recommendations = [
            {
                "icon": "🥗",
                "title": "Nutritious Diet",
                "desc": "Emphasize fruits, vegetables, whole grains, and lean proteins. Limit processed foods and excessive sugar intake. Antioxidant-rich foods may have protective benefits."
            },
            {
                "icon": "🏃‍♀️",
                "title": "Regular Exercise",
                "desc": "Aim for at least 150 minutes of moderate activity per week. Regular physical activity supports overall brain health and improves immune function."
            },
            {
                "icon": "💤",
                "title": "Quality Sleep",
                "desc": "Prioritize 7-9 hours of quality sleep daily. Sleep is essential for cellular repair and overall brain function."
            },
            {
                "icon": "🧠",
                "title": "Mental Stimulation",
                "desc": "Engage in activities that challenge your brain, such as puzzles, reading, learning new skills, or playing musical instruments."
            },
            {
                "icon": "🚭",
                "title": "Avoid Tobacco",
                "desc": "Don't smoke or use tobacco products. If you currently smoke, seek help to quit as soon as possible."
            },
            {
                "icon": "🍷",
                "title": "Limit Alcohol",
                "desc": "If you drink alcohol, do so in moderation (up to 1 drink per day for women and up to 2 for men)."
            },
            {
                "icon": "☀️",
                "title": "Sun Protection",
                "desc": "Use sunscreen and protective clothing to reduce radiation exposure, especially during peak hours."
            },
            {
                "icon": "📱",
                "title": "Limited Screen Time",
                "desc": "Consider using speakerphone or headphones when making calls. Avoid carrying your phone in pockets close to your body. While research is inconclusive, these precautions may be beneficial."
            }
        ]
        
        # Display recommendations in a grid
        cols = st.columns(2)
        half = len(recommendations) // 2
        
        for i, rec in enumerate(recommendations[:half]):
            with cols[0]:
                st.markdown(f"""
                <div class="recommendation-card">
                    <div class="rec-icon">{rec['icon']}</div>
                    <div class="rec-content">
                        <h3>{rec['title']}</h3>
                        <p>{rec['desc']}</p>
                    </div>
                </div>
                """, unsafe_allow_html=True)
        
        for i, rec in enumerate(recommendations[half:]):
            with cols[1]:
                st.markdown(f"""
                <div class="recommendation-card">
                    <div class="rec-icon">{rec['icon']}</div>
                    <div class="rec-content">
                        <h3>{rec['title']}</h3>
                        <p>{rec['desc']}</p>
                    </div>
                </div>
                """, unsafe_allow_html=True)
    
    # Care Guidelines Tab
    with prevention_tabs[3]:
        st.subheader("Treatment and Care Guidelines")
        
        # Get treatment options data
        treatment_options = create_treatment_options()
        
        # Introduction
        st.markdown("""
        <div class="care-intro">
            <p>Brain tumor treatment approaches vary based on tumor type, size, location, and the patient's overall health.
            Treatment often involves a multidisciplinary team of specialists working together to create the most effective plan.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Display treatment options in an accordion
        for treatment, details in treatment_options.items():
            with st.expander(f"{treatment}"):
                st.markdown(f"""
                <div class="treatment-details">
                    <p><strong>Description:</strong> {details['Description']}</p>
                    <p><strong>When Used:</strong> {details['When Used']}</p>
                    <p><strong>Benefits:</strong> {details['Benefits']}</p>
                    <p><strong>Risks:</strong> {details['Risks']}</p>
                </div>
                """, unsafe_allow_html=True)
        
        # Post-treatment care
        st.subheader("Post-Treatment Care")
        
        st.markdown("""
        <div class="post-care-card">
            <h4>Follow-up & Monitoring</h4>
            <p>Regular follow-up appointments with your healthcare team are essential for monitoring tumor status,
            evaluating treatment effectiveness, and addressing any side effects or complications.</p>
            <h4>Rehabilitation</h4>
            <p>Many patients benefit from various forms of rehabilitation, including:</p>
            <ul>
                <li><strong>Physical therapy</strong> to restore strength and mobility</li>
                <li><strong>Occupational therapy</strong> to improve daily functioning</li>
                <li><strong>Speech therapy</strong> to address communication difficulties</li>
                <li><strong>Neuropsychological services</strong> to manage cognitive changes</li>
            </ul>
            <h4>Supportive Care</h4>
            <p>Support groups, counseling, and mental health services can help patients and families cope
            with the emotional and psychological impacts of diagnosis and treatment.</p>
            <h4>Lifestyle Adjustments</h4>
            <p>Maintaining a healthy lifestyle through proper nutrition, regular exercise, stress management, 
            and sufficient sleep can support recovery and overall wellbeing during and after treatment.</p>
        </div>
        """, unsafe_allow_html=True)

def tumor_types_page():
    st.header("🩺 Brain Tumor Types")
    
    # Get tumor details
    tumor_details = create_tumor_details()
    
    # Introduction
    st.markdown("""
    <div class="info-card">
        <p>Brain tumors are classified based on their cell origin, behavior, and location. Understanding the 
        characteristics of different tumor types can help in grasping treatment options and prognosis.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Create tabs for different tumor types
    tumor_tabs = st.tabs(list(tumor_details.keys()))
    
    # For each tumor type
    for i, (tumor_name, tab) in enumerate(zip(tumor_details.keys(), tumor_tabs)):
        with tab:
            info = tumor_details[tumor_name]
            
            # Two-column layout
            col1, col2 = st.columns([2, 1])
            
            with col1:
                # Display the tumor name and description without HTML
                st.subheader(tumor_name)
                st.write(info['Description'])
                
                # Display characteristics using native Streamlit components
                st.write("**Key Characteristics:**")
                for char in info['Characteristics']:
                    st.write(f"• {char}")
                
                # Display symptoms using native Streamlit components
                st.write("**Common Symptoms:**")
                for symptom in info['Common Symptoms']:
                    st.write(f"• {symptom}")
                
                # Display treatment options using native Streamlit components
                st.write("**Treatment Options:**")
                for treatment in info['Treatment Options']:
                    st.write(f"• {treatment}")
            
            with col2:
                # Use static placeholders with brain-themed images
                if tumor_name == "Glioma Tumor":
                    # Create colored box with information
                    st.markdown("""
                    <div style="background-color: #f8d7da; border: 1px solid #f5c6cb; border-radius: 10px; padding: 15px; margin-bottom: 20px;">
                        <div style="text-align: center; margin-bottom: 10px;">
                            <span style="font-size: 48px;">🧠</span>
                        </div>
                        <h4 style="text-align: center; margin-bottom: 10px;">Glioma Visualization</h4>
                        <p style="font-style: italic; text-align: center;">
                            Gliomas develop from glial cells and typically form within the substance of the brain.
                            They vary in aggressiveness from slow-growing (low-grade) to rapidly growing (high-grade).
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
                elif tumor_name == "Meningioma Tumor":
                    st.markdown("""
                    <div style="background-color: #d1ecf1; border: 1px solid #bee5eb; border-radius: 10px; padding: 15px; margin-bottom: 20px;">
                        <div style="text-align: center; margin-bottom: 10px;">
                            <span style="font-size: 48px;">🧠</span>
                        </div>
                        <h4 style="text-align: center; margin-bottom: 10px;">Meningioma Visualization</h4>
                        <p style="font-style: italic; text-align: center;">
                            Meningiomas arise from the meninges, the membranes that surround the brain and spinal cord.
                            They typically grow inward, pressing on the brain, spinal cord, or nerves.
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
                elif tumor_name == "Pituitary Tumor":
                    st.markdown("""
                    <div style="background-color: #e2e3f9; border: 1px solid #d6d8f0; border-radius: 10px; padding: 15px; margin-bottom: 20px;">
                        <div style="text-align: center; margin-bottom: 10px;">
                            <span style="font-size: 48px;">🧠</span>
                        </div>
                        <h4 style="text-align: center; margin-bottom: 10px;">Pituitary Tumor Visualization</h4>
                        <p style="font-style: italic; text-align: center;">
                            Pituitary tumors develop in the pituitary gland at the base of the brain.
                            They can affect hormone production and sometimes press on nearby structures, including the optic nerves.
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Add a "Did you know?" box using native Streamlit
                st.markdown("### Did you know?")
                if tumor_name == "Glioma Tumor":
                    st.info("Glioblastoma is the most aggressive type of glioma (grade IV) and accounts for about 15% of all brain tumors.")
                elif tumor_name == "Meningioma Tumor":
                    st.info("Meningiomas are about twice as common in women as in men, suggesting potential hormonal influences on tumor development.")
                elif tumor_name == "Pituitary Tumor":
                    st.info("Though pituitary tumors are found in about 10% of brain scans, many are never diagnosed because they don't cause symptoms.")
                    
                # Add specific links to learn more about each tumor type
                if tumor_name == "Glioma Tumor":
                    st.markdown("[Learn more about Glioma Tumors](https://www.hopkinsmedicine.org/health/conditions-and-diseases/gliomas)")
                elif tumor_name == "Meningioma Tumor":
                    st.markdown("[Learn more about Meningioma Tumors](https://www.mayoclinic.org/diseases-conditions/meningioma/symptoms-causes/syc-20355643)")
                elif tumor_name == "Pituitary Tumor":
                    st.markdown("[Learn more about Pituitary Tumors](https://www.mayoclinic.org/diseases-conditions/pituitary-tumors/symptoms-causes/syc-20350548)")
def survival_statistics_page():
    st.header("📊 Brain Tumor Survival Statistics")
    
    # Introduction
    st.markdown("""
    <div class="info-card">
        <p>Survival rates for brain tumors vary widely based on tumor type, grade, location, and patient factors. 
        The data presented here provides general insights into survival trends over recent years.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Get survival data
    df = create_survival_data()
    
    # Create tabs for different views
    survival_tabs = st.tabs(["Survival Trends", "Treatment Success", "Comparative Analysis"])
    
    with survival_tabs[0]:
        st.subheader("Survival Rate Progression by Tumor Type")
        
        # Create a line chart with improved styling
        fig = px.line(
            df, 
            x='Tumor Type', 
            y=['2018 Survival Rate (%)', '2022 Survival Rate (%)', '2024 Survival Rate (%)'],
            title='5-Year Survival Rate Trends (2018-2024)',
            labels={'value': 'Survival Rate (%)', 'variable': 'Year'},
            height=500,
            markers=True,
            line_shape="spline",
            color_discrete_sequence=["#3498DB", "#2ECC71", "#9B59B6"]
        )
        
        fig.update_layout(
            xaxis_title="Tumor Type",
            yaxis_title="5-Year Survival Rate (%)",
            legend_title="Year",
            yaxis_range=[0, 105],
            hovermode="x unified"
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Add context and interpretation
        st.markdown("""
        <div class="interpretation-card">
            <h4>Key Observations:</h4>
            <ul>
                <li>Overall survival rates for all tumor types have improved over the past 6 years</li>
                <li>The most significant improvements are seen in glioma tumors (20% increase)</li>
                <li>Meningioma and pituitary tumors continue to have favorable survival outcomes</li>
                <li>Improvements can be attributed to advances in surgical techniques, targeted therapies, and multidisciplinary care approaches</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with survival_tabs[1]:
        st.subheader("Treatment Success Rates")
        
        # Get treatment success data
        treatment_data = generate_treatment_success_rates()
        
        # Create a heatmap for treatment success
        fig_heatmap = px.imshow(
            treatment_data[['Glioma Success Rate (%)', 'Meningioma Success Rate (%)', 'Pituitary Tumor Success Rate (%)']].values,
            x=['Glioma', 'Meningioma', 'Pituitary'],
            y=treatment_data['Treatment Method'],
            color_continuous_scale='Viridis',
            labels=dict(x="Tumor Type", y="Treatment Method", color="Success Rate (%)"),
            title="Treatment Success Rates by Method and Tumor Type"
        )
        
        fig_heatmap.update_layout(
            height=500,
            xaxis_title="Tumor Type",
            yaxis_title="Treatment Method",
            coloraxis_colorbar=dict(title="Success Rate (%)")
        )
        
        st.plotly_chart(fig_heatmap, use_container_width=True)
        
        # Add context
        st.markdown("""
        <div class="interpretation-card">
            <h4>Treatment Effectiveness:</h4>
            <p>The heatmap above illustrates the varying effectiveness of different treatment approaches across tumor types:</p>
            <ul>
                <li><strong>Combined Therapy</strong> consistently shows the highest success rates across all tumor types</li>
                <li><strong>Surgery</strong> is particularly effective for meningiomas and pituitary tumors</li>
                <li><strong>Radiation Therapy</strong> shows moderate effectiveness across all types</li>
                <li><strong>Chemotherapy</strong> has limited effectiveness when used alone, particularly for meningiomas</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with survival_tabs[2]:
        st.subheader("Comparative Survival Analysis")
        
        # Create a radar chart comparing current survival rates and factors
        data = {
            'Metrics': ['5-Year Survival', 'Treatment Response', 'Quality of Life', 'Recurrence Freedom', 'Symptom Management'],
            'Glioma': [65, 55, 60, 50, 70],
            'Meningioma': [90, 85, 80, 75, 85],
            'Pituitary': [95, 90, 85, 80, 90]
        }
        
        metrics = data['Metrics']
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatterpolar(
            r=data['Glioma'],
            theta=metrics,
            fill='toself',
            name='Glioma'
        ))
        
        fig.add_trace(go.Scatterpolar(
            r=data['Meningioma'],
            theta=metrics,
            fill='toself',
            name='Meningioma'
        ))
        
        fig.add_trace(go.Scatterpolar(
            r=data['Pituitary'],
            theta=metrics,
            fill='toself',
            name='Pituitary'
        ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 100]
                )
            ),
            title="Comprehensive Outcome Comparison by Tumor Type",
            height=600,
            showlegend=True
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Add explanation of metrics
        st.markdown("""
        <div class="metrics-explanation">
            <h4>Metrics Explained:</h4>
            <ul>
                <li><strong>5-Year Survival:</strong> Percentage of patients surviving 5 years after diagnosis</li>
                <li><strong>Treatment Response:</strong> Percentage of patients showing positive response to initial treatment</li>
                <li><strong>Quality of Life:</strong> Average patient-reported quality of life after treatment (0-100 scale)</li>
                <li><strong>Recurrence Freedom:</strong> Percentage of patients without tumor recurrence within 5 years</li>
                <li><strong>Symptom Management:</strong> Effectiveness of controlling tumor-related symptoms (0-100 scale)</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

def about_dataset_page():
    st.header("📚 About the Dataset")
    
    # Introduction
    st.markdown("""
    <div class="info-card">
        <p>The brain tumor detection model was developed using a comprehensive dataset of MRI scans, 
        carefully curated to ensure accurate and reliable classification results.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Dataset overview
    st.subheader("Dataset Overview")
    
    st.markdown("""
    The dataset used for this model is sourced from the Brain Tumor MRI Dataset available on Kaggle, 
    comprising 7,023 Magnetic Resonance Imaging (MRI) scans. These images are categorized into four 
    classifications:
    
    - Glioma tumors
    - Meningioma tumors
    - Pituitary tumors
    - Images with no tumor
    """)
    
    # Dataset distribution table
    st.markdown("### Dataset Distribution")
    
    dataset_distribution = {
        'Tumor Type': ['Glioma tumor', 'Meningioma tumor', 'Pituitary tumor', 'No tumor', 'Total'],
        'Training': [1060, 1072, 1158, 1279, 4569],
        'Validation': [261, 267, 299, 316, 1143],
        'Testing': [300, 306, 300, 405, 1311],
        'Total': [1621, 1645, 1757, 2000, 7023]
    }
    
    df_distribution = pd.DataFrame(dataset_distribution)
    st.dataframe(df_distribution, use_container_width=True, hide_index=True)
    
    # Visualize dataset distribution
    st.subheader("Dataset Visualization")
    
    # Prepare data for visualization
    visualization_data = []
    for i, tumor_type in enumerate(df_distribution['Tumor Type'][:-1]):  # Exclude the "Total" row
        for split in ['Training', 'Validation', 'Testing']:
            visualization_data.append({
                'Tumor Type': tumor_type,
                'Split': split,
                'Count': df_distribution[split][i]
            })
    
    df_viz = pd.DataFrame(visualization_data)
    
    # Create a grouped bar chart
    fig = px.bar(
        df_viz,
        x='Tumor Type',
        y='Count',
        color='Split',
        title='Distribution of MRI Images by Tumor Type and Dataset Split',
        barmode='group',
        color_discrete_sequence=["#3498DB", "#2ECC71", "#E74C3C"]
    )
    
    fig.update_layout(
        xaxis_title="Tumor Type",
        yaxis_title="Number of Images",
        legend_title="Dataset Split"
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Display sample images
    st.subheader("Sample MRI Images")
    
    st.markdown("""
    <div class="sample-images-note">
        <p>The dataset contains preprocessed MRI scans showing different types of brain tumors, as well as normal brain scans (no tumor). 
        Below are sample images representing each category in the dataset:</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Alternative implementation if you prefer to use a single full image
    st.image("brainTi.png", caption="Sample MRI Scans: No Tumor, Glioma, Meningioma, and Pituitary Tumor", use_container_width=True)
    
    # Add a note about the images
    st.markdown("""
    <div style="background-color: #f8f9fa; padding: 10px; border-radius: 5px; margin-top: 20px; border-left: 3px solid #3498db;">
        <p style="margin: 0;"><strong>Note:</strong> These sample MRI images demonstrate the distinctive visual characteristics of different brain tumor types. 
        The model has been trained to recognize these patterns and differentiate between various tumor types and normal brain scans.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Model information
    st.subheader("Model Architecture")
    
    st.markdown("""
    The brain tumor detection model uses a deep learning architecture based on ResNet, trained with the following specifications:
    
    - **Architecture:** Modified ResNet
    - **Input Size:** 200 x 200 pixels
    - **Training Epochs:** 25
    - **Batch Size:** 8
    - **Optimizer:** Adam
    - **Loss Function:** Categorical Cross-Entropy
    - **Data Augmentation:** Rotation, zoom, shift, and flip to improve generalization
    
    The model achieved 95% accuracy on the test dataset, with high precision and recall across all tumor categories.
    """)
    
    # Add a note about GradCAM
    st.markdown("""
    <div class="info-box">
        <h4>About GradCAM Visualization</h4>
        <p>Gradient-weighted Class Activation Mapping (GradCAM) is a technique used to visualize 
        important regions in the input image that influenced the model's decision. The heatmap highlights 
        areas that were most significant in the model's classification process, providing transparency 
        and interpretability to the AI's decision-making.</p>
    </div>
    """, unsafe_allow_html=True)

def create_download_link(val, filename):
    """
    Creates a download link for a given file value
    """
    b64 = base64.b64encode(val)
    return f'<a href="data:application/octet-stream;base64,{b64.decode()}" download="{filename}">Download {filename}</a>'

def export_pdf_report(predicted_class, confidence, original_img, gradcam_heatmap=None, overlayed_img=None):
    """
    Creates a PDF report with images using reportlab
    """
    from reportlab.lib.pagesizes import letter
    from reportlab.lib import colors
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as ReportLabImage
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from io import BytesIO
    from datetime import datetime
    from PIL import Image as PILImage
    import numpy as np
    import tempfile
    import os

    # Create a file-like buffer to receive PDF data
    buffer = BytesIO()
    
    # Create the PDF object using ReportLab
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    elements = []
    
    # Get styles
    styles = getSampleStyleSheet()
    title_style = styles['Title']
    heading_style = styles['Heading2']
    normal_style = styles['Normal']
    
    # Add title
    elements.append(Paragraph('Brain Tumor Analysis Report', title_style))
    elements.append(Spacer(1, 0.25*inch))
    
    # Add date
    date_text = f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    elements.append(Paragraph(date_text, normal_style))
    elements.append(Spacer(1, 0.25*inch))
    
    # Add diagnosis results
    elements.append(Paragraph('Diagnosis Results:', heading_style))
    elements.append(Spacer(1, 0.1*inch))
    
    # Format the tumor type nicely
    tumor_type = predicted_class.replace('_', ' ').title()
    
    # Create diagnosis table
    risk_level = "Low" if predicted_class == "no_tumor" else "High" if predicted_class == "glioma_tumor" else "Moderate"
    diagnosis_data = [
        ['Tumor Type:', tumor_type],
        ['Confidence Score:', f"{confidence}%"],
        ['Risk Level:', risk_level]
    ]
    
    diagnosis_table = Table(diagnosis_data, colWidths=[2*inch, 3.5*inch])
    diagnosis_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (0, -1), colors.lightgrey),
        ('TEXTCOLOR', (0, 0), (0, -1), colors.black),
        ('ALIGN', (0, 0), (0, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ('BACKGROUND', (1, 0), (-1, -1), colors.white),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    
    elements.append(diagnosis_table)
    elements.append(Spacer(1, 0.25*inch))
    
    # Add recommended actions
    elements.append(Paragraph('Recommended Actions:', heading_style))
    elements.append(Spacer(1, 0.1*inch))
    
    next_steps = {
        "no_tumor": "No immediate action required. Regular check-ups recommended.",
        "glioma_tumor": "Urgent consultation with a neurologist. Further diagnostic tests needed.",
        "meningioma_tumor": "Follow-up with a neurologist. Additional imaging may be required.",
        "pituitary_tumor": "Consult with an endocrinologist and neurosurgeon. Hormone tests recommended."
    }
    
    recommendation = next_steps.get(predicted_class, "Consult with a healthcare professional.")
    elements.append(Paragraph(recommendation, normal_style))
    elements.append(Spacer(1, 0.25*inch))
    
    # Add MRI Scan Images section
    elements.append(Paragraph('MRI Scan Analysis Images:', heading_style))
    elements.append(Spacer(1, 0.1*inch))
    
    # Create a temporary directory for image files
    temp_dir = tempfile.mkdtemp()
    image_files = []
    
    try:
        # Process original image
        if original_img is not None:
            # Create a table with all images
            elements.append(Paragraph('<b>Original MRI Scan:</b>', normal_style))
            elements.append(Spacer(1, 0.1*inch))
            
            # Save original image to temp file
            original_path = os.path.join(temp_dir, 'original.jpg')
            
            # Convert to PIL Image if needed
            if not isinstance(original_img, PILImage.Image):
                if isinstance(original_img, np.ndarray):
                    pil_img = PILImage.fromarray(original_img)
                else:
                    pil_img = original_img
            else:
                pil_img = original_img
                
            # Save and add to list of temp files
            pil_img.save(original_path)
            image_files.append(original_path)
            
            # Add to PDF
            img = ReportLabImage(original_path, width=3*inch, height=3*inch)
            elements.append(img)
            elements.append(Spacer(1, 0.2*inch))
        
        # Process overlay image if available
        if overlayed_img is not None:
            elements.append(Paragraph('<b>GradCAM Overlay (Tumor Localization):</b>', normal_style))
            elements.append(Spacer(1, 0.1*inch))
            
            # Save overlay image to temp file
            overlay_path = os.path.join(temp_dir, 'overlay.jpg')
            
            # Convert to PIL Image if needed
            if not isinstance(overlayed_img, PILImage.Image):
                if isinstance(overlayed_img, np.ndarray):
                    pil_img = PILImage.fromarray(overlayed_img)
                else:
                    pil_img = overlayed_img
            else:
                pil_img = overlayed_img
                
            # Save and add to list of temp files
            pil_img.save(overlay_path)
            image_files.append(overlay_path)
            
            # Add to PDF
            img = ReportLabImage(overlay_path, width=3*inch, height=3*inch)
            elements.append(img)
            elements.append(Spacer(1, 0.2*inch))
            
        # Add GradCAM explanation
        elements.append(Paragraph('<b>GradCAM Interpretation:</b>', normal_style))
        elements.append(Spacer(1, 0.05*inch))
        elements.append(Paragraph('The heatmap highlights areas that influenced the AI\'s decision. Red areas indicate regions that strongly influenced the model\'s prediction.', normal_style))
        elements.append(Spacer(1, 0.25*inch))
    except Exception as e:
        # If there's any error with images, add an error message
        elements.append(Paragraph(f"Error including images: {str(e)}", normal_style))
        elements.append(Spacer(1, 0.1*inch))
    
    # Add additional information based on tumor type
    if predicted_class != "no_tumor":
        elements.append(Paragraph(f'About {tumor_type}:', heading_style))
        elements.append(Spacer(1, 0.1*inch))
        
        # Get the tumor details
        tumor_key = tumor_type
        tumor_info = create_tumor_details().get(tumor_key, {})
        
        if tumor_info:
            # Description
            description_text = tumor_info.get('Description', 'No description available')
            elements.append(Paragraph(f"<b>Description:</b> {description_text}", normal_style))
            elements.append(Spacer(1, 0.1*inch))
            
            # Characteristics
            elements.append(Paragraph("<b>Key Characteristics:</b>", normal_style))
            elements.append(Spacer(1, 0.05*inch))
            
            for char in tumor_info.get('Characteristics', ['Information not available']):
                elements.append(Paragraph(f"• {char}", normal_style))
            
            elements.append(Spacer(1, 0.1*inch))
            
            # Common Symptoms
            elements.append(Paragraph("<b>Common Symptoms:</b>", normal_style))
            elements.append(Spacer(1, 0.05*inch))
            
            for symptom in tumor_info.get('Common Symptoms', ['Information not available']):
                elements.append(Paragraph(f"• {symptom}", normal_style))
            
            elements.append(Spacer(1, 0.1*inch))
            
            # Treatment Options
            elements.append(Paragraph("<b>Treatment Options:</b>", normal_style))
            elements.append(Spacer(1, 0.05*inch))
            
            for treatment in tumor_info.get('Treatment Options', ['Consult with a healthcare professional']):
                elements.append(Paragraph(f"• {treatment}", normal_style))
    
    elements.append(Spacer(1, 0.25*inch))
    
    # Add disclaimer
    disclaimer_style = ParagraphStyle(
        'Disclaimer',
        parent=normal_style,
        fontName='Helvetica-Oblique',
        fontSize=8,
    )
    
    disclaimer_text = "Note: This report is generated by an AI-based brain tumor detection system and should be reviewed by a qualified healthcare professional. The results should not be considered as a final diagnosis."
    elements.append(Paragraph(disclaimer_text, disclaimer_style))
    
    # Build PDF
    doc.build(elements)
    
    # Clean up temporary files
    try:
        for file in image_files:
            if os.path.exists(file):
                os.remove(file)
        os.rmdir(temp_dir)
    except:
        pass
    
    # Get the PDF value from the buffer
    pdf_value = buffer.getvalue()
    buffer.close()
    
    return pdf_value

def export_csv_report(predicted_class, confidence):
    """
    Creates a CSV report of the tumor detection results
    
    Args:
        predicted_class: The predicted tumor class
        confidence: Confidence score of the prediction
        
    Returns:
        CSV bytes to be downloaded
    """
    # Create a dataframe with the results
    risk_level = "Low" if predicted_class == "no_tumor" else "High" if predicted_class == "glioma_tumor" else "Moderate"
    next_steps = {
        "no_tumor": "No immediate action required. Regular check-ups recommended.",
        "glioma_tumor": "Urgent consultation with a neurologist. Further diagnostic tests needed.",
        "meningioma_tumor": "Follow-up with a neurologist. Additional imaging may be required.",
        "pituitary_tumor": "Consult with an endocrinologist and neurosurgeon. Hormone tests recommended."
    }
    
    data = {
        'Analysis Date': [datetime.now().strftime("%Y-%m-%d %H:%M:%S")],
        'Tumor Type': [predicted_class.replace('_', ' ').title()],
        'Confidence Score (%)': [confidence],
        'Risk Level': [risk_level],
        'Recommended Action': [next_steps.get(predicted_class, "Consult with a healthcare professional.")]
    }
    
    df = pd.DataFrame(data)
    
    # Convert to CSV
    csv = df.to_csv(index=False)
    
    return csv.encode()

# This function needs to be added to your tumor_detection_page() function,
# right after you display the results
def add_export_options(predicted_class, confidence, original_img, gradcam_heatmap=None, overlayed_img=None):
    """
    Adds export options to the tumor detection results page
    """
    st.markdown("---")
    st.subheader("📊 Export Analysis Results")
    
    export_col1, export_col2 = st.columns(2)
    
    with export_col1:
        st.info("Export a detailed PDF report with images and recommendations")
        if st.button("📄 Generate PDF Report", use_container_width=True):
            pdf_bytes = export_pdf_report(
                predicted_class, 
                confidence, 
                original_img, 
                gradcam_heatmap, 
                overlayed_img
            )
            
            # Create a download link
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            download_filename = f"brain_tumor_analysis_{timestamp}.pdf"
            st.markdown(
                create_download_link(pdf_bytes, download_filename),
                unsafe_allow_html=True
            )
    
    with export_col2:
        st.info("Export a simple CSV file with analysis results")
        if st.button("📊 Generate CSV Report", use_container_width=True):
            csv_bytes = export_csv_report(predicted_class, confidence)
            
            # Create a download link
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            download_filename = f"brain_tumor_analysis_{timestamp}.csv"
            st.markdown(
                create_download_link(csv_bytes, download_filename),
                unsafe_allow_html=True
            )


def faq_page():
    st.header("❓ Frequently Asked Questions")
    
    st.markdown("""
    <div class="info-card">
        <p>Find answers to common questions about brain tumors, MRI scans, and our AI detection system.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Create FAQ categories with expanders
    st.subheader("General Questions About Brain Tumors")
    
    with st.expander("What is a brain tumor?"):
        st.markdown("""
        A brain tumor is a mass or growth of abnormal cells in the brain. Brain tumors can be benign (non-cancerous) 
        or malignant (cancerous). Benign tumors typically grow slowly and don't spread to other parts of the body, 
        while malignant tumors can grow rapidly and are more likely to invade surrounding tissues.
        
        Brain tumors are classified based on:
        - Where they originated (primary brain tumors start in the brain; secondary tumors spread from elsewhere)
        - The type of tissue involved
        - Whether they are benign or malignant
        - Their grade (which indicates how abnormal the cells look and how likely they are to grow and spread)
        """)
    
    with st.expander("What are the common types of brain tumors?"):
        st.markdown("""
        Common types of brain tumors include:
        
        **Gliomas**: These tumors begin in the glial cells of the brain, which support and nourish neurons.
        Types of gliomas include:
        - Astrocytomas (including glioblastomas)
        - Oligodendrogliomas
        - Ependymomas
        
        **Meningiomas**: These tumors form in the meninges, the membranes that surround the brain and spinal cord.
        Most meningiomas are benign, but they can still cause serious symptoms depending on their location.
        
        **Pituitary Tumors**: These develop in the pituitary gland at the base of the brain. They can affect 
        hormone production and sometimes press on the optic nerves.
        
        **Other Types**: Include medulloblastomas, craniopharyngiomas, lymphomas, schwannomas, and more.
        """)
    
    with st.expander("What are the symptoms of a brain tumor?"):
        st.markdown("""
        Symptoms of brain tumors vary greatly depending on the tumor's size, location, and growth rate. 
        Common symptoms may include:
        
        - Headaches that gradually become more frequent and severe
        - Unexplained nausea or vomiting
        - Vision problems, such as blurred vision, double vision, or loss of peripheral vision
        - Loss of sensation or movement in an arm or leg
        - Balance difficulties or problems with walking
        - Speech difficulties
        - Confusion in everyday matters
        - Seizures, especially in someone who doesn't have a history of seizures
        - Hearing problems
        - Changes in personality or behavior
        
        It's important to note that these symptoms can also be caused by many other conditions. 
        Having one or more of these symptoms doesn't necessarily mean you have a brain tumor, 
        but it's important to consult with a healthcare provider for proper evaluation.
        """)
    
    with st.expander("Are brain tumors common?"):
        st.markdown("""
        Brain tumors are relatively uncommon compared to many other types of cancer. 
        According to global statistics:
        
        - The global incidence rate of brain tumors is about 3.5 per 100,000 people
        - Brain tumors account for about 2% of all cancers
        - Approximately 300,000 people worldwide are diagnosed with a primary brain tumor each year
        - Meningiomas are the most common type of primary brain tumor, accounting for about 37% of cases
        - Gliomas represent about 27% of all brain tumors and 80% of malignant brain tumors
        
        The risk of developing a brain tumor increases with age, with most diagnosed in people over 55, 
        but certain types can affect children and younger adults.
        """)
    
    st.subheader("About MRI Scans and Diagnosis")
    
    with st.expander("What is an MRI scan and how does it detect brain tumors?"):
        st.markdown("""
        Magnetic Resonance Imaging (MRI) is a non-invasive imaging technique that uses a strong magnetic field 
        and radio waves to create detailed images of the organs and tissues within the body. MRI is particularly 
        useful for examining the brain because it provides high-resolution images of soft tissues.
        
        MRI scans can detect brain tumors by revealing:
        - Abnormal masses or growths in the brain tissue
        - Changes in brain structure or anatomy
        - Areas of increased blood flow that may indicate a tumor
        - Contrast enhancement patterns (when contrast dye is used) that help identify tumors
        
        During an MRI scan, multiple images are taken from different angles, allowing radiologists and 
        neurologists to examine the brain in detail. Different types of MRI sequences (T1-weighted, T2-weighted, 
        FLAIR, etc.) can highlight different aspects of the brain and potential tumors.
        
        MRI doesn't use radiation like CT scans do, making it a preferred tool for diagnosing brain tumors, 
        especially when multiple scans are needed to monitor a condition over time.
        """)
    
    with st.expander("How accurate is MRI in detecting brain tumors?"):
        st.markdown("""
        MRI is highly accurate in detecting brain tumors and is considered the gold standard imaging technique 
        for brain tumor diagnosis. However, its accuracy depends on several factors:
        
        **Strengths of MRI for brain tumor detection:**
        - High sensitivity (85-95%) for detecting brain tumors
        - Excellent for defining the anatomical location and extent of tumors
        - Can distinguish between tumor types based on imaging characteristics
        - Very effective at detecting small tumors (as small as a few millimeters)
        
        **Limitations:**
        - Cannot always distinguish between tumor types with complete certainty
        - May not always differentiate between tumor and inflammation or infection
        - Some tumors may have similar appearances to non-cancerous conditions
        - The definitive diagnosis usually requires a biopsy (tissue sample)
        
        For these reasons, MRI findings are typically combined with clinical symptoms, medical history, 
        and sometimes biopsy results to make a definitive diagnosis.
        """)
    
    with st.expander("What happens if a brain tumor is suspected?"):
        st.markdown("""
        If a brain tumor is suspected, the typical process involves:
        
        1. **Initial evaluation**: A thorough neurological examination and review of symptoms and medical history
        
        2. **Imaging studies**: MRI is the primary imaging tool, often with contrast enhancement. CT scans may be 
        performed if MRI is not available or contraindicated
        
        3. **Additional tests**: Depending on the location and suspected type of tumor, these might include:
           - EEG (electroencephalogram) if seizures are present
           - Vision and hearing tests if relevant areas are affected
           - Blood tests to check hormone levels for pituitary tumors
        
        4. **Biopsy**: A small sample of the tumor tissue may be removed for laboratory analysis to determine:
           - Whether the tumor is benign or malignant
           - The exact type of tumor
           - The grade (level of aggressiveness)
        
        5. **Treatment planning**: Once diagnosed, a multidisciplinary team (neurosurgeons, neurologists, 
        oncologists, etc.) will develop a treatment plan based on:
           - Tumor type, size, and location
           - Whether it's primary or metastatic
           - The patient's age and overall health
           - Patient preferences
        
        Throughout this process, patients are usually referred to specialists such as neurosurgeons, neurologists, 
        or neuro-oncologists who have expertise in treating brain tumors.
        """)
    
    st.subheader("About Our AI Detection System")
    
    with st.expander("How does the AI brain tumor detection system work?"):
        st.markdown("""
        Our AI-powered brain tumor detection system uses deep learning technology to analyze MRI scans and identify 
        potential brain tumors. Here's how it works:
        
        1. **Input**: The system takes a brain MRI scan image as input
        
        2. **Preprocessing**: The image is resized and normalized to prepare it for analysis
        
        3. **AI Analysis**: A deep neural network (specifically a modified ResNet architecture) analyzes the image. 
        This neural network was trained on thousands of labeled MRI scans to recognize patterns associated with 
        different types of brain tumors
        
        4. **Classification**: The system classifies the MRI into one of four categories:
           - Glioma tumor
           - Meningioma tumor
           - Pituitary tumor
           - No tumor
        
        5. **Visualization**: The system generates a GradCAM (Gradient-weighted Class Activation Mapping) heatmap 
        that highlights the regions of the image that most influenced the AI's decision, helping to localize the 
        tumor
        
        6. **Results**: The system provides a classification, confidence score, and visual overlays to help 
        understand the results
        
        Our AI system achieved 95% accuracy on test datasets, but it's designed to be a supportive tool for 
        healthcare professionals rather than a replacement for expert diagnosis.
        """)
    
    with st.expander("How accurate is the AI prediction?"):
        st.markdown("""
        Our brain tumor detection AI system has demonstrated high accuracy in research settings:
        
        - **Overall accuracy**: 95% on test datasets
        - **Sensitivity**: 93% (ability to correctly identify positive cases)
        - **Specificity**: 96% (ability to correctly identify negative cases)
        - **Confidence score**: Each prediction comes with a confidence percentage that indicates how certain the AI is about its classification
        
        Performance varies slightly by tumor type:
        - Highest accuracy for "no tumor" cases (98%)
        - Slightly lower accuracy for differentiating between specific tumor types (92-95%)
        
        However, it's important to understand the limitations:
        
        - The system was trained on a specific dataset of MRI scans and may not perform as well on images taken with different equipment or protocols
        - The AI cannot diagnose rare tumor types that weren't represented in its training data
        - The system is designed to be a supportive tool for healthcare professionals, not a replacement for expert diagnosis
        - A definitive diagnosis always requires clinical correlation and often tissue biopsy
        
        We recommend using the AI predictions as a screening tool or second opinion, always in conjunction with professional medical evaluation.
        """)
    
    with st.expander("What is GradCAM and how should I interpret the heatmap?"):
        st.markdown("""
        GradCAM (Gradient-weighted Class Activation Mapping) is a visualization technique that helps explain 
        what regions of an image the AI is focusing on when making its prediction.
        
        **How GradCAM works:**
        - It identifies which areas of the MRI scan were most important in the AI's decision-making process
        - These areas are highlighted with a color overlay, typically using a heat map color scheme
        - Warmer colors (red, orange) indicate regions that strongly influenced the AI's prediction
        - Cooler colors (blue, green) indicate areas with less influence
        
        **How to interpret the heatmap:**
        
        - **Strong red/orange areas**: These areas strongly suggest the presence of abnormal tissue that the AI associates with a tumor. In an accurate prediction, these areas should correspond to the actual tumor location
        
        - **Diffuse or scattered highlights**: If the heatmap shows scattered highlights without clear focus, the AI might be less certain or might be picking up on subtle patterns
        
        - **No significant highlights**: If there are no strong highlighted areas but the AI still predicts a tumor, consider looking at the confidence score - it might be a lower confidence prediction
        
        - **Highlights in unexpected areas**: If the GradCAM highlights areas that don't make anatomical sense for the predicted tumor type, the prediction might be less reliable
        
        The GradCAM visualization is particularly useful for healthcare professionals to verify whether the AI is looking at relevant anatomical regions and as a tool to help localize the tumor for further investigation.
        """)
    
    with st.expander("Can this tool replace a doctor's diagnosis?"):
        st.markdown("""
        **No, this tool cannot and should not replace a doctor's diagnosis.**
        
        Our AI brain tumor detection system is designed to be a supportive tool for healthcare professionals, not a replacement for proper medical evaluation and diagnosis. Here's why:
        
        1. **Supplementary role**: The system is intended to assist healthcare providers by providing an additional analytical perspective, potentially helping with early detection or serving as a second opinion
        
        2. **Technical limitations**: While our AI achieves high accuracy, it:
           - Is limited by the quality and type of images it was trained on
           - Cannot account for all rare tumor variants
           - Does not have access to the patient's medical history, symptoms, or other crucial clinical information
        
        3. **Comprehensive diagnosis**: Proper diagnosis of brain tumors requires:
           - Clinical assessment of symptoms and neurological function
           - Interpretation of multiple imaging studies, often in different sequences
           - Consideration of the patient's medical history
           - Frequently, a biopsy for definitive tissue diagnosis
           - Integration of all this information by trained specialists
        
        4. **Treatment decisions**: Beyond diagnosis, treatment planning requires medical expertise to consider:
           - The patient's overall health
           - Risk-benefit analysis of different treatment options
           - Potential complications
           - Long-term management strategies
        
        **Always consult healthcare professionals**: Any results from this system should be reviewed by qualified healthcare providers who can place them in the proper clinical context and determine appropriate next steps.
        """)
    
    st.subheader("Treatment and Prognosis")
    
    with st.expander("What treatments are available for brain tumors?"):
        st.markdown("""
        Treatment options for brain tumors vary widely depending on tumor type, size, location, and the patient's overall health. Common treatments include:
        
        **Surgery**
        - Often the first line of treatment to remove as much of the tumor as possible
        - May be curative for many benign tumors
        - For some deep or sensitive locations, complete removal may not be possible
        - Advanced techniques include awake craniotomy, laser ablation, and minimally invasive approaches
        
        **Radiation Therapy**
        - Uses high-energy beams to kill tumor cells
        - May be primary treatment when surgery isn't possible
        - Often used after surgery to target remaining tumor cells
        - Types include conventional external beam radiation, stereotactic radiosurgery (like Gamma Knife), and proton therapy
        
        **Chemotherapy**
        - Uses drugs to kill rapidly dividing cells
        - May be given orally or intravenously
        - Often used in combination with surgery and radiation
        - Drugs like temozolomide are commonly used for malignant gliomas
        
        **Targeted Therapy**
        - Focuses on specific abnormalities in tumor cells
        - Examples include anti-angiogenic therapy (blocking blood vessel formation) and growth factor inhibitors
        - May have fewer side effects than traditional chemotherapy
        
        **Tumor Treating Fields (TTFields)**
        - A newer approach using electrical fields to disrupt cell division
        - FDA-approved for some high-grade gliomas
        
        **Immunotherapy**
        - Harnesses the body's immune system to fight tumor cells
        - Currently being studied in clinical trials for brain tumors
        
        **Supportive Care**
        - Medications to manage symptoms like seizures, headaches, or edema
        - Rehabilitation services (physical, occupational, speech therapy)
        - Psychological support and counseling
        
        Treatment plans are typically multimodal, combining several approaches, and are created by multidisciplinary teams of specialists including neurosurgeons, radiation oncologists, medical oncologists, and others.
        """)
    
    with st.expander("What is the prognosis for someone with a brain tumor?"):
        st.markdown("""
        The prognosis for someone with a brain tumor varies significantly based on multiple factors:
        
        **Key Factors Affecting Prognosis**
        
        **Tumor Type and Grade**:
        - Low-grade benign tumors (like many meningiomas) often have excellent prognosis
        - High-grade malignant tumors (like glioblastoma) typically have more challenging outcomes
        
        **Specific Tumor Types (5-year survival rates)**:
        - Meningioma: 80-90%
        - Low-grade astrocytoma: 65-80%
        - Oligodendroglioma: 50-80%
        - Pituitary tumors: 80-95%
        - Glioblastoma: 5-10% (the most aggressive common brain tumor)
        
        **Tumor Location**:
        - Tumors in accessible areas that can be surgically removed have better outcomes
        - Tumors in critical brain areas or deep structures may have worse prognosis
        
        **Extent of Tumor Removal**:
        - Complete removal generally improves prognosis
        - Partial removal may still provide symptomatic relief and extend survival
        
        **Patient Factors**:
        - Age: Younger patients often have better outcomes
        - General health: Better overall health status improves tolerance of treatments
        - Functional status: Higher functioning at diagnosis correlates with better prognosis
        
        **Molecular Features**:
        - Certain genetic mutations can significantly impact prognosis and treatment response
        - For example, IDH mutations in gliomas are associated with better outcomes
        
        **Response to Treatment**:
        - Early positive response to initial therapies is a favorable prognostic sign
        
        It's important to discuss individual prognosis with healthcare providers who can consider all relevant factors specific to each case. Many patients live well beyond statistical predictions, and quality of life can often be maintained with appropriate treatments and supportive care.
        """)
    
    with st.expander("What research is being done on brain tumors?"):
        st.markdown("""
        Brain tumor research is a dynamic field with numerous promising developments:
        
        **Molecular and Genetic Research**
        - Identifying genetic mutations and molecular markers in different tumor types
        - Developing classification systems based on molecular profiles rather than just microscopic appearance
        - Understanding the role of specific genes and pathways in tumor development and growth
        
        **Imaging Advances**
        - Advanced MRI techniques like perfusion imaging, diffusion tensor imaging, and spectroscopy
        - PET scans with new radiotracers to better identify tumor boundaries and metabolic activity
        - AI and machine learning applications for improved diagnosis and treatment planning
        
        **Treatment Innovations**
        
        **Surgical:**
        - Fluorescence-guided surgery to better visualize tumor boundaries
        - Laser interstitial thermal therapy for minimally invasive ablation
        - Improved intraoperative brain mapping techniques
        
        **Radiation:**
        - More precise radiation delivery systems
        - Hypofractionated approaches (higher doses in fewer sessions)
        - Combination strategies with radiosensitizing agents
        
        **Drug Development:**
        - Novel chemotherapy agents with better brain penetration
        - Targeted therapies addressing specific molecular alterations
        - Blood-brain barrier disruption techniques to improve drug delivery
        
        **Immunotherapy:**
        - Checkpoint inhibitors
        - CAR T-cell therapy adapted for brain tumors
        - Vaccine strategies
        - Oncolytic virus therapy
        
        **Clinical Trials**
        - Hundreds of active clinical trials testing new treatments
        - Innovative trial designs allowing more patients to access experimental therapies
        - Basket trials grouping patients by molecular features rather than tumor type
        
        **Quality of Life Research**
        - Better understanding and management of treatment side effects
        - Cognitive rehabilitation approaches
        - Supportive care optimization
        
        These research efforts provide hope for improved outcomes for brain tumor patients in the future. Patients interested in cutting-edge treatments should discuss clinical trial participation with their healthcare providers.
        """)
    
    # Add a "Still have questions?" section at the bottom
    st.markdown("---")
    st.subheader("Still Have Questions?")
    
    st.markdown("""
    <div style="background-color: #f8f9fa; padding: 15px; border-radius: 5px; margin-top: 10px; border-left: 3px solid #3498db;">
        <p>If you have additional questions about brain tumors or our detection system, we recommend:</p>
        <ul>
            <li>Consulting with a healthcare professional for medical advice</li>
            <li>Visiting reputable medical websites such as the <a href="https://www.cancer.org/cancer/types/brain-spinal-cord-tumors-adults.html" target="_blank">American Cancer Society</a>, <a href="https://www.mayoclinic.org/diseases-conditions/brain-tumor/symptoms-causes/syc-20350084" target="_blank">Mayo Clinic</a>, or the <a href="https://www.abta.org/" target="_blank">American Brain Tumor Association</a></li>
            <li>Contacting patient support groups for shared experiences and information</li>
            <li>Referring to medical journals for the latest research findings</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)      




def main():
    # Page Configuration
    st.set_page_config(
        page_title="BrainScan AI | Brain Tumor Detection",
        page_icon="🧠",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Initialize session state to track current page if it doesn't exist
    if 'current_page' not in st.session_state:
        st.session_state['current_page'] = "🏠 Home"

    # Custom CSS
    st.markdown("""
    <style>
    /* Base Styling */
    .stApp {
        background-color: #f8f9fa;
        font-family: 'Inter', sans-serif;
        color: #2c3e50;
    }
    
    /* Typography */
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Inter', sans-serif;
        font-weight: 600;
        color: #2c3e50;
    }
    
    p, li {
        font-size: 16px;
        line-height: 1.6;
    }
    
    /* Sidebar */
    .css-1d391kg {
        background-color: #2c3e50;
    }
    
    /* Cards and Containers */
    .info-card, .feature-card, .result-card, .glossary-item, 
    .tumor-info-card, .interpretation-card, .metrics-explanation,
    .factor-card, .symptoms-card, .recommendation-card, .treatment-details,
    .post-care-card, .info-box {
        background-color: white;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
        padding: 20px;
        margin-bottom: 20px;
        border-left: 5px solid #3498db;
    }
    
    /* Feature Cards */
    .feature-card {
        text-align: center;
        transition: transform 0.3s;
    }
    
    .feature-card:hover {
        transform: translateY(-5px);
    }
    
    .feature-icon {
        font-size: 2.5rem;
        margin-bottom: 15px;
    }
    
    /* Hero Section */
    .hero-section {
        padding: 20px 0;
    }
    
    .hero-text {
        font-size: 18px;
        color: #555;
        line-height: 1.6;
    }
    
    .call-to-action {
        margin: 20px 0;
        font-size: 18px;
        font-weight: 500;
    }
    
    /* Stats Counter */
    .stats-section {
        display: flex;
        justify-content: space-around;
        margin: 30px 0;
        text-align: center;
    }
    
    .stat-number {
        font-size: 40px;
        font-weight: 700;
        color: #3498db;
    }
    
    .stat-label {
        font-size: 16px;
        color: #7f8c8d;
    }
    
    /* Risk Level Indicator */
    .risk-level {
        font-size: 24px;
        font-weight: 600;
        margin: 10px 0;
    }
    
    .risk-high {
        color: #e74c3c;
    }
    
    .risk-moderate {
        color: #f39c12;
    }
    
    .risk-low {
        color: #27ae60;
    }
    
    /* Confidence Bar */
    .confidence-bar {
        width: 100%;
        height: 10px;
        background-color: #ecf0f1;
        border-radius: 5px;
        margin: 10px 0;
    }
    
    .confidence-value {
        height: 100%;
        border-radius: 5px;
        background-color: #3498db;
    }
    
    /* Result Tumor Type */
    .result-tumor-type {
        font-size: 24px;
        font-weight: 600;
        color: #2c3e50;
        margin: 10px 0;
    }
    
    /* GradCAM Legend */
    .gradcam-legend {
        background-color: white;
        border-radius: 10px;
        padding: 15px;
        margin-top: 20px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
    }
    
    .color-scale {
        display: flex;
        margin: 10px 0;
    }
    
    .color-box {
        width: 40px;
        height: 20px;
    }
    
    .scale-labels {
        display: flex;
        justify-content: space-between;
        font-size: 14px;
        color: #7f8c8d;
    }
    
    /* Factor Cards */
    .factor-card.genetic {
        border-left-color: #9b59b6;
    }
    
    .factor-card.environmental {
        border-left-color: #27ae60;
    }
    
    .factor-card.lifestyle {
        border-left-color: #f39c12;
    }
    
    /* Symptoms Cards */
    .symptoms-card.warning {
        border-left-color: #e74c3c;
    }
    
    /* Testimonial Cards */
    .testimonial-card {
        background-color: white;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        padding: 20px;
        margin-bottom: 20px;
        border-left: 5px solid #9b59b6;
    }
    
    .testimonial-quote {
        font-style: italic;
        color: #555;
        line-height: 1.6;
    }
    
    .testimonial-author {
        font-weight: 600;
        margin-top: 15px;
        text-align: right;
    }
    
    /* Brain Image Placeholder */
    .brain-img-placeholder {
        background-color: #3498db15;
        border: 2px dashed #3498db;
        border-radius: 10px;
        height: 250px;
        display: flex;
        justify-content: center;
        align-items: center;
        margin-bottom: 20px;
    }
    
    .brain-img-content {
        font-size: 24px;
        font-weight: 600;
        color: #3498db;
    }
    
    /* Sample Image Placeholder */
    .sample-img-placeholder {
        background-color: #3498db15;
        border: 2px dashed #3498db;
        border-radius: 10px;
        height: 150px;
        display: flex;
        justify-content: center;
        align-items: center;
        margin-bottom: 10px;
    }
    
    .sample-img-content {
        font-size: 16px;
        font-weight: 600;
        color: #3498db;
    }
    
    /* Buttons */
    .stButton>button {
        background-color: #3498db;
        color: white;
        border-radius: 5px;
        padding: 10px 20px;
        font-weight: 500;
        border: none;
        transition: all 0.3s ease;
    }
    
    .stButton>button:hover {
        background-color: #2980b9;
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    
    /* Upload Card */
    .upload-card {
        text-align: center;
        padding: 15px;
        background-color: #f8f9fa;
        border-radius: 10px;
        margin-bottom: 20px;
    }
    
    /* Tumor Image Placeholder */
    .tumor-img-placeholder {
        background-color: #3498db15;
        border: 2px dashed #3498db;
        border-radius: 10px;
        height: 200px;
        display: flex;
        justify-content: center;
        align-items: center;
        margin-bottom: 20px;
    }
    
    .tumor-img-content {
        font-size: 18px;
        font-weight: 600;
        color: #3498db;
    }
    
    /* Recommendations List */
    .recommendations li {
        margin-bottom: 10px;
        padding-left: 10px;
        border-left: 3px solid #3498db;
    }
    
    /* Risk Result Container */
    .result-container {
        margin-top: 20px;
    }
    
    .risk-level-card {
        padding: 15px;
        border-radius: 5px;
        margin-bottom: 15px;
    }
    
    .disclaimer {
        font-size: 14px;
        color: #7f8c8d;
        font-style: italic;
    }
    
    /* Info Note */
    .info-note {
        background-color: #f8f9fa;
        border-left: 3px solid #3498db;
        padding: 10px 15px;
        margin: 20px 0;
        font-size: 14px;
    }
    
    /* Search Container */
    .search-container {
        text-align: center;
        margin-bottom: 20px;
    }
    
    /* Note Card */
    .note-card {
        background-color: #f8f9fa;
        border-left: 3px solid #f39c12;
        padding: 15px;
        margin: 20px 0;
        border-radius: 5px;
    }
    
    /* Detection Info */
    .detection-info ul {
        padding-left: 20px;
    }
    
    .detection-info li {
        margin-bottom: 5px;
    }
    .tutorial-step {
        display: flex;
        margin-bottom: 30px;
        background-color: white;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
        overflow: hidden;
    }
    
    .step-number {
        display: flex;
        justify-content: center;
        align-items: center;
        background-color: #3498db;
        color: white;
        font-size: 24px;
        font-weight: bold;
        min-width: 60px;
        padding: 20px 0;
    }
    
    .step-content {
        padding: 20px;
        flex-grow: 1;
    }
    
    .step-content h3 {
        margin-top: 0;
        color: #3498db;
    }
    
    .resource-card, .tips-card {
        background-color: white;
        border-radius: 10px;
        padding: 15px;
        margin-bottom: 15px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
    }
    
    .resource-card h4, .tips-card h4 {
        color: #3498db;
        margin-top: 0;
    }
    
    .tutorial-resources {
        margin-bottom: 30px;
    }
    
    </style>
    """, unsafe_allow_html=True)

    # Navigation (check if we need to navigate somewhere specific)
    if 'navigate_to' in st.session_state:
        current_page = st.session_state['navigate_to']
        del st.session_state['navigate_to']
    else:
        current_page = "🏠 Home"

    # Navigation options
    pages = {
        "🏠 Home": home_page,
        "🔍 Tumor Detection": tumor_detection_page,
        "🚶‍♂️ Tutorial": tutorial_page,  # Add this line
        "📊 Survival Statistics": survival_statistics_page,
        "⚠️ Prevention & Care": prevention_page,
        "🩺 Tumor Types": tumor_types_page,
        "📚 Medical Glossary": medical_terminology_page,
        "📊 Global Tumor Statistics": global_statistics_page,
        "🔬 Tumor Progression": tumor_progression_page,
        "📝 Risk Assessment": risk_assessment_page,
        "❓ FAQ": faq_page,
        "📚 About Dataset": about_dataset_page
    }

    # Sidebar Navigation
    with st.sidebar:
        st.title("🧠 BrainScan AI")
        st.markdown("### Brain Tumor Detection & Analysis")
        
        # Navigation radio buttons
        selected_page = st.radio("Navigation", list(pages.keys()), index=list(pages.keys()).index(st.session_state['current_page']))
        
        # If user changes page using the sidebar, update session state
        if selected_page != st.session_state['current_page']:
            st.session_state['current_page'] = selected_page
            st.rerun()  # Rerun to update the page content
            
        # Sidebar footer
        st.markdown("---")
        st.markdown("### About")
        st.markdown("BrainScan AI is a tool for brain tumor detection and educational purposes. It is not a substitute for professional medical advice.")
        
        st.markdown("---")
        st.markdown("© 2025 BrainScan AI")
        st.markdown("Version 1.0")

    # Render the current page
    pages[st.session_state['current_page']]()


if __name__ == "__main__":
    main()