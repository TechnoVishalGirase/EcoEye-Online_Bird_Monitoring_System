from venv import logger
from django.shortcuts import render, redirect
from django.http import HttpResponse
import tensorflow as tf
from PIL import Image
import numpy as np
import io
import os
import tempfile
import requests
									  
def index(request):
    return render(request,"app/index.html")

def error_(request):
    return render(request,"app/404.html")

def about(request):
    features = ['Advanced Bird Tracking', 'AI Monitoring System', 'Wildlife Analytics', 'Real-Time Data Collection', 'User-Friendly Interface']
    return render(request,"app/about.html", {'features': features})

def booking(request):
    return render(request,"app/booking.html")

def bird_tracking(request):
    return render(request,"app/bird_tracking.html")						   
												   
def destination(request):
    return render(request,"app/destination.html")

def service(request):
    return render(request,"app/service.html")

def package(request):
    return render(request,"app/package.html")

def team(request):
    return render(request,"app/team.html")

def testimonial(request):
    return render(request,"app/testimonial.html")

def contact(request):
    return render(request,"app/contact.html")

#-------------backend logic----------------
# Load your trained model (ensure the path is correct)
# model = tf.keras.models.load_model('C:/Ecobird_week_12/venv/model.h5')
# Load your trained model (ensure the path is correct)
# model_path = 'C:/EcoBirdEye_week_14/model.h5'
#model_path = "https://raw.githubusercontent.com/TechnoVishalGirase/EcoEye-Online_Bird_Monitoring_System/main/week_14/model.h5"
#####--------------Testing--------------###########
from pathlib import Path
BASE_DIR = Path(__file__).resolve().parent.parent
model_local_path = os.path.join(BASE_DIR, 'model.h5')
model_url = "https://raw.githubusercontent.com/TechnoVishalGirase/EcoEye-Online_Bird_Monitoring_System/main/week_14/model.h5"

def download_model(model_url, model_local_path):
    try:
        response = requests.get(model_url)
        response.raise_for_status()  # Check if the download was successful
        with open(model_local_path, 'wb') as f:
            f.write(response.content)
        logger.info(f"Model downloaded successfully from {model_url}")
    except Exception as e:
        logger.error(f"Failed to download model from {model_url}: {e}")
        raise

# Function to load the model
def load_model_from_path(model_local_path):
    try:
											
																			 
        model = tf.keras.models.load_model(model_local_path)
									   
        logger.info(f"Model loaded successfully from {model_local_path}")
        return model
    except Exception as e:
        logger.error(f"Failed to load model from {model_local_path}: {e}")
        raise

							  
download_model(model_url, model_local_path)
		  
model = load_model_from_path(model_local_path)
																		   
					  
							
																			
			   

##-------------------------------------------------##################
												
											 


# try:
#     model = tf.keras.models.load_model(model_path)
#     logger.info(f"Model loaded successfully from {model_path}")
# except Exception as e:
#     logger.error(f"Failed to load model from {model_path}: {e}")

def preprocess_image(image_path):
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=(224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)  # Create a batch
    img_array = tf.keras.applications.efficientnet.preprocess_input(img_array)
    return img_array

def predict_image(model, processed_image):
    predictions = model.predict(processed_image)
    predicted_class = np.argmax(predictions, axis=1)
    return predicted_class

labels_dict = {0: 'AFRICAN EMERALD CUCKOO',
 1: 'AFRICAN PIED HORNBILL',
 2: 'ALBATROSS',
 3: 'AMERICAN BITTERN',
 4: 'GOLDEN CHEEKED WARBLER',
 5: 'GRAY KINGBIRD',
 6: 'LONG-EARED OWL',
 7: 'MYNA',
 8: 'RAZORBILL',
 9: 'RED TAILED HAWK'}

# def get_class_label(class_index):
#     # Assuming you have a dictionary that maps class indices to labels
#     # labels = {0: 'Class1', 1: 'Class2', ...}  # Fill in your actual labels
#     return labels_dict[class_index]

def get_class_label(class_index):
    return labels_dict.get(class_index, "Unknown class")

# def predict(request):
#     if 'image' not in request.files:
#         return 'No file part', 400
#     file = request.files['image']
#     if file.filename == '':
#         return 'No selected file', 400
#     if file:
#         # Preprocess the uploaded image
#         # processed_image = preprocess_image(file)

#         # Save the file to a temporary file
#         with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
#             file.save(tmp_file.name)
#             # Now you can use the saved file's path to preprocess the image
#             processed_image = preprocess_image(tmp_file.name)
        
#         # Remember to delete the temporary file after processing
#         os.unlink(tmp_file.name)
        
#         # Predict the class of the processed image
#         predicted_class_index = predict_image(model, processed_image)
        
#         # Convert the class index to a readable class label
#         class_label = get_class_label(predicted_class_index.item())
#         return render(request,"frontend/result.html", prediction=class_label)

# working
# def predict(request):
#     if request.method == 'POST':
#         file = request.FILES.get('image', None)  # Use request.FILES, not request.files
#         if file is None:
#             return HttpResponse('No selected file', status=400)

#         try:
#             with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
#                 for chunk in file.chunks():
#                     tmp_file.write(chunk)
#                 processed_image = preprocess_image(tmp_file.name)

#             predicted_class_index = predict_image(model, processed_image)
#             class_label = get_class_label(predicted_class_index.item())
#             logger.info(f"Prediction: {class_label}")

#             # Assuming you have set up a method to handle the image saving and URL generation
#             # For simplicity, just returning the label now
#             return HttpResponse(f'Predicted Label: {class_label}')
#         except Exception as e:
#             logger.error(f"Error processing image and making prediction: {e}")
#             return HttpResponse('Error processing the prediction', status=500)

#     return HttpResponse('Invalid request', status=400)

# Set the custom temporary directory
# CUSTOM_TEMP_DIR = 'C:/Ecobird_week_12/venv/ecobirdeye/ecobird/templates/frontend/static/img'
# CUSTOM_TEMP_DIR = 'C:/Ecobird_week_12/venv/templates/frontend/static/img/'

##---------------------Working----------------------------------------------------------

# CUSTOM_TEMP_DIR = 'C:/EcoBirdEye_week_14/app/templates/app/static/img/'

# def predict(request):
#     if request.method == 'POST':
#         file = request.FILES.get('image', None)
#         if file is None:
#             return HttpResponse('No selected file', status=400)

#         # try:
#             # Save the file temporarily in the custom directory
#         with tempfile.NamedTemporaryFile(dir=CUSTOM_TEMP_DIR, delete=False, suffix=".jpg") as tmp_file:
#             for chunk in file.chunks():
#                 tmp_file.write(chunk)
#             tmp_file.flush()
#             os.fsync(tmp_file.fileno())

#             processed_image = preprocess_image(tmp_file.name)

#         # Get the prediction
#         predicted_class_index = predict_image(model, processed_image)
#         class_label = get_class_label(predicted_class_index.item())

#         # Generate a URL to access the temporary image
#         image_url = f'CUSTOM_TEMP_DIR{os.path.basename(tmp_file.name)}'  # Adjust as per your media URL configuration

#         # Return both the predicted label and the image URL
#         # return HttpResponse(f'Predicted Label: {class_label}\nImage URL: {image_url}')
#         return render(request, "app/result.html", {'label': class_label, 'image_url': image_url})
#         # except Exception as e:
#         #     return HttpResponse('Error processing the prediction', status=500)

#     return HttpResponse('Invalid request', status=400)
######----------------------------------------------------------------------------------------------------------------

																														   
from django.conf import settings
from django.http import HttpResponse
from django.shortcuts import render
from django.core.files.storage import default_storage
from django.core.files.base import ContentFile

def predict(request):
    if request.method == 'POST':
        file = request.FILES.get('image', None)
        if file is None:
            return HttpResponse('No selected file', status=400)

        # Save the file temporarily
        file_name = default_storage.save("temp_images/" + file.name, ContentFile(file.read()))
        file_path = default_storage.path(file_name)

        # Process the image and predict
        processed_image = preprocess_image(file_path)
        predicted_class_index = predict_image(model, processed_image)
        class_label = get_class_label(predicted_class_index.item())

        # Generate a URL to access the temporary image
        # Assuming you have MEDIA_URL set in your settings and the media files are served correctly
        image_url = settings.MEDIA_URL + file_name

        # Clean up the temporary file if needed
        default_storage.delete(file_name)

        # Return the predicted label and the image URL
        return render(request, "app/result.html", {'label': class_label, 'image_url': image_url})

    return HttpResponse('Invalid request', status=400)



# import tempfile
# import os

# # Set the custom temporary directory
# CUSTOM_TEMP_DIR = 'C:/Ecobird_week_12/venv/ecobirdeye/ecobird/templates/frontend/static/img'

# def predict(request):
#     if request.method == 'POST':
#         file = request.FILES.get('image', None)
#         if file is None:
#             return HttpResponse('No selected file', status=400)

#         try:
#             # Save the file temporarily in the custom directory
#             with tempfile.NamedTemporaryFile(dir=CUSTOM_TEMP_DIR, delete=False, suffix=".jpg") as tmp_file:
#                 for chunk in file.chunks():
#                     tmp_file.write(chunk)
#                 tmp_file.flush()
#                 os.fsync(tmp_file.fileno())

#                 processed_image = preprocess_image(tmp_file.name)

#             # Get the prediction
#             predicted_class_index = predict_image(model, processed_image)
#             class_label = get_class_label(predicted_class_index.item())
#             logger.info(f"Prediction: {class_label}")

#             return render(request, "result.html", {'label': class_label, 'image_url': tmp_file.name})
#         except Exception as e:
#             logger.error(f"Error processing image and making prediction: {e}")
#             return HttpResponse('Error processing the prediction', status=500)

#     return HttpResponse('Invalid request', status=400)



# def predict(request):
#     if request.method == 'POST':
#         file = request.FILES.get('image', None)
#         if file is None:
#             return HttpResponse('No selected file', status=400)
        
#         with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
#             for chunk in file.chunks():
#                 tmp_file.write(chunk)
#             processed_image = preprocess_image(tmp_file.name)
        
#         os.unlink(tmp_file.name)
        
#         predicted_class_index = predict_image(model, processed_image)
#         class_label = get_class_label(predicted_class_index.item())
#         print("File received:", file.name)
#         return render(request, "frontend/index.html", {'prediction': class_label})
#     return HttpResponse('Invalid request', status=400)

# def predict(request):
#     if request.method == 'POST':
#         file = request.FILES.get('image', None)
#         if file is None:
#             return HttpResponse('No selected file', status=400)

#         try:
#             with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
#                 for chunk in file.chunks():
#                     tmp_file.write(chunk)
#                 processed_image = preprocess_image(tmp_file.name)

#             os.unlink(tmp_file.name)

#             predicted_class_index = predict_image(model, processed_image)
#             class_label = get_class_label(predicted_class_index.item())
#             logger.info(f"Prediction: {class_label}")

#             return render(request, "frontend/index.html", {'prediction': class_label})
#         except Exception as e:
#             logger.error(f"Error processing image and making prediction: {e}")
#             return HttpResponse('Error processing the prediction', status=500)

#     return HttpResponse('Invalid request', status=400)

# def predict(request):
#     if request.method == 'POST':
#         file = request.FILES.get('image', None)
#         if file is None:
#             return HttpResponse('No selected file', status=400)

#         try:
#             # Save the file temporarily
#             with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
#                 for chunk in file.chunks():
#                     tmp_file.write(chunk)
#                 processed_image = preprocess_image(tmp_file.name)

#             # Assuming model and other functions are properly defined
#             predicted_class_index = predict_image(model, processed_image)
#             class_label = get_class_label(predicted_class_index.item())
#             logger.info(f"Prediction: {class_label}")

#             # Generate a path to save the image in a static directory
#             static_path = 'C:/Ecobird_week_12/venv/templates/frontend/static/img'  # Change this to your static images directory
#             image_filename = os.path.basename(tmp_file.name)
#             saved_image_path = os.path.join(static_path, image_filename)

#             # Move the temporary image to the static directory
#             os.rename(tmp_file.name, saved_image_path)

#             # Construct URL to access the image
#             image_url = f'C:/Ecobird_week_12/venv/templates/frontend/static/img/{image_filename}'

#             return render(request, "result.html", {'label': class_label, 'image_url': image_url})
#         except Exception as e:
#             logger.error(f"Error processing image and making prediction: {e}")
#             return HttpResponse('Error processing the prediction', status=500)

#     return HttpResponse('Invalid request', status=400)

# def predict(request):
#     if request.method == 'POST':
#         file = request.FILES.get('image', None)
#         if file is None:
#             return HttpResponse('No selected file', status=400)

#         try:
#             # Save the file temporarily
#             with tempfile.NamedTemporaryFile(delete=True, suffix=".jpg") as tmp_file:
#                 for chunk in file.chunks():
#                     tmp_file.write(chunk)
#                 # Make sure to flush so the image file is written
#                 tmp_file.flush()
#                 os.fsync(tmp_file.fileno())

#                 processed_image = preprocess_image(tmp_file.name)

#             # Get the prediction
#             predicted_class_index = predict_image(model, processed_image)
#             class_label = get_class_label(predicted_class_index.item())
#             logger.info(f"Prediction: {class_label}")

#             # For now, just return the label as an HTTP response to ensure this part works
#             return HttpResponse(f'Predicted Label: {class_label}')
#         except Exception as e:
#             logger.error(f"Error processing image and making prediction: {e}")
#             return HttpResponse('Error processing the prediction', status=500)

#     return HttpResponse('Invalid request', status=400)
