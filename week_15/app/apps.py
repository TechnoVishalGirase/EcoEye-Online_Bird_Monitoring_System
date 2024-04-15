from django.apps import AppConfig
# import tensorflow as tf
# from flask import Flask, request, render_template
import tensorflow as tf
from PIL import Image
import numpy as np
import io
import os
import tempfile

class EcoeyeConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'app'

# # Load your trained model (ensure the path is correct)
# model = tf.keras.models.load_model('C:/Ecobird_week_12/venv/model.h5')

# # Specify your custom template folder path
# app = Flask(__name__, template_folder='C:/Ecobird_week_12/venv/templates/frontend')

# def preprocess_image(image_path):
#     img = tf.keras.preprocessing.image.load_img(image_path, target_size=(224, 224))
#     img_array = tf.keras.preprocessing.image.img_to_array(img)
#     img_array = tf.expand_dims(img_array, 0)  # Create a batch
#     img_array = tf.keras.applications.efficientnet.preprocess_input(img_array)
#     return img_array

# def predict_image(model, processed_image):
#     predictions = model.predict(processed_image)
#     predicted_class = np.argmax(predictions, axis=1)
#     return predicted_class

# labels_dict = {0: 'AFRICAN EMERALD CUCKOO',
#  1: 'AFRICAN PIED HORNBILL',
#  2: 'ALBATROSS',
#  3: 'AMERICAN BITTERN',
#  4: 'GOLDEN CHEEKED WARBLER',
#  5: 'GRAY KINGBIRD',
#  6: 'LONG-EARED OWL',
#  7: 'MYNA',
#  8: 'RAZORBILL',
#  9: 'RED TAILED HAWK'}

# def get_class_label(class_index):
#     # Assuming you have a dictionary that maps class indices to labels
#     # labels = {0: 'Class1', 1: 'Class2', ...}  # Fill in your actual labels
#     return labels_dict[class_index]

# @app.route('/predict', methods=['POST'])
# def predict():
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
        
#         # Render the template with the prediction result
#         return render_template('index.html', prediction=class_label)