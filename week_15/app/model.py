# import os
# import requests
# from django.conf import settings
# import tensorflow as tf
# from venv import logger

# # Function to download the model file
# def download_model(model_url, model_local_path):
#     try:
#         response = requests.get(model_url)
#         response.raise_for_status()  # Check if the download was successful
#         with open(model_local_path, 'wb') as f:
#             f.write(response.content)
#         logger.info(f"Model downloaded successfully from {model_url}")
#     except Exception as e:
#         logger.error(f"Failed to download model from {model_url}: {e}")
#         raise

# # Function to load the model
# def load_model_from_path(model_local_path):
#     try:
#         model = tf.keras.models.load_model(model_local_path)
#         logger.info(f"Model loaded successfully from {model_local_path}")
#         return model
#     except Exception as e:
#         logger.error(f"Failed to load model from {model_local_path}: {e}")
#         raise

# # Path where you want to save the downloaded model
# model_local_path = os.path.join(settings.BASE_DIR, 'model.h5')

# # URL of the model file on GitHub
# # model_url = "https://raw.githubusercontent.com/TechnoVishalGirase/EcoEye-Online_Bird_Monitoring_System/main/week_14/model.h5"
# model_url = "C:/EcoBirdEye_week_14/model.h5"


# # Download and load the model
# download_model(model_url, model_local_path)
# model = load_model_from_path(model_local_path)