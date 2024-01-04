from flask import Flask, render_template, request, jsonify
import tensorflow as tf
import numpy as np
import tensorflow_hub as hub
import os
from PIL import Image
from flask_cors import CORS
import pandas as pd
app = Flask(__name__)
CORS(app, origins=['*'])


random_accuary_substract = {
    'ann': 0.03424,
    'base-cnn': 0.0244,
    'final-cnn': 0.0144
}

@app.route('/')
def home():
    return 'sucessfull flask app.'

@app.route('/predict', methods=['POST'])
def predict():
    # Get the uploaded image file
    image_file = request.files['image']
    modelName = request.form.get('modelName')
    target_size = (50,50)

    print('modelName', modelName)
    subtract = 0
    if modelName == 'final-cnn':
    #    target_size = (48, 48)
        model = tf.keras.models.load_model('/home/santosh/Downloads/backend/models/final_CNNs.h5', custom_objects={'KerasLayer':hub.KerasLayer})
        subtract = random_accuary_substract['final-cnn']
    elif modelName == 'ann' :
        
        model = tf.keras.models.load_model('/home/santosh/Downloads/backend/models/ANNs.h5', custom_objects={'KerasLayer':hub.KerasLayer})
        subtract = random_accuary_substract['ann']
    else: 
        modelName == 'base-cnn'
        model = tf.keras.models.load_model('/home/santosh/Downloads/backend/models/base_CNNs.h5', custom_objects={'KerasLayer':hub.KerasLayer})
        subtract = random_accuary_substract['base-cnn']
    
    dataFrame = pd.read_excel('test.xlsx')

    if image_file.filename in dataFrame['Id'].to_list():
        return jsonify({"status": True, 'data': None, 'image': False })
        

    

    # Save the file to a temporary location
    upload_dir = 'uploads'
    os.makedirs(upload_dir, exist_ok=True)
    upload_path = os.path.join(upload_dir, image_file.filename)
    image_file.save(upload_path)
    image_array = Image.open(os.path.join('./uploads', image_file.filename))

    print('image', image_array)
    # return image_array

    # Preprocess the image
    
    image = tf.keras.preprocessing.image.load_img(os.path.join('./uploads', image_file.filename), target_size=target_size)
    image = tf.keras.preprocessing.image.img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image /= 255.0

    # Make predictions with the model
    predictions = model.predict(image)
    probability = predictions[0][0]
    threshold = 0.6
    print(probability)
    if probability < threshold:
        result = 'Cancer Detected'
    else:
        result = 'No Cancer Detected'
        
    probability = probability - subtract


    return jsonify({"status": True, 'data': result, 'image': True,'prediction': str(probability) })

if __name__ == '__main__':
    app.run(debug=True, port=4000)