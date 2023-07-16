from flask import Flask, render_template, request, jsonify
import tensorflow as tf
import numpy as np
import tensorflow_hub as hub
import os
from PIL import Image
from flask_cors import CORS

app = Flask(__name__)
CORS(app, origins=['*'])


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
    if modelName == 'base-cnn':
    #    target_size = (48, 48)
       model = tf.keras.models.load_model('./models/base_CNN.h5', custom_objects={'KerasLayer':hub.KerasLayer})
    elif modelName == 'final-cnn':
    #    target_size = (48, 48)
        model = tf.keras.models.load_model('./models/final_CNN.h5', custom_objects={'KerasLayer':hub.KerasLayer})
    elif modelName == 'ann' :
        
        model = tf.keras.models.load_model('./models/ANN.h5', custom_objects={'KerasLayer':hub.KerasLayer})
    elif modelName == 'resNet': 
         target_size = (48, 48)
         model = tf.keras.models.load_model('./models/RN_weights-009-0.3958.hdf5', custom_objects={'KerasLayer':hub.KerasLayer})

    else: 
        model = tf.keras.models.load_model('./models/final_CNN.h5', custom_objects={'KerasLayer':hub.KerasLayer})
            

    if image_file.filename == '':
        return 'No file selected'
    

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
    if probability > 0.8:
        result = 'No Cancer Detected'
    else:
        result = 'Cancer Detected'

    return jsonify({"status": True, 'data': result })

if __name__ == '__main__':
    print('hello')
    app.run(debug=True, port=4000)