from keras.models import load_model
import numpy as np
import pickle


from flask import Flask, render_template, request
import numpy as np
import tensorflow as tf

app = Flask(__name__)

# Load the trained model
model = pickle.load(open("static/model.pkl", "rb"))


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    # Get input values from the form
    TV = float(request.form['TV'])
    Radio = float(request.form['Radio'])
    Newspaper = float(request.form['Newspaper'])

    # Create input array for prediction
    input_array = np.array([[TV, Radio, Newspaper]])

    # Make prediction using the loaded model
    prediction = model.predict(input_array)

    # Extract the predicted output value
    output = prediction[0]

    return render_template('index.html', prediction=output)


if __name__ == '__main__':
    app.run(debug=True)