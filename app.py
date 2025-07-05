import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # ✅ Prevent OpenMP runtime conflict error

from flask import Flask, render_template, request
import torch
import torch.nn as nn
import numpy as np

app = Flask(__name__)

# Define the PyTorch model architecture matching saved model (6 input features)
class TaxiFareModel(nn.Module):
    def __init__(self):
        super(TaxiFareModel, self).__init__()
        self.linear = nn.Linear(6, 1)  # 6 input features

    def forward(self, x):
        return self.linear(x)

# Load the PyTorch model weights (✅ Added weights_only=True)
model = TaxiFareModel()
model.load_state_dict(torch.load("C:/Users/turningpointKS/Desktop/torch_taxi_fare_model.pth", weights_only=True))
model.eval()  # set to eval mode

# Define encoding mappings
time_mapping = {'morning': 0, 'afternoon': 1, 'evening': 2, 'night': 3}
day_mapping = {'weekday': 0, 'weekend': 1}
traffic_mapping = {'low': 0, 'medium': 1, 'high': 2}
weather_mapping = {'clear': 0, 'rain': 1, 'snow': 2}

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input data from form
        trip_distance = float(request.form['trip_distance'])
        time_of_day = request.form['time_of_day']
        day_of_week = request.form['day_of_week']
        traffic_condition = request.form['traffic_condition']
        weather_condition = request.form['weather_condition']
        trip_duration = float(request.form['trip_duration'])

        # Encode categorical features
        time_encoded = time_mapping.get(time_of_day.lower(), 0)
        day_encoded = day_mapping.get(day_of_week.lower(), 0)
        traffic_encoded = traffic_mapping.get(traffic_condition.lower(), 1)
        weather_encoded = weather_mapping.get(weather_condition.lower(), 0)

        # Prepare input tensor with 6 features
        features = np.array([[trip_distance, time_encoded, day_encoded,
                              traffic_encoded, weather_encoded, trip_duration]], dtype=np.float32)
        input_tensor = torch.tensor(features)

        # Predict with PyTorch model
        with torch.no_grad():
            prediction = model(input_tensor).item()

        prediction = round(prediction, 2)

        # Confidence interval ±10%
        lower = round(prediction * 0.9, 2)
        upper = round(prediction * 1.1, 2)
        confidence_range = f"{lower} - {upper}"

        # Render result keeping form data
        return render_template('index.html',
                               prediction=prediction,
                               confidence=confidence_range,
                               form_data=request.form)

    except Exception as e:
        return f"Error during prediction: {str(e)}"

if __name__ == '__main__':
    app.run(debug=True)
