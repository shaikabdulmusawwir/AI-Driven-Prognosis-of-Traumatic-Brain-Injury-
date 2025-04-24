import pickle
import numpy as np
import os

def load_model():
    try:
        # Use a relative path to the model file
        model_path = os.path.join(os.path.dirname(__file__), 'model.pkl')
        print("Model path:", model_path)  # Debugging line
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        return model
    except FileNotFoundError:
        raise FileNotFoundError(f"Model file not found at {model_path}. Ensure the file exists.")


def predict_prognosis(age, time_since_injury, gcs, gos):
    model = load_model()
    input_data = np.array([[age, time_since_injury, gcs, gos]])
    prediction = model.predict(input_data)
    return prediction[0]


