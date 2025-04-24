from django.shortcuts import render
from .forms import PatientForm
import numpy as np
import pickle
import os

def home(request):
    """Renders the home page."""
    return render(request, 'prognosis/home.html')

def load_model(file_name):
    """Helper function to load a model from a file."""
    model_path = os.path.join(os.path.dirname(__file__), file_name)
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    return model

def predict(request):
    """Handles the prediction form and renders the result."""
    if request.method == 'POST':
        form = PatientForm(request.POST)
        if form.is_valid():
            try:
                # Extract patient data from the form
                patient_data = form.cleaned_data
                age = patient_data['age']
                time_since_injury = patient_data['time_since_injury']
                gcs = patient_data['gcs']
                gos = patient_data['gos']

                # Load the models for mortality and morbidity
                mortality_model = load_model('mortality_model.pkl')
                morbidity_model = load_model('morbidity_model.pkl')

                # Prepare input for prediction
                input_data = np.array([[age, time_since_injury, gcs, gos]])

                # Make predictions
                mortality_prediction = mortality_model.predict(input_data)[0]  # 0 = low risk, 1 = high risk
                morbidity_prediction = morbidity_model.predict(input_data)[0]  # Scale of 0 to 1

                # Determine prognosis state
                prognosis_state = (
                    "Recovery state" if morbidity_prediction > 0.5 and mortality_prediction == 0
                    else "Critical state"
                )

                # Interpret predictions
                mortality_message = (
                    "The patient has a high risk of mortality."
                    if mortality_prediction == 1 else
                    "The patient has a low risk of mortality."
                )
                morbidity_message = (
                    f"The predicted morbidity level is {morbidity_prediction:.2f} "
                    "(higher values indicate better recovery)."
                )

                # Pass data to the result template
                context = {
                    'prognosis_state': prognosis_state,
                    'mortality_message': mortality_message,
                    'morbidity_message': morbidity_message,
                    'patient_data': patient_data
                }
                return render(request, 'prognosis/result.html', context)
            except Exception as e:
                # Handle exceptions during prediction
                return render(request, 'prognosis/predict.html', {
                    'form': form,
                    'error_message': f"An error occurred during prediction: {str(e)}"
                })
        else:
            # Handle invalid form submission
            return render(request, 'prognosis/predict.html', {
                'form': form,
                'error_message': "Please correct the errors in the form and try again."
            })
    else:
        # Render the prediction form for GET requests
        return render(request, 'prognosis/predict.html', {
            'form': PatientForm(),
        })

