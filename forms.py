from django import forms
from .models import PatientData

class PatientForm(forms.ModelForm):
    class Meta:
        model = PatientData
        fields = ['name', 'age', 'time_since_injury', 'gcs', 'gos']
