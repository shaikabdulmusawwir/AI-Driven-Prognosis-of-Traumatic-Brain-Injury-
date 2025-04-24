from django.db import models

class PatientData(models.Model):
    name = models.CharField(max_length=100)
    age = models.PositiveIntegerField()
    time_since_injury = models.FloatField(help_text="Time in hours since the injury occurred")
    gcs = models.IntegerField(help_text="Glasgow Coma Scale (3-15)")
    gos = models.IntegerField(help_text="Glasgow Outcome Scale (1-5)")
    prognosis = models.CharField(max_length=50, blank=True, help_text="Predicted outcome (auto-generated)")
    
    def __str__(self):
        return self.name
