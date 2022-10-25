from django.db import models

# Create your models here.
class DiseaseModel(models.Model):
    # disease_img = models.ImageField(upload_to="disease_images/")
    diseaseimg = models.FileField(upload_to="", null=True, default=None)


