from django import forms
from .models import *

class DiseaseForm(forms.Form):
    diseaseimg = forms.FileField(label = None)