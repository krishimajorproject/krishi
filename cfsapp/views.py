from django.shortcuts import render
from sklearn.feature_extraction.text import TfidfVectorizer
import requests
import pickle
import numpy as np
import pandas as pd
from .models import DiseaseModel
from .forms import DiseaseForm
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder
import requests
import config
import pickle
import io
import torch
from torchvision import transforms
from PIL import Image
from .model import ResNet9
from .disease import disease_dic
from .fertilizer import fertilizer_dic
import numpy as np
import pandas as pd

disease_classes = ['Apple___Apple_scab',
                   'Apple___Black_rot',
                   'Apple___Cedar_apple_rust',
                   'Apple___healthy',
                   'Blueberry___healthy',
                   'Cherry_(including_sour)___Powdery_mildew',
                   'Cherry_(including_sour)___healthy',
                   'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
                   'Corn_(maize)___Common_rust_',
                   'Corn_(maize)___Northern_Leaf_Blight',
                   'Corn_(maize)___healthy',
                   'Grape___Black_rot',
                   'Grape___Esca_(Black_Measles)',
                   'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
                   'Grape___healthy',
                   'Orange___Haunglongbing_(Citrus_greening)',
                   'Peach___Bacterial_spot',
                   'Peach___healthy',
                   'Pepper,_bell___Bacterial_spot',
                   'Pepper,_bell___healthy',
                   'Potato___Early_blight',
                   'Potato___Late_blight',
                   'Potato___healthy',
                   'Raspberry___healthy',
                   'Soybean___healthy',
                   'Squash___Powdery_mildew',
                   'Strawberry___Leaf_scorch',
                   'Strawberry___healthy',
                   'Tomato___Bacterial_spot',
                   'Tomato___Early_blight',
                   'Tomato___Late_blight',
                   'Tomato___Leaf_Mold',
                   'Tomato___Septoria_leaf_spot',
                   'Tomato___Spider_mites Two-spotted_spider_mite',
                   'Tomato___Target_Spot',
                   'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
                   'Tomato___Tomato_mosaic_virus',
                   'Tomato___healthy']

def homepage(request):
	return render(request, "homepage.html")

def ferti(request):
	return render(request, "ferti.html")

def cropreco(request):
	return render(request, "cropreco.html")

def aboutus(request):
	return render(request, "aboutus.html")

def cropresult(request):
	XB = pickle.load(open('XGBoost.pkl', 'rb'))
	listt = []
	listt.append(float(request.GET['Nitrogen']))
	listt.append(float(request.GET['Phosphorous']))
	listt.append(float(request.GET['Pottasium']))
	listt.append(float(request.GET['Temperature']))
	listt.append(float(request.GET['Humidity']))
	listt.append(float(request.GET['Ph']))
	listt.append(float(request.GET['Rainfall']))
	print(listt)
	data = np.array([listt])
	ans = XB.predict(data)
	return render(request, "cropresult.html", {'ans':ans[0]})

def fertiresult(request):
	ferti_model = pickle.load(open('ferti.pkl', 'rb'))
	cropname = request.GET['crop_name']
	cropname = LabelEncoder().fit_transform([cropname])
	listt = []
	listt.append(cropname)
	listt.append(int(request.GET['nitro']))
	listt.append(int(request.GET['pottas']))
	listt.append(int(request.GET['phospho']))
	print(listt)
	data = np.array([listt])
	ans = ferti_model.predict(data)
	return render(request, "fertiresult.html", {'ans':ans[0]})

def predict_disease(imgfile):
	disease_model_path = 'plant_disease_model.pth'
	disease_model = ResNet9(3, len(disease_classes))
	disease_model.load_state_dict(torch.load(disease_model_path, map_location=torch.device('cpu')))
	disease_model.eval()
	transform = transforms.Compose([transforms.Resize(256),transforms.ToTensor(),])
	byteImgIO = io.BytesIO()
	byteImg = Image.open(imgfile)
	byteImg.save(byteImgIO, "JPEG")
	byteImgIO.seek(0)
	img_t = transform(byteImg)
	img_u = torch.unsqueeze(img_t, 0)
	model = disease_model
	yb = model(img_u)
	_, preds = torch.max(yb, dim=1)
	predict = disease_classes[preds[0].item()]
	return predict

def disease(request):
	# diseasefm = DiseaseForm()
	if request.method == 'POST':
		imgfile = request.FILES.get("myfile")
		prediction = predict_disease(imgfile)
		print("success")
		return render(request, "disease.html", { "msg":prediction})
	else:
		return render(request, "disease.html")

# Create your views here.
