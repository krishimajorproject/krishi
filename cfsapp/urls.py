from django.urls import path
from . import views
from django.conf import settings
from django.conf.urls.static import static

# Add URLConf
urlpatterns = [
    path("", views.homepage, name="homepage"),
    path("ferti/", views.ferti, name="ferti"),
    path("cropreco/", views.cropreco, name="cropreco"),
    path("disease/", views.disease, name="disease"),
    path("cropresult/", views.cropresult, name="cropresult"),
    path("aboutus/", views.aboutus, name="aboutus"),
    path("fertiresult/", views.fertiresult, name="fertiresult"),
]