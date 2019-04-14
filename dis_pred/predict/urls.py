# from django.urls import path
from django.conf.urls import url
from .views import train_model, login, add_user_details, predict_diseases, predict_symptoms, generate_prescription

urlpatterns = [
    url('train_model/', train_model),
    url('login/', login),
    url('add_details', add_user_details),
    url('predict_diseases/', predict_diseases),
    url('predict_symptoms/', predict_symptoms),
    url('generate_prescription/',generate_prescription),
]
