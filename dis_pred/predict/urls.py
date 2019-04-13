from django.urls import path
from .views import train_model, login, add_user_details, predict_diseases

urlpatterns = [
    path('train_model/', train_model),
    path('login/', login),
    path('add_details', add_user_details),
    path('predict_diseases/', predict_diseases),
]
