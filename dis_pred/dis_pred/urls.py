from django.contrib import admin
from django.urls import path, re_path, include
from predict import views

urlpatterns = [
    path('admin/', admin.site.urls),
    re_path('^', include('predict.urls')),
]
