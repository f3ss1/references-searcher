from django.urls import path
from .views import predict_references

urlpatterns = [
    path("", predict_references, name="predict_references"),
]
