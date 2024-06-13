from django.urls import path
from .views import predict_references, task_status

urlpatterns = [
    path("", predict_references, name="predict_references"),
    path("status", task_status, name="task_status"),
]
