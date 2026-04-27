from django.urls import path
from . import views

urlpatterns = [
    path('apply/', views.submit_application, name='submit_application'),
    path('apply/<int:pk>/documents/', views.upload_documents, name='upload_documents'),
    path('status/', views.application_status, name='application_status'),
]