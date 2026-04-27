from django.urls import path

from . import views


urlpatterns = [
    path("", views.admin_dashboard, name="admin_dashboard"),
    path("review/<int:pk>/", views.review_application, name="review_application"),
]
