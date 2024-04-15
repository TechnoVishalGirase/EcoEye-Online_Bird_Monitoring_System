from django.urls import path
from . import views

urlpatterns = [
    path('',views.index, name ="index"),
    path('error/',views.error_, name = 'error'),
    path('about/',views.about, name = "about"),
    path('booking/',views.booking, name = "booking"),
    path('bird_tracking/', views.bird_tracking, name="bird_tracking"),
    path('destination/',views.destination, name = "destination"),
    path('service/',views.service, name = "service"),
    path('package/',views.package, name = "package"),
    path('team/',views.team, name = "team"),
    path('testimonial/',views.testimonial, name = "testimonial"),
    path('contact/',views.contact, name = "contact"),
    path('predict/', views.predict, name="predict"),
]
