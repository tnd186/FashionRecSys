from django.contrib import admin
from django.urls import path, include
from django.http import HttpResponse

def home(request):
    return HttpResponse("Welcome to the fashion web project!")

urlpatterns = [
    path('admin/', admin.site.urls),
    path('api/', include('fashion_web_app.urls')),  
    path('', home),  
]