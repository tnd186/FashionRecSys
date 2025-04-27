from django.urls import path
from .views import chatbot_response
from .views import similarity_fashion_response

urlpatterns = [
    path('chatbot_response/', chatbot_response, name='chatbot_response'),
    path('similarity_fashion_response/', similarity_fashion_response, name='similarity_fashion_response'),
]