from django.urls import path
from .views import VideoCreate
from django.conf import settings
from django.conf.urls.static import static

urlpatterns = [
    path('', VideoCreate.as_view(), name='postVideo'),
    path('<int:pk>', VideoCreate.as_view(), name='postVideo'),
]+static(settings.MEDIA_URL,document_root=settings.MEDIA_ROOT)
