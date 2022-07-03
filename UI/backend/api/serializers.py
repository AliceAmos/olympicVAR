from django.urls import path, include
from rest_framework import serializers
from .models import VideoUpload

class VideoSerializer(serializers.ModelSerializer):
    class Meta:
        model = VideoUpload
        fields = ('id','video','score')
    
    def create(self, validated_data):
        return VideoUpload.objects.create(**validated_data)