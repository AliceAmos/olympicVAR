from django.shortcuts import render
from django.http import Http404
from rest_framework.views import APIView
from rest_framework import permissions
from rest_framework.response import Response
from rest_framework import status
from .models import  *
from .serializers import VideoSerializer
from rest_framework.generics import ListAPIView
from rest_framework.parsers import FileUploadParser
from rest_framework.exceptions import ParseError
from rest_framework.decorators import api_view
import json
import pickle
import pandas as pd
from .predictVid import predict

FRAMES_PER_VID = 103

def AllVideos():
    # base_url = 'https://res.cloudinary.com/dtedceoh4/video/upload/v1655144541/'
    Link = VideoUpload.objects.values('video').filter(score=None)
    stu = VideoUpload.objects.filter(score=None)
    serializer = VideoSerializer(stu, many=True)
    jFormat = json.dumps(serializer.data)
    pFormat = json.loads(jFormat)
    for value in pFormat:
        loaded_model_path = '/Users/I555250/mini_proj/backend/api/model/random_forest_model.sav'
        loaded_model = pickle.load(open(loaded_model_path, 'rb'))
        frames_predicton_model_path = '/Users/I555250/mini_proj/backend/api/model/extended.model'
        full_url = str(value.get('video'))
        new_vid_scores = predict(full_url, frames_predicton_model_path)
        new_vid_scores = new_vid_scores[:103]
        vid_df = pd.DataFrame(data=[new_vid_scores])
        y_new_vid = loaded_model.predict(vid_df)
        print("Predicted score: ", y_new_vid)
        videoscore = VideoUpload.objects.filter(id=value.get('id')).update(score=str(float(y_new_vid[0])))

    
class VideoCreate(APIView):
    def get(self, request, pk=None, format=None):
        if pk is not None:
            video = VideoUpload.objects.get(id=pk)
            serializer = VideoSerializer(video)
            return Response(serializer.data)
        AllVideos()
        stu = VideoUpload.objects.all().order_by('-id').first()
        serializer = VideoSerializer(stu)
        return Response(serializer.data)
    
    def post(self, request, format=None):
        serializer = VideoSerializer(data=request.data)
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data, status=status.HTTP_201_CREATED)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
    

