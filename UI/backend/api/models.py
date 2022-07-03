from django.db import models
from cloudinary_storage.storage import VideoMediaCloudinaryStorage
from cloudinary_storage.validators import validate_video

class VideoUpload(models.Model):
    video = models.FileField(upload_to='videos/', blank=False, storage=VideoMediaCloudinaryStorage())
    score = models.CharField(max_length=50,blank=True,null=True)