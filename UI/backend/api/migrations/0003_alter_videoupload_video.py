# Generated by Django 3.2.12 on 2022-06-01 15:43

import cloudinary_storage.storage
from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('api', '0002_alter_videoupload_video'),
    ]

    operations = [
        migrations.AlterField(
            model_name='videoupload',
            name='video',
            field=models.FileField(storage=cloudinary_storage.storage.VideoMediaCloudinaryStorage(), upload_to='videos/'),
        ),
    ]
