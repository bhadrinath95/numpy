from django.db import models

# Create your models here.
class Topic(models.Model):
    heading = models.CharField(max_length=120)
    created_date = models.DateField()
    urlpath = models.URLField()
    
    def __str__(self):
        return str(self.heading)