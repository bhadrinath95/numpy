'''
Created on 04-Jul-2020

@author: BHADRINATH
'''
from django.urls import path, re_path
from . import views

urlpatterns = [
    path('', views.home, name='display'),
    path('about', views.about, name='about'),
    path('types', views.types, name='types'),
    path('initialization', views.initialization, name='initialization'),
    path('insertion', views.insertion, name='insertion'),
    path('maths',views.maths,name='maths'),
    path('arraycomparison',views.array_comparison,name='array_comparison'),
    path('aggregate',views.aggregate,name='aggregate'),
    path('broadcast',views.broadcast,name='broadcast'),
    path('indexslice',views.indexslice,name='indexslice'),
    path('manipulation',views.manipulation,name='manipulation'),
    path('split',views.split,name='split'),
    path('advantage',views.advantage,name='advantage'),
]