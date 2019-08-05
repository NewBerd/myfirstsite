"""MyDjangoProject URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/2.0/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path, include
from . import views, deepviews
urlpatterns = [
    path('admin/', admin.site.urls),
    path('FTQ/', views.index),

    path("FTQ/get-movie/", views.get_movie),
    path("FTQ/get-actor/", views.get_actor),
    path("FTQ/get-relation/", views.get_relation),

    path('FTQ/add-base-KG/', views.add_base_kg_from_csv),
    path('FTQ/add-actor/', views.show_add_actor),
    path('FTQ/addactor/', views.add_actor),
    path('FTQ/add-movie/', views.show_add_movie),
    path('FTQ/addmovie/', views.add_movie),
    path('FTQ/add-relation/', views.show_add_relation),#注意，千万不要忘了最后的斜杠'/'和逗号
    path('FTQ/addrelation/', views.add_relation),

    path('FTQ/delete-actor/', views.show_del_actor),
    path('FTQ/delactor/', views.delete_actor),  
    path('FTQ/delete-movie/', views.show_del_movie),  
    path('FTQ/delmovie/', views.delete_movie),

    path('FTQ/update-actor/', views.show_update_actor),
    path('FTQ/updateactor/', views.update_actor),
    path('FTQ/update-movie/', views.show_update_movie),
    path('FTQ/updatemovie/', views.update_movie),

    path("FTQ/search-info/", views.show_search),
    path('FTQ/search-actor/', views.search_actor),
    path('FTQ/search-movie/', views.search_movie),
    path("FTQ/search-allMoviesSomeoneAct/", views.search_allMoviesSomeoneAct),
    path("FTQ/search-allActorsinMovie/", views.search_allActorsinMovie),
    path("FTQ/search-friends/", views.search_friends),
    # path("FTQ/search-shortpath/", views.search_shortpath),
    path("FTQ/search-actorbyxingzuo/", views.search_actorbyxingzuo),
    path("FTQ/search-actorbyguoji/", views.search_actorbyguoji),
    path("FTQ/search-actorbyachievement/", views.search_actorbyachievement),
    path("FTQ/search-moviebyscore/", views.search_moviebyscore),

    



    
    #deeplearning
    path("deeplearning/logistic/", deepviews.show_logistic),
    path("deeplearning/logistic/source-code/", deepviews.logistic_source_code),
    path("deeplearning/logistic/uploadpicture/", deepviews.uploadpicture),
    path("deeplearning/logistic/prepicture/", deepviews.log_pre),

    path("deeplearning/mlp/", deepviews.mlp_brief_intro),
    path("deeplearning/mlp/source-code/", deepviews.mlp_source_code),
    path("deeplearning/mlp/uploadpicture/", deepviews.uploadpicture),
    path("deeplearning/mlp/prepicture/", deepviews.mlp_pre),

    path("deeplearning/cnn/", deepviews.cnn_brief_intro),
    path("deeplearning/cnn/source-code/", deepviews.cnn_source_code),
    path("deeplearning/cnn/uploadpicture/", deepviews.uploadpicture),
    path("deeplearning/cnn/prepicture/", deepviews.cnn_pre),

    path("deeplearning/rnn/", deepviews.rnn_brief_intro),
    path("deeplearning/rnn/source-code/", deepviews.rnn_source_code),
    path("deeplearning/rnn/uploadpicture/", deepviews.uploadpicture),
    path("deeplearning/rnn/prepicture/", deepviews.rnn_pre),

    path("deeplearning/brief/", deepviews.con_brief),
    path("deeplearning/con/uploadpicture1/", deepviews.uploadpicture1),
    path("deeplearning/con/uploadpicture/", deepviews.uploadpicture),
    path("deeplearning/con/prepicture/", deepviews.con_pre),
    path("deeplearning/con/genpicture/", deepviews.con_gen),

]
