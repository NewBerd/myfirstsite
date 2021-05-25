from django.shortcuts import render
from django.http import HttpResponse
from . import spider
from urllib.parse import unquote, quote
import requests
from .neo_db import Neo4j
import json

host = '127.0.0.1'
port = '127.0.0.1:8000'

def index(request):
    return render(request, '1_index.html')

def show_add_actor(request):
    return render(request, 'addactor.html')

def show_add_movie(request):
    return render(request, 'addmovie.html')

def show_add_relation(request):
    return render(request, 'addrelation.html')

def show_del_actor(request):
    return render(request, 'delactor.html')

def show_del_movie(request):
    return render(request, 'delmovie.html')

def get_movie(request):
    movies = spider.movies_main("static/files/movies.json")#注意文件位置的写法，就是这么奇怪,后台调用不是这么写的
    return render(request, "showmovies.html", movies)

def get_actor(request):
    actors = spider.get_all_actors("static/files/movies.json", "static/files/actors.txt")
    return render(request, "showactors.html", actors)

def get_relation(request):
    movies_filename = "static/files/movies.json"
    actors_file = "static/files/actors.txt"
    movies_csv = "static/files/movies.csv"
    actors_csv = "static/files/actors.csv"
    relations_csv = "static/files/relations.csv"
    context = spider.get_relation(movies_filename, actors_file, movies_csv, actors_csv, relations_csv)
    return render(request, "relations.html", context)

def add_base_kg_from_csv(request):
    global host
    global port
    actors_file = "http://%s/static/files/actors.csv" % (port)#协议与文件位置
    movies_file = "http://%s/static/files/movies.csv" % (port)
    relations_file = "http://%s/static/files/relations.csv" % (port)

    graph = Neo4j()
    graph.connect(host, "neo4j", "123456")
    graph.add_actors_from_csv(actors_file)
    graph.add_movies_from_csv(movies_file)
    graph.add_relations_from_csv(relations_file)

    html = "successfully add a base movie and actor knowledge graph."
    return HttpResponse(html)

def add_actor(request):
    request.encoding = 'utf-8'
    if 'name' in request.GET:
        name = request.GET['name']
        global host
        graph = Neo4j()
        graph.connect(host, 'neo4j', '123456')
        html = graph.add_actor(name)
        return HttpResponse(html)

def add_movie(request):
    request.encoding = 'utf-8'
    if 'name' in request.GET:
        name = request.GET['name']
        global host
        graph = Neo4j()
        graph.connect(host, 'neo4j', '123456')
        html = graph.add_movie(name)
        return HttpResponse(html)

def add_relation(request):
    request.encoding = 'utf-8'
    actor = request.GET['actor']
    movie = request.GET['movie']
    global host
    graph = Neo4j()
    graph.connect(host, 'neo4j', '123456')
    state = graph.add_relation(actor, movie)
    if state == 2:
        html = "the relation has been existde"
    if state == 1:
        html = "successfully create a relation on actor %s and movie %s " % (actor, movie)
    if state == -1:
        html = "cannot create the relation because we cannot find the actor %s" % (actor)
    if state == 0:
        html = "cannot create the relation because we cannot find the movie %s" % (movie)
    return HttpResponse(html)

def delete_actor(request):
    request.encoding = 'utf-8'
    actor = request.GET['actor']
    global host
    graph = Neo4j()
    graph.connect(host, 'neo4j', '123456')
    state = graph.delete_actor(actor)
    if state == -1:
        html = "the actor does not exist"
    if state == 0:
        html = "successfully delete an actor"
    if state == 1:
        html = "successfully delete an actor and a relation on it"

    return HttpResponse(html)


def delete_movie(request):
    request.encoding = 'utf-8'
    movie = request.GET['movie']
    global host
    graph = Neo4j()
    graph.connect(host, 'neo4j', '123456')
    state = graph.delete_movie(movie)
    if state == -1:
        html = "the movie does not exist"
    if state == 0:
        html = "successfully delete a movie"
    if state == 1:
        html = "successfully delete a movie and a relation on it"

    return HttpResponse(html)

def show_update_actor(request):
    context = {'name':'演员'}#传入一个字典作为参数
    return render(request, 'update-actor.html', context)#模板里通过引用字典的键使用参数值

def show_update_movie(request):
    context = {'name':'电影'}
    return render(request, 'update-movie.html', context)

def update_actor(request):
    request.encoding = 'utf-8'
    sub_dict = request.GET#request.GET是一个类字典对象，是字典的子类
    # s = ""
    # for k, v in request.GET.items():
    #     s += k+':'+v+'<br>'
    global host
    graph = Neo4j()
    graph.connect(host, 'neo4j', '123456')
    html = graph.update_actor(sub_dict)
    return HttpResponse(html)

def update_movie(request):
    request.encoding = 'utf-8'
    sub_dict = request.GET#request.GET是一个类字典对象，是字典的子类
    # s = ""
    # for k, v in request.GET.items():
    #     s += k+':'+v+'<br>'
    global host
    graph = Neo4j()
    graph.connect(host, 'neo4j', '123456')
    html = graph.update_movie(sub_dict)
    return HttpResponse(html)

def show_search(request):
    return render(request, 'search.html')

def search_actor(request):
    request.encoding = 'utf-8'
    actor = request.GET['actor']
    global host

    graph = Neo4j()
    graph.connect(host, 'neo4j', '123456')
    actor_dict = graph.search_actor(actor)#返回值是一个字典
    if actor_dict:
        context = {}
        context["object"] = actor_dict
        data = []#nodes
        links = []#edges
        for k, v in actor_dict.items():
            for item in v.split("、"):
                if item:
                    # dic = {}#点
                    # dic["name"] = "%s" % (item)
                    # dic["symbolSize"] = 60 #点的大小
                    # dic = "{name:'%s',symbolSize:60}" % (item)
                    node_dic = {}#点
                    node_dic['name'] = item
                    if item == actor_dict['中文名']:
                        node_dic['category'] = 1
                    else:
                        node_dic['category'] = 3
                    data.append(node_dic)

                    dit = {}#边
                    if not item == actor_dict["中文名"]:#不建立指向自己的边
                        dit['source'] = "%s" % (actor_dict["中文名"])
                        dit['target'] = "%s" % (item)
                        links.append(dit)

        # context['data'] = json.dumps(data, ensure_ascii=False)
        context['nodes'] = data
        # context['links'] = json.dumps(links, ensure_ascii=False)
        context['edges'] = links
        return render(request, 'xx.html', context)
    return HttpResponse("cannot find THE actor")

def search_movie(request):
    request.encoding = 'utf-8'
    movie = request.GET['movie']
    global host

    graph = Neo4j()
    graph.connect(host, 'neo4j', '123456')
    movie_dict = graph.search_movie(movie)#返回值是一个字典
    if movie_dict:
        context = {}
        context["object"] = movie_dict
        data = []
        links = []
        for k, v in movie_dict.items():
            for item in v.split(","):
                if item:
                    node_dic = {}#点
                    node_dic['name'] = item
                    if item == movie_dict['中文名']:#如果该点表示电影名，则为第二类
                        node_dic['category'] = 2
                    else:
                        node_dic['category'] = 3
                    data.append(node_dic)
                    
                    link = {}#边
                    if not item == movie_dict['中文名']:
                        link['source'] = movie_dict['中文名']
                        link['target'] = item
                        links.append(link)
        context['nodes'] = data
        context['edges'] = links
        return render(request, 'xx.html', context)
    return HttpResponse("cannot find THE movie")

def search_allActorsinMovie(request):
    request.encoding = 'utf-8'
    movie = request.GET['movie']
    global host

    graph = Neo4j()
    graph.connect(host, 'neo4j', '123456')
    movies_list = graph.search_allActorsinMovie(movie)#返回值是一个列表，元素是字典

    if movies_list:
        data = []
        links = []
        first = True
        for dic in movies_list:
            #添加节点
            if first:
                data.append({'name':dic['m.中文名'], 'category':2})
                first = False
            data.append({'name':dic['a.中文名'], 'category':1})
            #添加边
            link = {}
            link['source'] = dic['m.中文名']
            link['target'] = dic['a.中文名']
            links.append(link)
        # data = set(data)#去重
        # data = list(data)

        context = {}
        context['nodes'] = data
        context['edges'] = links
        return render(request, '00.html', context)
    return HttpResponse("cannot find THE movie")

def search_allMoviesSomeoneAct(request):
    request.encoding = 'utf-8'
    actor = request.GET['actor']
    global host

    graph = Neo4j()
    graph.connect(host, 'neo4j', '123456')
    actors_list = graph.search_allMoviesSomeoneAct(actor)#返回值是一个列表，元素是字典

    if actors_list:
        data = []
        links = []
        first = True
        for dic in actors_list:
            #添加节点
            if first:
                data.append({'name':dic['a.中文名'], 'category':1})
                first = False
            data.append({'name':dic['m.中文名'], 'category':2})
            #添加边
            link = {}
            link['source'] = dic['a.中文名']
            link['target'] = dic['m.中文名']
            links.append(link)
        # data = set(data)#去重
        # data = list(data)

        context = {}
        context['nodes'] = data
        context['edges'] = links
        return render(request, '00.html', context)
    return HttpResponse("cannot find THE actor")

def search_friends(request):
    request.encoding = "utf-8"
    actor = request.GET["actor"]
    global host

    graph = Neo4j()
    graph.connect(host, 'neo4j', '123456')
    friends = graph.search_friends(actor)

    if friends:
        data = []
        links = []
        for info in friends:
            # for k, v in info.items():
            #     data.append(v)
            if not {'name':info['a.中文名'], 'category':1} in data:#如果节点不在列表中，那就添加
                data.append({'name':info['a.中文名'], 'category':1})
            if not {'name':info['m.中文名'], 'category':2} in data:
                data.append({'name':info['m.中文名'], 'category':2})
            if not {'name':info['f.中文名'], 'category':1} in data:
                data.append({'name':info['f.中文名'], 'category':1})

            link = {}
            link['source'] = info['a.中文名']
            link['target'] = info['m.中文名']
            links.append(link)
            link = {}
            link['source'] = info['f.中文名']
            link['target'] = info['m.中文名']
            links.append(link)  
        # data = set(data)#去重
        # data = list(data)

        context = {}
        context['nodes'] = data
        context['edges'] = links
        return render(request, '00.html', context)
    return HttpResponse("cannot find THE actor")  

def search_actorbyguoji(request):
    request.encoding = 'utf-8'
    guoji = request.GET['guoji']
    global host

    graph = Neo4j()
    graph.connect(host, 'neo4j', '123456')
    actors = graph.search_actorbyguoji(guoji)

    if actors:
        data = []
        links = []
        for info in actors:
            # for k, v in info.items():
            #     data.append(v)
            if not {'name':info['a.中文名'], 'category':1} in data:#如果节点不在列表中，那就添加
                data.append({'name':info['a.中文名'], 'category':1})
            if not {'name':info['a.国籍'], 'category':3} in data:
                data.append({'name':info['a.国籍'], 'category':3})
            link = {}
            link['source'] = info['a.中文名']
            link['target'] = info['a.国籍']
            links.append(link)  
        # data = set(data)#去重
        # data = list(data)

        context = {}
        context['nodes'] = data
        context['edges'] = links
        return render(request, '00.html', context)
    return HttpResponse("cannot find any actor")

def search_actorbyxingzuo(request):
    request.encoding = 'utf-8'
    xingzuo = request.GET['xingzuo']
    global host

    graph = Neo4j()
    graph.connect(host, 'neo4j', '123456')
    actors = graph.search_actorbyxingzuo(xingzuo)

    if actors:
        data = []
        links = []
        for info in actors:
            # for k, v in info.items():
            #     data.append(v)
            if not {'name':info['a.中文名'], 'category':1} in data:#如果节点不在列表中，那就添加
                data.append({'name':info['a.中文名'], 'category':1})
            if not {'name':info['a.星座'], 'category':3} in data:
                data.append({'name':info['a.星座'], 'category':3})
            link = {}
            link['source'] = info['a.中文名']
            link['target'] = info['a.星座']
            links.append(link)  
        # data = set(data)#去重
        # data = list(data)

        context = {}
        context['nodes'] = data
        context['edges'] = links
        return render(request, '00.html', context)
    return HttpResponse("cannot find any actor")

def search_actorbyachievement(request):
    request.encoding = 'utf-8'
    ach = request.GET['achievement']
    global host

    graph = Neo4j()
    graph.connect(host, 'neo4j', '123456')
    actors = graph.search_actorbyachievement(ach)

    if actors:
        data = []
        links = []
        for info in actors:
            for k, v in info.items():
                for item in v.split("、"):
                    if item and (item==info['a.中文名']):#如果是人物节点，分为1类
                        if not {'name':item, 'category':1} in data:
                            data.append({'name':item, 'category':1})

                    if item and (not item == info['a.中文名']):#如果是属性节点，则需要分为3类，并且建立一条到
                        #人物节点的边
                        if not {'name':item, 'category':3} in data:
                            data.append({'name':item, 'category':3})
                        link = {}
                        link['source'] = info['a.中文名']
                        link['target'] = item 
                        links.append(link)  
        # data = set(data)#去重
        # data = list(data)

        context = {}
        context['nodes'] = data
        context['edges'] = links
        return render(request, '00.html', context)
    return HttpResponse("cannot find any actor")

def search_moviebyscore(request):
    request.encoding = 'utf-8'
    low = request.GET['low']
    high = request.GET['high']
    low = eval(low)
    high = eval(high)
    global host

    graph = Neo4j()
    graph.connect(host, 'neo4j', '123456')
    movies = graph.search_moviebyscore(low, high)

    data = []
    links = []
    for info in movies:
        score = float(info['m.得分'])
        if score >= low and score <= high:
            if not {'name':info['m.中文名'], 'category':2} in data: 
                data.append({'name':info['m.中文名'], 'category':2})
            if not {'name':info['m.得分'], 'category':3} in data: 
                data.append({'name':info['m.得分'], 'category':3})

            link = {}
            link['source'] = info['m.中文名']
            link['target'] = info['m.得分'] 
            links.append(link)  
    # data = set(data)#去重
    # data = list(data)

    context = {}
    context['nodes'] = data
    context['edges'] = links
    return render(request, '00.html', context)
