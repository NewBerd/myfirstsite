import json
import os
from urllib.parse import unquote, quote
import requests
import re
from bs4 import BeautifulSoup
import bs4
import time

import pymysql
import csv
from collections import Counter

def get_one_movie_page(url):
    try:
        headers={
            'User-Agent':'Mozilla/5.0(Macintosh;Intel Mac OS X 10_13_3) \
            AppleWebKit/537.36(KHTML,like Gecko) \
            Chrome/65.0.3325.162 Safari/537.36'
        }#设置代理浏览器
        response=requests.get(url,headers=headers)#提交请求
        if response.status_code==200:
            response.encoding = 'utf-8'
            return response.text#返回内容
        return None
    except RequestException:
        print(RequestException)
        return None

def parse_one_movie_page(html):
    pattern=re.compile(
        r'<dd>.*?board-index.*?>(.*?)</i>.*?data-src="(.*?)".*?name.*?a.*?>(.*?)</a>.*?star.*?>(.*?)</p>.*?' +
        r'releasetime">(.*?)</p>.*?integer">(.*?)</i>.*?fraction.*?>(.*?)</i>.*?</dd>',re.S)
        #跨行字符串使用反斜杠\，此处不可使用，否则不出结果
    items=re.findall(pattern,html)
    for item in items:#使用生成器，减少内存占用
        yield {
            '排名':item[0],
            '图片':item[1],
            '中文名':item[2].strip(),#去掉首尾的空白符
            '主演':item[3].strip()[3:] if len(item[3])>3 else '',#去掉'主演：'这三个字符
            '上映时间':item[4].strip()[5:] if len(item[4])>5 else '',#同样地，去掉'上映时间:'这五个字符
            '得分':item[5].strip()+item[6].strip()#分数被拆开了，现在将他们合在一起
        }
#     print(items)

def save_movie_to_json(content_dict, filename):
#     print(content_dict)
#     print(json.dumps(content_dict,ensure_ascii=False))#需要使用ensure_ascii参数，否则会产生乱码
    with open(filename,'a',encoding='utf-8') as f:
        # f.writelines(content_list)
    # 必须使用追加模式，不过使用时也要注意
        json.dump(content_dict, f, ensure_ascii=False)
        f.write('\n')#每写入一个电影，加一个换行
    # with open('maoyan.json','a',encoding='utf-8') as f:
    #     f.write(json.dumps(content_dict,ensure_ascii=False)+'\n')

def get_movies(filename):
    url_base='http://maoyan.com/board/4'
    movies = {"movies":[]}
    for i in range(10):
        if i==0:
            url=url_base
        else:
            url=url_base+'?offset='+str(i*10)
        html=get_one_movie_page(url)

        for item in parse_one_movie_page(html):
            movies["movies"].append(item)
            save_movie_to_json(item, filename)
        time.sleep(0.2)#防止爬取太快被禁，睡一秒
    return movies
####################################################
#以下是处理演员的
##
def get_one_page(url):
    try:
        headers={
            'User-Agent':'Mozilla/5.0(Macintosh;Intel Mac OS X 10_13_3) \
            AppleWebKit/537.36(KHTML,like Gecko) \
            Chrome/65.0.3325.162 Safari/537.36'
        }#设置代理浏览器
        response=requests.get(url,headers=headers)#提交请求
        if response.status_code==200:
            response.encoding = 'utf-8'
            return response.text#返回内容
        return None
    except RequestException:
        print(RequestException)
        return None

def parse_html(html):
    '''解析的部分有点复杂，靠注释也说不清楚，必须看到真实的html，才可以理解'''
    soup = BeautifulSoup(html, 'lxml')
    div = soup.find(name='div', attrs={'class': 'basic-info cmn-clearfix'})
    if div:
        d1 = div.contents
    #     d1.remove('\n')
        dict_ = {}
        key = []
        value = []
        for d in d1:
            if isinstance(d, bs4.element.Tag):
        #         print('------------------------------------------------')
        #         print(d.prettify())
                for item in d.find_all(name='dt'):#属性名
                    key.append(re.sub(r'\s*', '', item.string))
        #             print('属性名',re.sub(r'\s*', '', item.string))
                for tag in d.find_all(name='dd'):#遍历所有的dd标签
                #     print(type(tag))
                #     print(tag)
                #     re.sub(r'.*(<br/>).*', '、', tag)
                    s = ''
                    for item in tag.contents:#dd标签的孩子
                        if not item == '\n':
                            if isinstance(item, bs4.element.Tag) and item.name == 'a':#dd下有嵌套，但是我们只要a标签
                #                 print('+', item, '+')
                                for item in item.contents:#遍历子节点
                                    if isinstance(item, bs4.element.Tag):#子节点是标签
                #                         print('00', item, '00')
                                        if item.string:
                                            s += item.string
                                    elif isinstance(item, bs4.element.NavigableString):#子节点是字符串
                                        s += item
                #                 print('=', re.sub(r'\s*','',s))#去除空白符
                            elif isinstance(item, bs4.element.NavigableString):#没有嵌套
                                s += item
                #                 print('-', re.sub(r'\s*', '', item), '-')
                #     print(re.sub(r'\s*', '', s))
        #             value.append(s.split('\n'))
        #             value.append(s.replace('\n', '、'))
                    value.append(s)
        #             print(s.split('\n'))
        dict_ = dict(zip(key,value))
        for key, value in dict_.items():   
            dict_[key] = value.strip()
            if key == '主要成就':
                dict_['主要成就'] = dict_['主要成就'].replace('\n', '、').rstrip('、收起')
        return dict_

def save_dict(dict, filename):
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(dict, f, ensure_ascii=False)

def get_person(name):
    base_url = 'https://baike.baidu.com/item/'
    url = base_url + quote(name)
    try:
        html = get_one_page(url)
        if html:#如果可以获取到页面，进行解析
            per_info = parse_html(html)#返回一个字典
            return per_info
        else:
            return None
    except:
        return None

def get_html(name):
    base_url = 'https://baike.baidu.com/item/'
    url = base_url + quote(name)
    try:
        html = get_one_page(url)
        if html:#如果可以获取到页面，进行解析
            return html
        else:
            return "<p><b>cannot find the actor</b></p>"
    except:
        return "<p><b>cannot find the actor</b></p>"


def get_all_actors(movies_filename, actors_file):
   #从电影信息中获取所有的电影主要演员
    actors = []
    with open(movies_filename, 'r', encoding='utf8') as f:
        for line in f.readlines():
            dict_ = json.loads(line)#加载为字典
            actor = dict_['主演']#获取电影的主演
            actor = actor.split(',')
            actors.extend(actor)#
    actors = set(actors)#因为会出现一个演员演过不止一部电影的情况，所以使用集合去重
    actors = list(actors) 

    #爬取演员百科信息，并提取其中的基本信息，保存为json
    actors_dict = {}
    for name in actors:
        base_url = 'https://baike.baidu.com/item/'
        url = base_url + quote(name)
        # url = 'https://baike.baidu.com/item/费雯·丽/40031?fr=aladdin'
        print(name)
        try:
            html = get_one_page(url)#获取页面
            dic = parse_html(html)#解析
            if dic:#如果可以解析出信息
                actors_dict[name] = dic
        except:
            pass    
        time.sleep(0.1)
    #     print(dic)
    #     print('----------------------')
    save_dict(actors_dict, actors_file)
    actors = {}
    actors["actors"] = actors_dict
    return actors

def json2list(filename):
    '''读取猫眼json文件，将其值转换为列表'''
    maoyan_list = []
    first = True
    with open(filename, 'r', encoding='utf8') as f:
        for line in f.readlines():
            header = []
            row = []
            dic = json.loads(line)
            if first:
                maoyan_list.append(list(dic))
                maoyan_list.append(list(dic.values()))
                first = False
            else:
                maoyan_list.append(list(dic.values()))
    return maoyan_list

def save_list(maoyan_list, filename):
    '''将list存为csv文件'''
    with open(filename, 'w', newline='', encoding='utf8') as f:
        writer = csv.writer(f)
        writer.writerows(maoyan_list)

def get_main_info(filename):
    '''从文件中获取需要存数据库的信息，返回一个嵌套列表，每一行是一条记录'''
    with open(filename, 'r', encoding='utf8') as f:
        actors = json.load(f)
    col_name = []#保存所有的属性（列）
    for key, value in actors.items():
#         print(key)
        if not value == None:
            col_name.extend(list(value.keys()))#注意此处不是append
    dic = Counter(col_name)#统计每个属性出现的次数
    attrs = dic.most_common(10)#取出现次数最多的10个属性,[(attr1,k1),(attr2,k2)...]，
    #因为大多数属性都不是演员共有的，
    #都存数据库会造成很多空值出现，浪费空间。其实这个10可以设为一个参数，不用写死
    actors_list = []
    header = ['演员ID']#表头
    for item in attrs:
        header.append(item[0])
    actors_list.append(header)#将表头加入表中
    id_ = 1
    for key, value in actors.items():
        if value:
            row = [id_]#添加id字段
            for item in header[1:]:#对于每一个属性，抽取其属性值
                if item == '中文名':#之所以这样处理是因为有的演员在百科的基本信息中没有中文名这个属性，
                #但是中文名是存在的
                    row.append(key)#用key代替中文名属性
                else:
                    row.append(value.get(item, 'null'))
            actors_list.append(row)#添加一条记录
            id_ += 1
    return actors_list

def get_relation(movies_filename, actors_file, movies_csv, actors_csv, relations_csv):
    data = {}
    #将电影数据处理成符合neo4j-import的格式，并保存为csv文件
    list_movies = json2list(movies_filename)
    first = True
    for line in list_movies:
        if first:
            line[0] = 'movieId'
            line.append(':LABEL')
            first = False
        else:
            line[0] = 'mv'+line[0]
            line.append('movie')
    data["movies"] = list_movies
    save_list(list_movies, movies_csv)
    
    #获取常见属性
    actors = get_main_info(actors_file)
    #获取演员及其id的映射字典
    actor2id = {}
    for line in actors[1:]:
        actor2id[line[1]] = 'p' + str(line[0])
    
    #获取演员与电影之间的关系
    relations = []
    header = [':START_ID', ':END_ID', ':TYPE']
    relations.append(header)
    for line in list_movies[1:]:
        for actor in line[3].split(','):
            rela = []        
            try:
                rela.append(actor2id[actor])#因为有的演员信息百科没有
                rela.append(line[0])
                rela.append('ACTED_IN')
                relations.append(rela)
            except KeyError:
                pass     
    #保存关系
    data["rela"] = relations
    save_list(relations, relations_csv)

    #处理演员信息，并保存
    first = True
    for line in actors:
        if first:
            line[0] = 'personId'
            line.append(':LABEL')
            first = False
        else:
            line[0] = 'p' + str(line[0])
            line.append('actor')
    data["actors"] = actors
    save_list(actors, actors_csv)
    context = {}
    context["data"] = data#为了传递给render，必须再封装一层
    return context

def movies_main(movies_filename):
    if os.path.exists(movies_filename):#因为是追加模式，所以每一次操作文件都必须保证该文件为空（不存在）
        os.remove(movies_filename)
    return get_movies(movies_filename)

def main():
    movies_filename = "../static/files/movies.json"
    actors_file = "../static/files/actors.txt"
    movies_csv = "../static/files/movies.csv"
    actors_csv = "../static/files/actors.csv"
    relations_csv = "../static/files/relations.csv"

    # movies_main(movies_filename)

    # get_all_actors(movies_filename, actors_file)

    # get_relation(movies_filename, actors_file, movies_csv, actors_csv, relations_csv)

if __name__ == '__main__':
    main()
