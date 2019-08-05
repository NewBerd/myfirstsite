from py2neo import Graph, Node, Relationship, cypher, Path
from py2neo.database import ClientError
from . import spider
# import neo4j
class Neo4j():
    graph = None#定义一个类变量
    def __init__(self):#构造函数
        print("create neo4j class ...")

    def connect(self, host, usr, password):#连接数据库
        self.graph = Graph(host=host, auth=(usr, password))

    def add_actor(self, name):
        sql = "match (p:actor {中文名: '%s'}) with p return p" % (name)
        data = self.graph.run(sql).data()
        if not data:#如果不存在该演员
            actor_info = spider.get_person(name)#获取演员信息
            if actor_info:#如果获取到该演员信息
                actor_info['中文名'] = name#进行对齐，因为在测试时发现，通过张国荣添加的节点，其中文名不是张国荣
                #通过战狼2添加的节点，其中文名是战狼Ⅱ，这样就出现了歧义
                s = ""
                first = True
                for key, value in actor_info.items():#构造查询语句
                    if first:   
                        s += key
                        first = False
                    else:
                        s += ', ' + key
                    s += " : '" + value + "'"

                sql = "create (p:actor {%s})" % (s)
                self.graph.run(sql)
                return "successfully add the %s\'s infomation" % (name)
            else:
                return "cannot find the %s\'s infomation" % (name)
        else:
            return "the actor %s has been existed" % (name)

    def add_movie(self, name):
        sql = "match (p:movie {中文名: '%s'}) with p return p" % (name)
        data = self.graph.run(sql).data()
        if not data:#如果不存在该演员
            movie_info = spider.get_person(name)#获取演员信息
            if movie_info:#如果获取到该电影信息
                movie_info['中文名'] = name
                s = ""
                first = True
                for key, value in movie_info.items():#构造查询语句
                    if first:   
                        s += key
                        first = False
                    else:
                        s += ', '
                        s += key
                    s += ' : \''
                    s += value
                    s += '\''
                sql = "create (p:movie {%s})" % (s)
                self.graph.run(sql)
                return "successfully add the %s\'s infomation" % (name)
            else:
                return "cannot find the %s\'s infomation" % (name)
        else:
            return "the actor %s has been existed" % (name)

    def add_relation(self, actor, movie):
        sqla = "match (a:actor{中文名: '%s'}) return a" % (actor)
        sqlm = "match (m:movie{中文名: '%s'}) return m" % (movie)
        sqlr = "match (a:actor{中文名: '%s'})-[r]-(m:movie{中文名: '%s'}) return r" % (actor, movie)
        sqlc = "match (a:actor{中文名: '%s'}), (m:movie{中文名: '%s'}) \
                with a, m create (a)-[:ACTED_IN]->(m)" % (actor, movie)

        a = self.graph.run(sqla).data()
        m = self.graph.run(sqlm).data()
        r = self.graph.run(sqlr).data()

        if not a:#如果不存在该演员则添加
            state = self.add_actor(actor)
            if state == "cannot find the %s\'s infomation" % (actor):#如果找不到信息，则返回失败信息
                return -1
        if not m:
            state = self.add_movie(movie)
            if state == "cannot find the %s\'s infomation" % (movie):
                return 0

        if not r:#如果不存在关系，则创建关系
            self.graph.run(sqlc)
            return 1
        else:
            return 2#如果关系已经存在，则返回

    def delete_actor(self, name):
        sql0 = "match (p:actor {中文名: '%s'})-[r]-(m) delete r,p,m" % (name)#
        sql1 = "match (p:actor {中文名: '%s'}) delete p" % (name)#删除节点
        sql2 = "match (p:actor {中文名: '%s'})-[r]-(m) delete p,r" % (name)#删除节点与节点上的边

        sql = "match (p:actor {中文名: '%s'}) with p return p" % (name)
        data = self.graph.run(sql).data()
        if not data:#如果不存在该节点
            return -1

        try:
            self.graph.run(sql1)#尝试删掉该节点,如果删不掉则会抛出一个错误ClientError
            return 0
        except ClientError as e:#如果节点上有关系，则连关系一起删除
            self.graph.run(sql2)
            return 1
            # raise e

    def delete_movie(self, movie):
        sql0 = "match (p:movie {中文名: '%s'}) delete p" % (movie)#删除节点
        sql1 = "match (p:movie {中文名: '%s'})-[r]-(m) delete p,r" % (movie)#删除节点与其上的边

        sql = "match (m:movie {中文名: '%s'}) with m return m" % (movie)
        data = self.graph.run(sql).data()
        if not data:#如果不存在该节点
            return -1

        try:
            self.graph.run(sql0)#尝试删掉该节点
            return 0
        except ClientError as e:#如果节点上有关系，则连关系一起删除
            self.graph.run(sql1)
            return 1

    def update_actor(self, info):
        actor = info['actor']
        sql = "match (p:actor {中文名: '%s'}) with p return p" % (actor)
        data = self.graph.run(sql).data()
        if not data:#如果不存在该节点
            return "the actor does not exist, please add him first"

        s = ""#构造更新语句
        if info['key1']:
            s += " set a.%s = '%s'" % (info['key1'], info['value1'])
        if info['key2']:
            s += " set a.%s = '%s'" % (info['key2'], info['value2'])
        if info['key3']:
            s += " set a.%s = '%s'" % (info['key3'], info['value3'])
        if info['key4']:
            s += " set a.%s = '%s'" % (info['key4'], info['value4'])
        if info['key5']:
            s += " set a.%s = '%s'" % (info['key5'], info['value5'])

        sql = "match (a:actor {中文名: '%s'})  %s return a" % (actor, s)
        data = self.graph.run(sql).data()
        return "updated the actor's infomation"

    def update_movie(self, info):
        movie = info['movie']
        sql = "match (m:movie {中文名:'%s'}) return m" % (movie)
        data = self.graph.run(sql).data()
        if not data:#如果不存在该电影，则首先要添加它
            return "the movie does not exist, please add it first"

        s = ""#构造更新语句
        if info['key1']:
            s += " set a.%s = '%s'" % (info['key1'], info['value1'])
        if info['key2']:
            s += " set a.%s = '%s'" % (info['key2'], info['value2'])
        if info['key3']:
            s += " set a.%s = '%s'" % (info['key3'], info['value3'])
        if info['key4']:
            s += " set a.%s = '%s'" % (info['key4'], info['value4'])
        if info['key5']:
            s += " set a.%s = '%s'" % (info['key5'], info['value5'])

        sql = "match (a:movie {中文名: '%s'})  %s return a" % (movie, s)
        data = self.graph.run(sql).data()
        return "updated the movie's infomation"

    def search_actor(self, actor):
        sql = "match (a:actor {中文名:'%s'}) return a" % (actor)
        data = self.graph.run(sql).data()#返回的结果是Unicode编码的
        if data:
            actor_dict = {}
            data = data[0]['a']#这是一个Node对象，可以像字典一样取值与遍历
            for k, v in data.items():
                actor_dict[k] = v
            return actor_dict#返回一个字典
        else:
            return None

    def search_movie(self, movie):
        sql = "match (m:movie {中文名:'%s'}) return m" % (movie)
        data = self.graph.run(sql).data()#返回的结果是Unicode编码的
        if data:
            movie_dict = {}
            data = data[0]['m']
            for k, v in data.items():
                movie_dict[k] = v
            return movie_dict#返回一个字典
        else:
            return None

    def add_actors_from_csv(self, file):
        # sql = "USING PERIODIC COMMIT 50\
        #     LOAD CSV FROM '%s' AS line\
        #     create (a:actors{actorId:line[0],中文名:line[1],职业:line[2],出生日期:line[3],\
        #     外文名:line[4],国籍:line[5],出生地:line[6],代表作品:line[7],星座:line[8],主要成就:line[9],\
        #     身高:line[10],别名:line[11],毕业院校:line[12],血型:line[13],民族:line[14],体重:line[15],\
        #     经纪公司:line[16],逝世日期:line[17],性别:line[18],妻子:line[19],信仰:line[20]})" % (file)
        sql = "USING PERIODIC COMMIT 50\
            LOAD CSV WITH HEADERS FROM '%s' AS line\
            merge (a:actor{actorId:line.personId,中文名:line.中文名,职业:line.职业,出生日期:line.出生日期,\
            外文名:line.外文名,国籍:line.国籍,出生地:line.出生地,代表作品:line.代表作品,星座:line.星座,主要成就:line.主要成就,\
            身高:line.身高})" % (file)#使用merge是为了防止重复导入
        data = self.graph.run(sql).data()
        return data

    def add_movies_from_csv(self, file):
        sql = "USING PERIODIC COMMIT 50\
            LOAD CSV WITH HEADERS FROM '%s' AS line\
            merge (m:movie{movieId:line.movieId,图片:line.图片,中文名:line.中文名,主演:line.主演,\
            上映时间:line.上映时间,得分:line.得分})" % (file)
        data = self.graph.run(sql).data()
        return data

    def add_relations_from_csv(self, file):
        sql = "USING PERIODIC COMMIT 10\
            LOAD CSV FROM '%s' AS line\
            MATCH (from:actor{actorId:line[0]}),(to:movie{movieId:line[1]})\
            merge (from)-[r:ACT_IN]->(to)" % (file)#导入关系一般使用merge，防止重复导入
        data = self.graph.run(sql).data()
        return data
 
    def search_allMoviesSomeoneAct(self, actor):
        sql = "match (a:actor{中文名: '%s'})-[]->(m:movie) return a.中文名, m.中文名" % (actor)
        result = self.graph.run(sql).data()#如果有值，返回的是字典，格式如下
             # [{'a.中文名': '张国荣', 'm.中文名': '英雄本色'},
             # {'a.中文名': '张国荣', 'm.中文名': '阿飞正传'},
             # {'a.中文名': '张国荣', 'm.中文名': '倩女幽魂'},
             # {'a.中文名': '张国荣', 'm.中文名': '东邪西毒'},
             # {'a.中文名': '张国荣', 'm.中文名': '射雕英雄传之东成西就'},
             # {'a.中文名': '张国荣', 'm.中文名': '春光乍泄'},
             # {'a.中文名': '张国荣', 'm.中文名': '霸王别姬'}]
        return result

    def search_allActorsinMovie(self, movie):
        sql = "match (a:actor)-[]->(m:movie{中文名: '%s'}) return  m.中文名,a.中文名" % (movie)
        result = self.graph.run(sql).data()#如果有值，返回的是字典，格式如下
             # [{'m.中文名': '倩女幽魂', 'a.中文名': '午马'},
             # {'m.中文名': '倩女幽魂', 'a.中文名': '张国荣'},
             # {'m.中文名': '倩女幽魂', 'a.中文名': '王祖贤'}]
        return result

    def search_friends(self, actor):
        sql = "match (a:actor{中文名:'%s'})-[]->(m:movie)<-[]-(f:actor) return a.中文名, m.中文名,f.中文名"%(actor)
        result = self.graph.run(sql).data()
        return result

    def search_actorbyguoji(self, guoji):
        sql = "match (a:actor{国籍: '%s'}) return a.中文名, a.国籍" % (guoji)
        result = self.graph.run(sql).data()
        return result

    def search_actorbyxingzuo(self, xingzuo):
        sql = "match (a:actor{星座: '%s'}) return a.中文名, a.星座" % (xingzuo)
        result = self.graph.run(sql).data()
        return result

    def search_actorbyachievement(self, achievement):
        sql = "match (a:actor) where a.主要成就 contains '%s' return a.中文名, a.主要成就" % (achievement)
        result = self.graph.run(sql).data()
        return result

    def search_moviebyscore(self, low, high):
        sql = "match (m:movie) return m.中文名, m.得分"#因为cypher处理数据的能力不如python，所以筛选交给views
        result = self.graph.run(sql).data()
        return result

