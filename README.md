# 综合实验介绍
## 一、关于设计
### 1.1 知识图谱
#### 1.1.1 知识的获取

	知识的获取主要是两个来源，包括猫眼评分最高的100部中外电影，通过爬虫可以得到电影的格式化信息，其中包括了该部电影的中文名，得分，上映日期，主要演员，等。通过对主要演员的解析，可以得到该电影的主要演员，然后再去百度百科将获取到的演员的信息爬取下来，抽取其中共现频率最高的10个属性，处理为格式化数据。然后通过分析，建立电影与演员的关系，将它们导入到图数据库Neo4j中，形成一个小规模的知识图谱（大约400个节点，400条边）。

#### 1.1.2 知识的增删改查

	通过第一步的操作，我们已经得到了一个基础的知识图谱，但是这是不够的，因为知识图谱是需要动态更新的。所以我又添加了增删该查的功能，最主要的是查找。其中增的功能，可以增加一个演员，一个电影，或者一个关系；删的功能，可以删除一个演员或者一个电影，因为关系是依附于节点的，所以关系不能够被单独删去；可以更新电影或者演员的属性信息；查找的功能最丰富，并且是通过图的形式展示的，我们提供了通过演员的名字查找演员信息，通过电影名查找电影，按评分查找电影，查找某演员演的所有（仅指本知识图谱中的所有，下同）电影，某电影的所有主演，某演员的所有合作者（同演过一部电影），按星座、国籍、成就查找演员。

### 1.2 深度学习
#### 1.2.1 手写数字识别

	在手写数字识别模块，本系统提供了四个模型，分别是逻辑回归，多层感知机，卷积神经网络和循环神经网络（这个虽然不常见，但是也可以提供很好的预测效果）。这一部分有三个功能，一是学习功能，每一个模型都有图文并茂的原理介绍，帮助快速了解模型的原理；二是实践功能，提供了模型的源码，使得你在阅读了模型原理后，可以直接上手照着敲代码，进行实战的训练，加深记忆与理解，并通过实践提升兴趣；三是预测功能，在这个部分，你可以直接上传一张手写数字，我的模型会告诉你你刚才上传的图片是几，并给出预测概率。

#### 1.2.2 神经网络对抗图片生成

	在这一个模块中，同样有三个功能，一个是对抗图片生成的原理，二是对抗图片的生成，因为生成一个图片需要耗费数个小时，因此不建议使用，不过我已经为你生成了十几张对抗图片，你可以拿来直接使用，三是预测功能，你可以上传一张图片，模型会告诉你它的类别。

#### 1.2.3 waitting for you

	也许你已经发现了，以上功能都是涉及的图片，缺失了另一个重要的领域，自然语言处理。这个以后会慢慢添加上去，而且假如你有兴趣的话，也可以上传相关代码给我的这个小项目，大家一起帮助改进。

## 二、关于使用
### 2.1 底层应用的安装
#### 2.1.1 python

	本项目使用的python语言，因此你首先需要安装python，版本为3.6.8，需要的库有：
	
	Package              Version   

	-------------------- -----------   

	absl-py              0.7.1   

	astor                0.8.0  

	attrs                19.1.0   

	backcall             0.1.0  

	beautifulsoup4       4.7.1  

	bleach               3.1.0  

	bosonnlp             0.11.1  

	boto                 2.49.0  

	boto3                1.9.159  

	botocore             1.12.159  

	certifi              2019.6.16  

	Click                7.0  

	colorama             0.4.1  

	cssselect            1.0.3  

	cycler               0.10.0   

	Cython               0.29.3   

	DateTime             4.3   

	decorator            4.4.0   

	defusedxml           0.6.0   

	Django               2.0   

	docopt               0.6.2   

	docutils             0.14   

	entrypoints          0.3   

	funcsigs             1.0.2   

	future               0.17.1   

	gast                 0.2.2   

	gensim               3.7.0   

	graphviz             0.10.1   

	grpcio               1.21.1   

	h5py                 2.9.0    

	ipykernel            5.1.1    

	ipython              7.3.0   

	ipython-genutils     0.2.0   

	ipywidgets           7.4.2    

	itchat               1.2.32    

	jedi                 0.13.3    

	jieba                0.39    

	Jinja2               2.10.1    

	jmespath             0.9.4    

	jsonschema           3.0.1    

	jupyter              1.0.0    

	jupyter-client       5.2.4    

	jupyter-console      6.0.0   

	jupyter-core         4.4.0   

	jupyterthemes        0.20.0   

	Keras                2.2.4   

	Keras-Applications   1.0.8   

	Keras-Preprocessing  1.1.0    

	kiwisolver           1.1.0   

	lesscpy              0.13.0    

	lxml                 4.3.0   

	Markdown             3.0.1   

	MarkupSafe           1.1.1   

	matplotlib           2.2.3    

	mistune              0.8.4    

	mock                 3.0.5   

	mysqlclient          1.4.2.post1   

	nbconvert            5.5.0   

	nbformat             4.4.0   

	neobolt              1.7.13    

	neotime              1.7.4    

	networkx             2.3    

	nltk                 3.4    

	notebook             5.7.5    

	numpy                1.16.4     

	opencv-python        4.1.0.25    

	paddlepaddle         1.3.1    

	pandas               0.23.4     

	pandocfilters        1.4.2     

	parso                0.4.0    

	pickleshare          0.7.5    

	Pillow               5.4.1    

	pip                  19.1.1    

	ply                  3.11   

	pqi                  2.0.6   

	prometheus-client    0.6.0   

	prompt-toolkit       2.0.9   

	protobuf             3.8.0    

	py2neo               4.3.0    

	Pygments             2.3.1     

	PyMySQL              0.9.3    

	pyparsing            2.4.0     

	pypng                0.0.19    

	PyQRCode             1.2.1     

	pyquery              1.4.0     

	pyrsistent           0.15.2    

	python-dateutil      2.7.5     

	pytz                 2018.9     

	pywinpty             0.5.5    

	PyYAML               5.1     

	pyzmq                18.0.1     

	qtconsole            4.5.1     

	rarfile              3.0     

	recordio             0.1.7     

	requests             2.9.2     

	s3transfer           0.2.0     

	scikit-learn         0.20.2     

	scipy                1.2.0     

	seaborn              0.9.0     

	Send2Trash           1.5.0      

	setuptools           40.6.2     

	singledispatch       3.4.0.3     

	six                  1.12.0    

	smart-open           1.8.4    

	soupsieve            1.9.1    

	tensorboard          1.13.1    

	tensorflow           1.13.1    

	tensorflow-estimator 1.13.0    

	termcolor            1.1.0    

	terminado            0.8.2    

	testpath             0.4.2     

	tflearn              0.3.2     

	tornado              6.0.2    

	traitlets            4.3.2        

	urllib3              1.24.3     

	wcwidth              0.1.7    

	webencodings         0.5.1    

	Werkzeug             0.15.4    

	wheel                0.33.4    

	widgetsnbextension   3.4.2    

	wxpy                 0.3.9.8    

	zope.interface       4.6.0    

	一个一个安装太麻烦，你可以新建一个requirement.txt文件，将上面的信息复制到文件中，然后运行命令pip install -r requirement.txt进行批量安装，如果你嫌速度慢，还可以为pip 换源，这里就不教你怎么换了，你只需要知道还有换源这么个操作就行了，我告诉了你这里有条路（我认为这是比较重要的，因为我自己就是从来没人告诉过我还有这么一条路可以走，是我无意间发现的），至于怎么走网上有很多教程。

	然后，你需要安装一个python的编辑器，尽管python自带了shell，但是你知道的，那个非常难用。因此我建议你至少安装以下编辑器中的一个：Sublime Text,PyCharm,Note Pad++,Jupyter NodeBook。我没用过Note Pad++,所以不做评价，对于剩下的三种，我都安装了，平时用的比较多的是Jupyter和Sublime，写单个小程序的时候用的多是jupyter，因为便于调试，写一行代码就可以查看一行的结果，强烈推荐安装一个（事实上，如果你安装了上面我列出的所有的库，你就已经将它安装上了），并且这个是客户-服务器模式的，不用额外安装什么软件，可以直接在浏览器中使用。然后就是写工程项目，你可以选择sublime或者pycharm，我写这个项目使用的是sublime，挺好用的，pycharm功能更强大，但是随之而来的是非常占用内存，因为它太大了，我每次打开它都要等上半分钟，尽管我的电脑配置算不错了。还有就是，pycharm是分企业版和社区版的，企业版需要收费，社区版是免费的，功能差不太多。

#### 2.1.2 Neo4j

	本项目使用的数据库为图数据库Neo4j，它也有企业版和社区版之分，企业版收费，社区版缺少一些功能，但免费，对于学生来说够用了。

	Neo4j是基于java开发的，所以首先需要安装一个java，并且需要是java1.8.xx版本。Linux系统中已经自带了，你可以直接运行命令java -version查看你的java版本是否符合要求，不符合要求的要进行更新或者重装。

	然后安装neo4j：首先去[官网](https://neo4j.com/download-thanks/?edition=community&release=3.3.9&flavour=unix#)下载社区版的neo4j，它分为windows,mac,linux版本，下载自己需要的（后面的讲解以linux为例）。然后解压tar -xf neo4j-community-3.3.9-unix.tar.gz，首先找到conf文件夹，打开neo4j.conf文件，找到dbms.directories.import=import将其注释掉，然后找到dbms.connectors.default_listen_address=0.0.0.0 将其取消注释并改为你服务器的地址，然后找到#dbms.connector.bolt.listen_address=:7687 与#dbms.connector.http.listen_address=:7474 ，取消注释。然后退回父目录，你将看到bin文件夹，不要进入（再说一遍不要进入），直接在命令行中输入bin/neo4j console初始化服务，按照提示的网址打开浏览器，使用http协议登陆，端口号为7474，设置用户名为neo4j，密码为123456。以后在运行本项目的时候要输入bin/neo4j strat ，关闭服务使用bin/neo4j stop, 查看状态使用bin/neo4j status， 重启使用bin/neo4j restart.

#### 2.1.3 keras

	linux下的keras有一个bug，那就是它找不到imagenet_class_index.json ，因此在使用imagenet 训练出来的模型预测图片时该库会报错，无法预测，这个问题我在windows里面是没有发现的，当时写老师布置的作业时我就只能使用服务训练图片但却不能使用它来进行预测，当时也没找到解决办法，但是现在不行了，这个问题不解决这个项目就不能布置到服务器上。于是我花了两天时间慢慢地在库里面查看源代码，后来找到了解决办法。首先你需要去网上把imagenet_class_index.json 下载下来（本项目已经给你下载好了），然后将它放到 ~/.keras/models目录下即可。
	
### 2.2 使用本项目

	当上面的软件都正确安装完成后，进入项目目录中，首先你需要修改几个配置项。首先进入MyDjangoProject/MyDjangoProject 中，打开settings.py文件，找到ALLOWED_HOSTS = []，将你的服务器地址添加进去，或者你可以直接添加"*" ，这样所有的地址都可以访问不过不建议这么做。然后打开views.py文件，修改host = '127.0.0.1'，port = '127.0.0.1:8000' 为*自己的服务器地址*。
	
	然后回到上级目录，你可以看到manage.py文件，运行命令python manage.py runserver '你的服务器地址:8000' ，然后在浏览器中输入网站主页地址：你的服务器地址:8000/FTQ/，你就可以看到我的网站了。然后你就可以使用了。
	
## 三、效果展示

	见 网站展示.docx
	
## 四、源代码

>MyDjangoProject
>>db.sqlite3
>>manage.py
>>README.md
>>网站展示.docx
>>MyDjangoProject
>>MyKG
>>static
>>Templates

	第一层包括MyDjangoProject、KG、static、Templates目录和manage.py、README.md、网站展示.docx 等文件。

### 4.1 MyDjangoProject
>MyDjangoProject
>>deepviews.py 深度学习的视图函数文件
>>neo_db.py 知识图谱的数据库操作函数文件
>>prediction_models.py 深度学习预测函数文件
>>settings.py 网站配置文件
>>spider.py  网络爬虫文件
>>urls.py  路由文件
>>views.py 知识图谱视图文件
>>wsgi.py
>>__init__.py
>>__pycache__
### 4.2 MyKG

	没有使用
	
### 4.3 static
>static
>>files 这是爬虫生成的导入数据库的标准文件，删除也没关系还会自动生成
>>>actors.csv 
>>>actors.txt
>>>movies.csv
>>>movies.json
>>>relations.csv
>>img 这是网页的背景图和效果图
>>>03.jpg
>>>12.jpg
>>>backgroundimg00.jpg
>>>backgroundimg02.jpg
>>>con_brief01.png
>>>con_brief02.png
>>js 这是js文件
>>>echarts.min.js
>>>jquery-1.12.2.js
>>pictures 这是网页上传的预测图片
>>>0.png
>>>001.jpg
>>>003.jpg
>>>004.jpg
>>>010.jpg
>>>1.png
>>>hack_car.png
>>>hack_car1.png
>>>raw_car1_gaitubao_299x299.png
>>RNN_mnist_files 这是网页里面的图片
>>>724315-20170601201859321-1107873695.png
>>>724315-20170601203508868-924538983.png
>>trained_models 这是训练好的模型数据文件
>>>inception_v3_on_imageNet.h5
>>>cnn_model
>>>>checkpoint
>>>>cnn_model.data-00000-of-00001
>>>>cnn_model.index
>>>>cnn_model.meta
>>>logistic_model
>>>>checkpoint
>>>>logistic_model.data-00000-of-00001
>>>>logistic_model.index
>>>>logistic_model.meta
>>>mlp_model
>>>>checkpoint
>>>>mlp_model.data-00000-of-00001
>>>>mlp_model.index
>>>>mlp_model.meta
>>>rnn_mnist
>>>>checkpoint
>>>>rnn_mnist.data-00000-of-00001
>>>>rnn_mnist.index
>>>>rnn_mnist.meta
>>一文读懂逻辑回归_files
>>>133003c4fda7293564c58eda27a178997225.gif
>>>2bf62e7fb74a74664fdeaa7c11f73d4d9121.gif
>>>2caf3c84ebb361da259d528394610e732929.gif
>>>516884747928241d8eb7305735df4f207759.gif
>>卷积神经网络_files
>>>1093303-20170430194934006-705271151.jpg
>>>1093303-20170430194958725-2144325242.png
>>>dl_3_12.gif
>>>dl_3_2.png
>>>dl_3_3.png
>>多层感知机原理详解_files
>>>1344061-20180506113004354-1567433739.png
>>>1344061-20180506115221964-706278801.png
>>>1344061-20180506115759574-1741102562.png
>>>1344061-20180506120127501-1511095651.png
>>>1344061-20180506120148259-1417578443.png

### 4.4 Templates
>Templates 这里面都是每个网页的html文件
>>00.html
>>1_index.html
>>addactor.html
>>addmovie.html
>>addrelation.html
>>con-brief.html
>>con_picture.html
>>con_pre.html
>>delactor.html
>>delmovie.html
>>一文读懂逻辑回归.html
>>卷积神经网络.html
>>卷积神经网络源码.html
>>多层感知机原理详解.html
>>情感分析-CNN.html
>>神经网络-感知机-source-code.html

## 五、以下是测试内容
    first test
