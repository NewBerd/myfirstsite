from django.shortcuts import render
from django.http import HttpResponse
import os
import json
from . import prediction_models

print('deepviewspath:', __file__)
pic_path = os.path.abspath(__file__)
pic_path = os.path.dirname(pic_path)
pic_path = os.path.dirname(pic_path)
pic_path = os.path.join(pic_path, 'static')
pic_path = os.path.join(pic_path, 'pictures')#之所以写这么麻烦是为了可移植性，尽量避免绝对路径出现。
print('picturespath:',pic_path)

def show_logistic(request):
    return render(request, "一文读懂逻辑回归.html")

def logistic_source_code(request):
    return render(request, "logistic_source_code.html")

def mlp_brief_intro(request):
    return render(request, "多层感知机原理详解.html")

def mlp_source_code(request):
    return render(request, "神经网络-感知机-source-code.html")

def cnn_brief_intro(request):
    return render(request, "卷积神经网络.html")

def rnn_brief_intro(request):
    return render(request, "RNN_mnist.html")

def rnn_source_code(request):
    return render(request, "RNN_mnist_source_code.html")

def cnn_source_code(request):
    return render(request, "卷积神经网络源码.html")

def uploadpicture(request):
    return render(request, "uploadpicture.html")
    # return HttpResponse("test")
def uploadpicture1(request):
    return render(request, "uploadpicture1.html")

def log_pre(request):
    if request.method == "POST":    # 请求方法为POST时，进行处理
        myFile =request.FILES.get("myfile", None)    # 获取上传的文件，如果没有文件，则默认为None
        if not myFile:
            returnHttpResponse("no files for upload!")
        global pic_path
        destination = open(os.path.join(pic_path, myFile.name), 'wb+')    # 打开特定的文件进行二进制的写操作
        for chunk in myFile.chunks():      # 分块写入文件
            destination.write(chunk)
        destination.close()

    pro, y = prediction_models.logistic_prediction("static/pictures/" + myFile.name)
    argument = {}
    argument["picture"] = "static/pictures/" + myFile.name
    argument["zero"] = pro[0]
    argument["one"] = pro[1]
    argument["two"] = pro[2]
    argument["three"] = pro[3]
    argument["four"] = pro[4]
    argument["five"] = pro[5]
    argument["six"] = pro[6]
    argument["seven"] = pro[7]
    argument["eight"] = pro[8]
    argument["nine"] = pro[9]
    # argument["pro"] = pro
    argument["pre"] = y
    return render(request, "prediction.html", argument)


def mlp_pre(request):
    if request.method == "POST":    # 请求方法为POST时，进行处理
        myFile =request.FILES.get("myfile", None)    # 获取上传的文件，如果没有文件，则默认为None
        if not myFile:
            returnHttpResponse("no files for upload!")
        global pic_path
        destination = open(os.path.join(pic_path, myFile.name), 'wb+')    # 打开特定的文件进行二进制的写操作
        for chunk in myFile.chunks():      # 分块写入文件
            destination.write(chunk)
        destination.close()

    pro, y = prediction_models.mlp_prediction("static/pictures/" + myFile.name)#有一点需要注意的是，在后台的路径和
    #前端的有一点点区别，即前端的路径需要在static前面加上/
    argument = {}
    argument["picture"] = "static/pictures/" + myFile.name
    argument["zero"] = pro[0]#因为参数不能使用数字作为键，字符数字也不可以，所以这几句话有点啰嗦。
    argument["one"] = pro[1]
    argument["two"] = pro[2]
    argument["three"] = pro[3]
    argument["four"] = pro[4]
    argument["five"] = pro[5]
    argument["six"] = pro[6]
    argument["seven"] = pro[7]
    argument["eight"] = pro[8]
    argument["nine"] = pro[9]
    # argument["pro"] = pro
    argument["pre"] = y
    return render(request, "prediction.html", argument)

def cnn_pre(request):
    if request.method == "POST":    # 请求方法为POST时，进行处理
        myFile =request.FILES.get("myfile", None)    # 获取上传的文件，如果没有文件，则默认为None
        if not myFile:
            returnHttpResponse("no files for upload!")
        global pic_path
        destination = open(os.path.join(pic_path, myFile.name), 'wb+')    # 打开特定的文件进行二进制的写操作
        # destination = open(os.path.join("D:/JupyterNotebook/小型知识图谱构建/MyDjangoProject/static/pictures",myFile.name),'wb+')    # 打开特定的文件进行二进制的写操作
        for chunk in myFile.chunks():      # 分块写入文件
            destination.write(chunk)
        destination.close()
    pro, y = prediction_models.cnn_prediction("static/pictures/" + myFile.name)
    argument = {}
    argument["picture"] = "static/pictures/" + myFile.name
    argument["zero"] = pro[0]#因为参数不能使用数字作为键，字符数字也不可以，所以这几句话有点啰嗦。
    argument["one"] = pro[1]
    argument["two"] = pro[2]
    argument["three"] = pro[3]
    argument["four"] = pro[4]
    argument["five"] = pro[5]
    argument["six"] = pro[6]
    argument["seven"] = pro[7]
    argument["eight"] = pro[8]
    argument["nine"] = pro[9]
    # argument["pro"] = pro
    argument["pre"] = y
    return render(request, "prediction.html", argument)

def rnn_pre(request):
    if request.method == "POST":    # 请求方法为POST时，进行处理
        myFile =request.FILES.get("myfile", None)    # 获取上传的文件，如果没有文件，则默认为None
        if not myFile:
            returnHttpResponse("no files for upload!")
        global pic_path
        destination = open(os.path.join(pic_path, myFile.name), 'wb+')    # 打开特定的文件进行二进制的写操作
        # destination = open(os.path.join("D:/JupyterNotebook/小型知识图谱构建/MyDjangoProject/static/pictures",myFile.name),'wb+')    # 打开特定的文件进行二进制的写操作
        for chunk in myFile.chunks():      # 分块写入文件
            destination.write(chunk)
        destination.close()
    pro, y = prediction_models.rnn_prediction("static/pictures/" + myFile.name)
    argument = {}
    argument["picture"] = "static/pictures/" + myFile.name
    argument["zero"] = pro[0]#因为参数不能使用数字作为键，字符数字也不可以，所以这几句话有点啰嗦。
    argument["one"] = pro[1]
    argument["two"] = pro[2]
    argument["three"] = pro[3]
    argument["four"] = pro[4]
    argument["five"] = pro[5]
    argument["six"] = pro[6]
    argument["seven"] = pro[7]
    argument["eight"] = pro[8]
    argument["nine"] = pro[9]
    # argument["pro"] = pro
    argument["pre"] = y
    return render(request, "prediction.html", argument)

def con_brief(request):
    return render(request, "con-brief.html")

def con_pre(request):
    if request.method == "POST":    # 请求方法为POST时，进行处理
        myFile =request.FILES.get("myfile", None)    # 获取上传的文件，如果没有文件，则默认为None
        if not myFile:
            returnHttpResponse("no files for upload!")
        global pic_path
        destination = open(os.path.join(pic_path, myFile.name), 'wb+')    # 打开特定的文件进行二进制的写操作
        # destination = open(os.path.join("D:/JupyterNotebook/小型知识图谱构建/MyDjangoProject/static/pictures",myFile.name),'wb+')    # 打开特定的文件进行二进制的写操作
        for chunk in myFile.chunks():      # 分块写入文件
            destination.write(chunk)
        destination.close()
    html = prediction_models.confront_prediction("static/pictures/" + myFile.name)
    path = "/static/pictures/" + myFile.name
    return render(request, "con_pre.html", {'pic':path, 'text':html})

def con_gen(request):
    if request.method == "POST":    # 请求方法为POST时，进行处理
        myFile =request.FILES.get("myfile", None)    # 获取上传的文件，如果没有文件，则默认为None
        if not myFile:
            returnHttpResponse("no files for upload!")
        global pic_path
        destination = open(os.path.join(pic_path, myFile.name), 'wb+')    # 打开特定的文件进行二进制的写操作
        # destination = open(os.path.join("D:/JupyterNotebook/小型知识图谱构建/MyDjangoProject/static/pictures",myFile.name),'wb+')    # 打开特定的文件进行二进制的写操作
        for chunk in myFile.chunks():      # 分块写入文件
            destination.write(chunk)
        destination.close()
    prediction_models.confront_generate("static/pictures/" + myFile.name)
    path = "/static/pictures/" + myFile.name
    return render(request, "con_picture.html", {'pic':path})

def gen_text_brief(request):
    return render(request, "text_generation_brief.html")

def gen_text_source_code(request):
    return render(request, "文本生成_LSTM.html")

def get_text(request):
    return render(request, "get-text.html")

def generate_text(request):
    request.encoding = "utf-8"
    pre = request.GET["pre"]
    lenth = request.GET["len"]
    lenth = int(lenth)
    # print(type(lenth))
    text = prediction_models.text_generation(lenth, pre)
    # print(text)
    return render(request, "text_show.html", {"text": text, "pre":pre, "len":lenth})
