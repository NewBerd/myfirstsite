# coding: utf-8
#导入需要的库
import tensorflow as tf
from PIL import Image 
import numpy as np
from keras.preprocessing import image
from keras.applications import inception_v3
from keras.models import load_model
from keras import backend as K

print('filepath:',__file__)
#这是一个坑，web与tensorflow理念的冲突导致的。解决办法参考链接
# https://cloud.tencent.com/developer/article/1167171
graph = tf.get_default_graph()
#模型加载
print("开始加载模型")
model = load_model("static/trained_models/inception_v3_on_imageNet.h5")#从本地加载
print("模型加载完毕")


def logistic_prediction(file):
    #将图片转换为模型需要的输入格式
    img = Image.open(file)
    array = np.array(img)
    # array = 255 - array#黑白反转
    array = array / 255.0#归一化
    array = array.reshape(-1, 784)#改造形状
    #加载模型进行预测
    with tf.Session() as sess:
        new_saver = tf.train.import_meta_graph("static/trained_models/logistic_model/logistic_model.meta")#加载图
        new_saver.restore(sess, tf.train.latest_checkpoint("static/trained_models/logistic_model"))#通过检查点文件加载最新的模型
        graph = tf.get_default_graph()
        x = graph.get_tensor_by_name("X:0")#获取输入变量
        y_pre = graph.get_tensor_by_name("wx_b/Softmax:0")#获取输出值
        y_ = tf.argmax(y_pre, axis=1)
        pre, y = sess.run([y_pre, y_], feed_dict={x:array})
    return pre[0], y[0]

def mlp_prediction(file):
    #将图片转换为模型需要的输入格式
    img = Image.open(file)
    array = np.array(img)
    array = array / 255.0#归一化
    array = array.reshape(-1, 784)#改造形状
    #加载模型进行预测
    with tf.Session() as sess:
        new_saver = tf.train.import_meta_graph("static/trained_models/mlp_model/mlp_model.meta")
        new_saver.restore(sess, tf.train.latest_checkpoint("static/trained_models/mlp_model"))
        graph = tf.get_default_graph()
        x = graph.get_tensor_by_name("input_placeholder:0")
        y_pro = graph.get_tensor_by_name("y_pro:0")
        y_pre = graph.get_tensor_by_name("y_pre:0")
        pro, pre = sess.run([y_pro, y_pre], feed_dict={x:array})
    return pro[0], pre[0]

def cnn_prediction(file):
    #将图片转换为模型需要的输入格式
    img = Image.open(file)
    array = np.array(img)
    array = array / 255.0#归一化
    array = array.reshape(-1, 784)#改造形状
    #加载模型进行预测
    with tf.Session() as sess:
        new_saver = tf.train.import_meta_graph("static/trained_models/cnn_model/cnn_model.meta")
        new_saver.restore(sess, tf.train.latest_checkpoint("static/trained_models/cnn_model"))
        graph = tf.get_default_graph()
        x = graph.get_tensor_by_name("input:0")
        dropout = graph.get_tensor_by_name("dropout:0")
        y_pro = graph.get_tensor_by_name("pro:0")
        y_pre = graph.get_tensor_by_name("pre:0")
        pro, pre = sess.run([y_pro, y_pre], feed_dict={x:array, dropout:1.0})
    return pro[0], pre[0]

def rnn_prediction(file):
    #将图片转换为模型需要的输入格式
    img = Image.open(file)
    array = np.array(img)
    array = array / 255.0#归一化
    array = array.reshape(-1, 784)#改造形状
    #加载模型进行预测
    with tf.Session() as sess:
        new_saver = tf.train.import_meta_graph("static/trained_models/rnn_mnist/rnn_mnist.meta")
        new_saver.restore(sess, tf.train.latest_checkpoint("static/trained_models/rnn_mnist"))
        graph = tf.get_default_graph()
        x = graph.get_tensor_by_name("input:0")
        output = graph.get_tensor_by_name("output/BiasAdd:0")
        y_pro = tf.nn.softmax(output, 1)
        y_pre = tf.argmax(y_pro, 1)
        pro, pre = sess.run([y_pro, y_pre], feed_dict={x:array})
    return pro[0], pre[0]

def confront_prediction(file):
    #加载图片并进行处理，使之符合网络的输入格式
    img = image.load_img(file, target_size=(299, 299))#加载猫的照片
    input_image = image.img_to_array(img)#将其转换为数组(299,299,3)
    # 归一化图片到[-1,1.]
    input_image /= 255. #变为0-1
    input_image -= 0.5 #变为-0.5—0.5
    input_image *= 2. #变为-1—1
    # 为图片添加第四维（batch），使之符合模型的输入
    input_image = np.expand_dims(input_image, axis=0)#用于扩展数组的形状，axis=0表示在0的地方扩展形状，将(299,299,3)变为（1，299，299，3）
    
    # model = inception_v3.InceptionV3(weights='imagenet')#从网络加载
    # model = load_model("static/trained_models/inception_v3_on_imageNet.h5")#从本地加载
    #进行预测，并输出预测结果
    global graph
    with graph.as_default():
        predictions = model.predict(input_image)#使用模型进行预测
        predicted_classes = inception_v3.decode_predictions(predictions, top=1)#top=1表示输出概率最高的前1个类别
    # print(predicted_classes)#查看输出的结果
    imagenet_id, name, confidence = predicted_classes[0][0]#取概率最高的预测类别的信息
    # print("This is a {0} with {1:.4}% confidence!".format(name, confidence * 100))
    return "This is a {0}!".format(name)

def confront_generate(file):
    img = image.load_img(file, target_size=(299, 299))  # 加载原图像
    original_image = image.img_to_array(img)  # 将其转变为矩阵
    original_image /= 255.  # 数据归一化到[-1,1.]上
    original_image -= 0.5
    original_image *= 2.
    original_image = np.expand_dims(original_image, axis=0)  # 维度扩充
    hacked_image = np.copy(original_image)  # 使用副本训练

    # 图片改动范围不能太大，不然人看起来就变样了，因为经训练的图片要看起来不失真
    max_change_above = original_image + 0.01  # 图像改变的范围上限
    max_change_below = original_image - 0.01  # 图像改变的范围下界

    global graph
    with graph.as_default():
        #提取模型的输入输出用做新模型的接口
        model_input_layer = model.layers[0].input  # 模型的输入
        model_output_layer = model.layers[-1].output  # 模型的输出

        # Class #859 is "toaster"
        object_type_to_fake = 859  # 原图像将要改变成的类别

        # 设置学习率
        learning_rate = 0.1
        #定义代价
        cost = model_output_layer[0, object_type_to_fake]  # 定义损失为：模型将经过欺骗训练的图片分类为欺骗目标时的概率
        #model_output_layer:<tf.Tensor 'predictions/Softmax:0' shape=(?, 1000) dtype=float32>
        #[0, object_type_to_fake]表示第一个样本预测为object_type_to_fake类时的概率。
        #
        # 求损失对输入的梯度，[0]表示是第一个样本（因为我们只输入了一个样本）
        # K.gradients（y，x）用来求y对x的导数
        gradient = K.gradients(cost, model_input_layer)[0]

        # K.learning_phase()函数，用于指示模型是处于训练模式（1）还是预测模式（0）
        # K.function([input],[output])用于定义一个新的模型，输出为代价和梯度
        grab_cost_and_gradients_from_model = K.function([model_input_layer, K.learning_phase()], [cost, gradient])

        cost = 0.0  # 代价
        count = 0  # 循环次数
        # print('start training...')
        
        while cost < 0.80:  # 只要最终的图片让模型误以为它是目标的概率达到80%，就可以停止训练了，因为我们只取top-1的类别
            count += 1
            # 需要注意的是，网络的一些层的行为在训练模式与测试模式下是不同的，所以必须指定网络的模式，本次虽然是训练，但对于网络本身来说
            # 其各层参数都不会变，我们训练的是图片，而不是网络，只是利用网络做预测，因此其模式应该是预测模式
            cost, gradients = grab_cost_and_gradients_from_model([hacked_image, 0])  # '0'指明该函数运行在预测模式下
            hacked_image += gradients * learning_rate  # 使用梯度对整个图片进行更新，属于全像素攻击

            # np.clip(x,min,max)用于将x进行取值范围的限制，当x<min时，x取值为min，当x>max时，x取max
            hacked_image = np.clip(hacked_image, max_change_below, max_change_above)
            hacked_image = np.clip(hacked_image, -1.0, 1.0)

            if count % 100 == 0:  # 每循环一千次打印一次结果
                print("Model's prediction probability that the hack_image is a toaster is: {:.8}%".format(cost * 100))
        print('training done')
    #保存图片
    img = hacked_image[0]  # 先进行降维（因为输出是四维的），然后恢复成彩色图，最后转换成array保存成图片，和上面的图像处理过程正好是相反的
    img /= 2.
    img += 0.5
    img *= 255.
    im = Image.fromarray(img.astype(np.uint8))#只有变成0-255整型才可保存
    im.save(file)
    print(file+' is saved!\n')
    return None

# confront_generate("../static/pictures/haha.jpg")
