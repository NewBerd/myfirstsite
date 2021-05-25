# coding: utf-8
#导入需要的库
import tensorflow as tf
from PIL import Image 
import numpy as np
from keras.preprocessing import image
from keras.applications import inception_v3
from keras.models import load_model
from keras import backend as K

# print('filepath:',__file__)
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

# ****************--以下是文本生成的代码--*************** #

# with open('static/files/anna.txt', 'r') as f:
#     text=f.read()
# vocab = set(text)
# vocab_size = len(vocab)
vocab_size = 83#字符集的大小
# vocab_to_int = {c: i for i, c in enumerate(vocab)}
# int_to_vocab = dict(enumerate(vocab))
# encoded = np.array([vocab_to_int[c] for c in text], dtype=np.int32)

def get_batches(arr, n_seqs, n_steps):
    '''
    对已有的数组进行mini-batch分割
    arr: 待分割的数组
    n_seqs: 一个batch中序列个数（行数，样本数或者batch_size）
    n_steps: 单个序列包含的字符数（列数，样本的大小）
    '''
    batch_size = n_seqs * n_steps
    n_batches = int(len(arr) / batch_size)
    # 这里我们仅保留完整的batch，对于不能整出的部分进行舍弃
    arr = arr[:batch_size * n_batches]
    # 重塑
    arr = arr.reshape((n_seqs, -1))
    for n in range(0, arr.shape[1], n_steps):
        # inputs
        x = arr[:, n:n + n_steps]
        # targets
        y = np.zeros_like(x)
        y[:, :-1], y[:, -1] = x[:, 1:], x[:, 0]
        yield x, y

def build_inputs(num_seqs, num_steps):
    '''
    构建输入层
    num_seqs: 每个batch中的序列个数
    num_steps: 每个序列包含的字符数
    '''
    inputs = tf.placeholder(tf.int32, shape=(num_seqs, num_steps), name='inputs')
    targets = tf.placeholder(tf.int32, shape=(num_seqs, num_steps), name='targets')
    # 加入keep_prob
    keep_prob = tf.placeholder(tf.float32, name='keep_prob')
    return inputs, targets, keep_prob

def build_lstm(lstm_size, num_layers, batch_size, keep_prob):
    '''
    构建lstm层
    keep_prob：保留多少权重被更新
    lstm_size: lstm隐层中结点数目，也即细胞数，units，cells等
    num_layers: lstm的隐层数目
    batch_size: batch_size
    '''
    # 构建一个基本lstm单元
    #     lstm = tf.contrib.rnn.BasicLSTMCell(lstm_size)
    lstm = [tf.contrib.rnn.BasicLSTMCell(lstm_size) for _ in range(num_layers)]
    # 添加dropout
    drop = [tf.contrib.rnn.DropoutWrapper(lstm_, output_keep_prob=keep_prob) for lstm_ in lstm]  # 加入了dropout的基本单元
    # 堆叠
    cell = tf.contrib.rnn.MultiRNNCell(drop)  # 构建多层循环神经网络
    initial_state = cell.zero_state(batch_size, tf.float32)
    return cell, initial_state

def build_output(lstm_output, in_size, out_size):
    '''
    构造输出层
    lstm_output: lstm层的输出结果
    in_size: lstm输出层重塑后的size
    out_size: softmax层的size
    '''
    # 将lstm的输出按照列concate，例如[[1,2,3],[7,8,9]],
    # tf.concat的结果是[1,2,3,7,8,9]
    # 假如有t1.shape(2,3,4),t2.shape(2,3,4),t3.shape(2,3,4),则concat([t1,t2,t3],axis=0)的结果是t.shape(2+2+2,3,4)
    # axis=1,则结果为(2,3+3+3,4),axis=2,则结果为(2,3,12)
    seq_output = tf.concat(lstm_output, axis=1)  # tf.concat(concat_dim, values)
    # reshape
    x = tf.reshape(seq_output, [-1, in_size])
    # 将lstm层与softmax层全连接
    with tf.variable_scope('softmax'):
        softmax_w = tf.Variable(tf.truncated_normal([in_size, out_size], stddev=0.1))
        softmax_b = tf.Variable(tf.zeros(out_size))
    # 计算logits
    logits = tf.matmul(x, softmax_w) + softmax_b
    # softmax层返回概率分布
    out = tf.nn.softmax(logits, name='predictions')
    return out, logits


def build_loss(logits, targets, lstm_size, num_classes):
    '''
    根据logits和targets计算损失
    logits: 全连接层的输出结果（不经过softmax）
    targets: targets
    lstm_size
    num_classes: vocab_size
    '''
    # One-hot编码
    y_one_hot = tf.one_hot(targets, num_classes)
    y_reshaped = tf.reshape(y_one_hot, logits.get_shape())
    # Softmax cross entropy loss
    loss = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y_reshaped)
    loss = tf.reduce_mean(loss)
    return loss

def build_optimizer(loss, learning_rate, grad_clip):
    '''
    构造Optimizer
    loss: 损失
    learning_rate: 学习率
    '''
    # 使用clipping gradients
    # tf.gradients(y,x),求y对x的导数
    #     Gradient Clipping的引入是为了处理gradient explosion或者gradients vanishing的问题。当在一次迭代中权重的更新过于迅猛的话，
    #     很容易导致loss divergence。Gradient Clipping的直观作用就是让权重的更新限制在一个合适的范围。
    #     具体的细节是 ：
    #     １．在solver中先设置一个clip_gradient
    #     ２．在前向传播与反向传播之后，我们会得到每个权重的梯度diff，这时不像通常那样直接使用这些梯度进行权重更新，
    #         而是先求所有权重梯度的平方和sumsq_diff，如果sumsq_diff > clip_gradient，
    #         则求缩放因子scale_factor = clip_gradient / sumsq_diff。这个scale_factor在(0,1)之间。
    #         如果权重梯度的平方和sumsq_diff越大，那缩放因子将越小。
    #     ３．最后将所有的权重梯度乘以这个缩放因子，这时得到的梯度才是最后的梯度信息。
    #     这样就保证了在一次迭代更新中，所有权重的梯度的平方和在一个设定范围以内，这个范围就是clip_gradient.
    tvars = tf.trainable_variables()  # 获取所有的可训练变量
    grads, _ = tf.clip_by_global_norm(tf.gradients(loss, tvars), grad_clip)  # 1.求损失函数对可训练变量（网络权重）的梯度，并防止
    # 梯度爆炸
    train_op = tf.train.AdamOptimizer(learning_rate)
    optimizer = train_op.apply_gradients(zip(grads, tvars))  # 2.将前面计算出的梯度用在权重上进行更新
    # 平时使用的minimize()实际上包含了这两步
    return optimizer

class CharRNN:
    def __init__(self, num_classes, batch_size=64, num_steps=50,
                 lstm_size=128, num_layers=2, learning_rate=0.001,
                 grad_clip=5, sampling=False):
        # 如果sampling是True，则采用SGD
        if sampling == True:
            batch_size, num_steps = 1, 1
        else:
            batch_size, num_steps = batch_size, num_steps  # batch_size行, num_steps列
        tf.reset_default_graph()
        # 输入层
        self.inputs, self.targets, self.keep_prob = build_inputs(batch_size, num_steps)
        # LSTM层
        cell, self.initial_state = build_lstm(lstm_size, num_layers, batch_size, self.keep_prob)
        # 对输入进行one-hot编码
        x_one_hot = tf.one_hot(self.inputs, num_classes)
        # 运行RNN
        outputs, state = tf.nn.dynamic_rnn(cell, x_one_hot, initial_state=self.initial_state)
        self.final_state = state
        # 预测结果
        self.prediction, self.logits = build_output(outputs, lstm_size, num_classes)
        # Loss 和 optimizer (with gradient clipping)
        self.loss = build_loss(self.logits, self.targets, lstm_size, num_classes)
        self.optimizer = build_optimizer(self.loss, learning_rate, grad_clip)

def pick_top_n(preds, vocab_size, top_n=5):
    """
    从预测结果中选取前top_n个最可能的字符
    preds: 预测结果
    vocab_size：词汇表大小,其实也是字符表的序号
    top_n
    """
    p = np.squeeze(preds)
    # 将除了top_n个预测值的位置都置为0
    #np.argsort(p)求数组p的从小到大排列后的索引序列
    p[np.argsort(p)[:-top_n]] = 0
    # 归一化概率
    p = p / np.sum(p)
    # 随机选取一个字符，vocab_size是词汇表，1表示取一个，p是取得概率，默认为均匀分布
    c = np.random.choice(vocab_size, 1, p=p)[0]
    return c


def sample(checkpoint, n_samples, lstm_size, vocab_to_int, int_to_vocab, prime="The "):
    """
    生成新文本
    checkpoint: 检查点文件
    n_sample: 新文本的字符长度
    lstm_size: 隐层结点数
    vocab_size：字符表大小
    prime: 起始文本
    """
    # with open('static/files/anna.txt', 'r') as f:
    #     text = f.read()
    # vocab = set(text)
    # vocab_to_int = {c: i for i, c in enumerate(vocab)}
    # int_to_vocab = dict(enumerate(vocab))

    # 将输入的单词转换为单个字符组成的list
    samples = [c for c in prime]
    # sampling=True意味着batch的size=1 x 1
    model = CharRNN (vocab_size, lstm_size=lstm_size, sampling=True)
    saver = tf.train.Saver()
    with tf.Session() as sess:
        # 加载模型参数，恢复训练
        saver.restore(sess, checkpoint)
        new_state = sess.run(model.initial_state)
        for c in prime:
            x = np.zeros((1, 1))
            # 输入单个字符
            x[0, 0] = vocab_to_int[c]
            feed = {model.inputs: x,
                    model.keep_prob: 1.,  # 注意预测的时候一定要是1
                    model.initial_state: new_state}
            preds, new_state = sess.run([model.prediction, model.final_state],
                                        feed_dict=feed)
            c = pick_top_n(preds, vocab_size, 3)  # 此处是一个整数
            # 添加字符到samples中，先由映射表转为字符
            samples.append(int_to_vocab[c])  #

        # 不断生成字符，直到达到指定数目
        for i in range(n_samples):
            x[0, 0] = c
            feed = {model.inputs: x,
                    model.keep_prob: 1.,
                    model.initial_state: new_state}
            preds, new_state = sess.run([model.prediction, model.final_state], feed_dict=feed)
            c = pick_top_n(preds, vocab_size, 3)
            samples.append(int_to_vocab[c])
    return ''.join(samples)

def text_generation(text_size, prime):
    """
    生成指定长度的文本
    :param text_size: 文本长度
    :param prime: 文本开头
    :return: 生成的文本
    """
    lstm_size = 512  # Size of hidden layers in LSTMs
    # vocab_size = 83#字符集的大小
    # vocab_to_int = {'5': 0, '.': 1, 'c': 2, 'N': 3, '7': 4, 'U': 5, 'J': 6, 'v': 7, 'B': 8, '/': 9, 't': 10, 'a': 11, 'o': 12, '-': 13, 'C': 14, '"': 15, '8': 16, 'u': 17, 'b': 18, 'A': 19, 'm': 20, 'S': 21, '?': 22, '1': 23, '3': 24, 'T': 25, '4': 26, 'X': 27, '&': 28, 'I': 29, 'g': 30, 'n': 31, 'p': 32, 'l': 33, ':': 34, '\n': 35, 'k': 36, 's': 37, 'E': 38, "'": 39, 'e': 40, 'P': 41, 'W': 42, 'f': 43, 'j': 44, '(': 45, '0': 46, '%': 47, '@': 48, '$': 49, 'K': 50, '9': 51, 'i': 52, 'd': 53, 'F': 54, '!': 55, 'q': 56, 'D': 57, '6': 58, 'w': 59, 'V': 60, '*': 61, 'Y': 62, '2': 63, 'z': 64, 'Z': 65, 'y': 66, ';': 67, 'M': 68, 'L': 69, 'R': 70, 'H': 71, ' ': 72, 'G': 73, ')': 74, '`': 75, 'r': 76, 'h': 77, ',': 78, 'x': 79, 'O': 80, 'Q': 81, '_': 82}
    # int_to_vocab = {0: '5', 1: '.', 2: 'c', 3: 'N', 4: '7', 5: 'U', 6: 'J', 7: 'v', 8: 'B', 9: '/', 10: 't', 11: 'a', 12: 'o', 13: '-', 14: 'C', 15: '"', 16: '8', 17: 'u', 18: 'b', 19: 'A', 20: 'm', 21: 'S', 22: '?', 23: '1', 24: '3', 25: 'T', 26: '4', 27: 'X', 28: '&', 29: 'I', 30: 'g', 31: 'n', 32: 'p', 33: 'l', 34: ':', 35: '\n', 36: 'k', 37: 's', 38: 'E', 39: "'", 40: 'e', 41: 'P', 42: 'W', 43: 'f', 44: 'j', 45: '(', 46: '0', 47: '%', 48: '@', 49: '$', 50: 'K', 51: '9', 52: 'i', 53: 'd', 54: 'F', 55: '!', 56: 'q', 57: 'D', 58: '6', 59: 'w', 60: 'V', 61: '*', 62: 'Y', 63: '2', 64: 'z', 65: 'Z', 66: 'y', 67: ';', 68: 'M', 69: 'L', 70: 'R', 71: 'H', 72: ' ', 73: 'G', 74: ')', 75: '`', 76: 'r', 77: 'h', 78: ',', 79: 'x', 80: 'O', 81: 'Q', 82: '_'}
    with open("static/trained_models/charsetforanna", "r") as f:
        vocab = eval(f.readline())
        vocab_size = len(vocab)
        vocab_to_int = eval(f.readline())
        int_to_vocab = eval(f.readline())
    checkpoint = "static/trained_models/text_gen/lstm_gene_text.ckpt"
    samp = sample(checkpoint, text_size, lstm_size, vocab_to_int, int_to_vocab, prime)
    return samp