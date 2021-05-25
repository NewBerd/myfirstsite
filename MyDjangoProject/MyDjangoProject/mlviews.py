from django.shortcuts import render
from django.http import HttpResponse
import os
from . import hanyue

import csv
from sklearn import datasets
from sklearn.model_selection import KFold
import matplotlib as plt
# %matplotlib inline 在jupyter 中使用此代码使得在每个cell下面显示画图内容
import numpy as np
import pandas as pd
# from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.model_selection import cross_val_score
# 正则化regulation
import random
import matplotlib.pyplot as plt
from matplotlib.pylab import rcParams
 # 复杂度可变
from sklearn.linear_model import LinearRegression
# 实验二、使用Ridge回归对抗过拟合
from sklearn.linear_model import Ridge
# 使用Lasso回归对抗过拟合
from sklearn.linear_model import Lasso


def show_overfitting(request):
    dic = {}
    
    # 实验一：对比不同模型的cross-validation结果

    # 用load_wine方法导入数据
    wine_data = datasets.load_wine()
    # print(wine_data.feature_names)
    data_input = wine_data.data
    data_output = wine_data.target

    
    rf_class = RandomForestClassifier()
    lr_class = LogisticRegression()
    svm_class = svm.LinearSVC()

    # print(cross_val_score(rf_class, data_input, data_output, scoring='accuracy', cv=4))

    # 1.使用随机森林方法观察准确率
    accuracy_rf = cross_val_score(rf_class, data_input, data_output, scoring='accuracy', cv=10).mean() * 100
    print('Accuracy of Random Forest is:', accuracy_rf)

    # 2.使用支持向量机方法观察准确率
    accuracy_svm = cross_val_score(svm_class, data_input, data_output, scoring='accuracy', cv=10).mean() * 100
    print('Accuracy of SVM is:', accuracy_svm)

    # 3.使用逻辑回归方法观察准确率
    accuracy_lr = cross_val_score(lr_class, data_input, data_output, scoring='accuracy', cv=10).mean() * 100
    print('Accuracy of LogisticRegression is:', accuracy_lr)

    
    rcParams['figure.figsize'] = 12, 10

    x = np.array([1.4 * i * np.pi / 180 for i in range(0, 300, 4)])
    np.random.seed(20)  # 固定每次生成的随机数
    y = np.sin(x) + np.random.normal(0, 0.2, len(x))
    data = pd.DataFrame(np.column_stack([x, y]), columns=['x', 'y'])
    plt.plot(data['x'], data['y'], '.')
    file = "static/img/han01.png"
    dic["pic1"] = "/"+file
    plt.savefig(file)
    # plt.show()

    for i in range(2, 16):  # power of 1 is already there
        colname = 'x_%d' % i  # new var will be x_power
        data[colname] = data['x'] ** i
    # print(data.head())

   
    def linear_regression(data, power, models_to_plot):
        # initialize predictors:
        predictors = ['x']
        if power >= 2:
            predictors.extend(['x_%d' % i for i in range(2, power + 1)])

        # Fit the model
        linreg = LinearRegression(normalize=True)
        linreg.fit(data[predictors], data['y'])
        y_pred = linreg.predict(data[predictors])

        # Check if a plot is to be made for the entered power
        if power in models_to_plot:
            plt.subplot(models_to_plot[power])
            plt.tight_layout()
            plt.plot(data['x'], y_pred)
            plt.plot(data['x'], data['y'], '.')
            plt.title('Plot for power: %d' % power)

        # Return the result in pre-defined_format
        rss = sum((y_pred - data['y']) ** 2)
        ret = [rss]
        ret.extend([linreg.intercept_])
        ret.extend(linreg.coef_)
        return ret

    col = ['rss', 'intercept'] + ['coef_x_%d' % i for i in range(1, 16)]
    ind = ['model_pow_%d' % i for i in range(1, 16)]
    coef_matrix_simple = pd.DataFrame(index=ind, columns=col)
    # 注意上行代码的columns不能携程column单数，画图就无法画出来了
    # 定义作图的位置与模型的复杂度
    models_to_plot = {1: 231, 3: 232, 6: 233, 8: 234, 11: 235, 14: 236}

    # 画出来
    for i in range(1, 16):
        coef_matrix_simple.iloc[i - 1, 0:i + 2] = linear_regression(data, power=i, models_to_plot=models_to_plot)
    file = "static/img/han02.png"
    dic["pic2"] = "/" + file
    plt.savefig(file)
    # plt.show()

    

    # 定义作图的位置与模型的复杂度
    models_to_plot = {1e-15: 231, 1e-10: 232, 1e-4: 233, 1e-3: 234, 1e-2: 235, 5: 236}

    def ridge_regression(data, predictors, alpha, models_to_plot={}):
        # Fit the model:1.初始化模型配置  2.模型拟合  3.模型预测
        ridgereg = Ridge(alpha=alpha, normalize=True)
        ridgereg.fit(data[predictors], data['y'])
        # predictors的内容实际是data(定义的一种DataFrame数据结构)的某列名称
        y_pred = ridgereg.predict(data[predictors])

        # Check if a plot is to be made for the entered alpha
        if alpha in models_to_plot:
            plt.subplot(models_to_plot[alpha])
            plt.tight_layout()
            plt.plot(data['x'], y_pred)  # 画出拟合曲线图
            plt.plot(data['x'], data['y'], '.')  # 画出样本的散点图
            plt.title('Plot for alpha: %.3g' % alpha)

        # Return the result in pre-defined format
        rss = sum((y_pred - data['y']) ** 2)
        ret = [rss]
        ret.extend([ridgereg.intercept_])
        ret.extend(ridgereg.coef_)
        return ret

    predictors = ['x']
    predictors.extend(['x_%d' % i for i in range(2, 16)])

    # Set the different values of alpha to be tested
    alpha_ridge = [1e-15, 1e-10, 1e-8, 1e-4, 1e-3, 1e-2, 1, 5, 10, 20]

    # Initialize the dataframe for storing coeficients
    col = ['rss', 'intercept'] + ['coef_x_%d' % i for i in range(1, 16)]
    ind = ['alpha_%.2g' % alpha_ridge[i] for i in range(0, 10)]
    coef_matrix_ridge = pd.DataFrame(index=ind, columns=col)

    models_to_plot = {1e-15: 231, 1e-10: 232, 1e-4: 233, 1e-3: 234, 1e-2: 235, 5: 236}
    for i in range(10):
        coef_matrix_ridge.iloc[i,] = ridge_regression(data, predictors, alpha_ridge[i], models_to_plot)
    file = "static/img/han03.png"
    dic["pic3"] = "/" + file
    plt.savefig(file)
    # plt.show()

    
    def lasso_regression(data, predictors, alpha, models_to_plot={}):
        # Fit the model
        lassoreg = Lasso(alpha=alpha, normalize=True, max_iter=1e5)
        lassoreg.fit(data[predictors], data['y'])
        y_pred = lassoreg.predict(data[predictors])
        # Check if a plot is to be made for the entered alpha
        if alpha in models_to_plot:
            plt.subplot(models_to_plot[alpha])
            plt.tight_layout()
            plt.plot(data['x'], y_pred)
            plt.plot(data['x'], data['y'], '.')
            plt.title('Plot for alpha:%.3g' % alpha)

        # Return the result in pre-defined format
        rss = sum((y_pred - data['y']) ** 2)
        ret = [rss]
        ret.extend([lassoreg.intercept_])
        ret.extend(lassoreg.coef_)
        return ret

    predictors = ['x']
    predictors.extend(['x_%d' % i for i in range(2, 16)])

    # Define the alpha values to test
    alpha_lasso = [1e-15, 1e-10, 1e-8, 1e-5, 1e-4, 1e-3, 1e-2, 1, 5, 10]

    # Initialize the dataframe to store coefficients
    col = ['rss', 'intercept'] + ['coef_x_%d' % i for i in range(1, 16)]
    ind = ['alpha_%.2g' % alpha_lasso[i] for i in range(0, 10)]
    coef_matrix_lasso = pd.DataFrame(index=ind, columns=col)

    # Define the models_to_plot
    models_to_plot = {1e-10: 231, 1e-5: 232, 1e-4: 233, 1e-3: 234, 1e-2: 235, 1: 236}

    # Iterate over the 10 alpha values:
    for i in range(10):
        coef_matrix_lasso.iloc[i,] = lasso_regression(data, predictors, alpha_lasso[i], models_to_plot)
    file = "static/img/han04.png"
    dic["pic4"] = "/" + file
    plt.savefig(file)
    # plt.show()
    return render(request, "overfitting.html", dic)

def show_cross_validation(request):
    context = {}
    normal_rf_accuracy, crossv_rf_accuracy = hanyue.randomforest_1_1()
    rf_nor = '使用2/3的数据集作训练集，rf方法获得的准确率:{}%'.format(normal_rf_accuracy)
    rf_cro = '结合10折交叉验证，rf方法获得的准确率:{}%'.format(crossv_rf_accuracy)
    context["rf_nor"] = rf_nor
    context["rf_cro"] = rf_cro
    normal_lr_accuracy, crossv_lr_accuracy = hanyue.logisticregression_1_1()
    lr_nor = '使用2/3的数据集作训练集，lr方法获得的准确率:{}%'.format(normal_lr_accuracy)
    lr_cro = '结合10折交叉验证，lr方法获得的准确率:{}%'.format(crossv_lr_accuracy)
    context['lr_nor'] = lr_nor
    context['lr_cro'] = lr_cro
    normal_svm_accuracy, crossv_svm_accuracy = hanyue.linersvc_1_1()
    svm_nor = '使用2/3的数据集作训练集，lr方法获得的准确率:{}%'.format(normal_svm_accuracy)
    svm_cro = '结合10折交叉验证，lr方法获得的准确率:{}%'.format(crossv_svm_accuracy)
    context['svm_nor'] = svm_nor
    context['svm_cro'] = svm_cro

    file1 = "static/img/001.png"
    file2 = "static/img/002.png"
    # hanyue.Regulation_origin_1_2(file1, file2)
    context["pic1"] = "/" + file1
    context["pic2"] = "/" + file2

    file3 = "static/img/003.png"
    # hanyue.Regulation_ridge_1_3(file3)#L2
    context["pic3"] = "/" + file3

    file4 = "static/img/004.png"
    # hanyue.Regulation_lasso_1_4(file4)#L1
    context["pic4"] = "/" + file4
    
    
    return render(request, "cross_show.html", context)

def svm_models_show(request):
    context = {}
    file5 = "static/img/005.png"
    file6 = "static/img/006.png"
    file7 = "static/img/007.png"
    file8 = "static/img/008.png"
    file9 = "static/img/009.png"
    file10 = "static/img/010.png"
    file11 = "static/img/011.png"
    file12 = "static/img/012.png"
    file13 = "static/img/013.png"
    file14 = "static/img/014.png"

    # hanyue.SVM_show_sample_1(file5)
    # precision_score, recall_score, f1_score, confusion_matrix = hanyue.SVM_Linear_2(file6)
    # hanyue.SVM_RBF_Kernel_3(file14)
    # hanyue.SVM_Dynamic_Analysis_Parameter_4(file7, file8, file9)
    # hanyue.Multiclass_Classification_5(file10, file11)
    # hanyue.SVM_6_Support_Vector_Regression_6(file12, file13)

    context["pic5"] = "/" + file5
    context["pic6"] = "/" + file6
    context["pic7"] = "/" + file7
    context["pic8"] = "/" + file8
    context["pic9"] = "/" + file9
    context["pic10"] = "/" + file10
    context["pic11"] = "/" + file11
    context["pic12"] = "/" + file12
    context["pic13"] = "/" + file13
    context["pic14"] = "/" + file14
    # context["precision_score"] = precision_score
    # context["recall_score"] = recall_score
    # context["f1_score"] = f1_score
    # context["confusion_matrix"] = confusion_matrix

    return render(request, "svm_models_show.html", context)

def weather_crawler(request):
    file = "static/files/China_city_aqi_update.csv"
    # weather = hanyue.Weather_Web_Crawler_3_1(file)
    weather = []
    with open(file, 'r', encoding='utf-8', newline='') as f:
        reader = csv.reader(f)
        for line in reader:
            weather.append(line)
    context = {}
    context["weather"] = weather

    return render(request, "weather_data.html", context)

def weather_analysize(request):
    file1 = "static/files/China_city_aqi_update.csv"
    file2 = "static/img/top50_aqi_bar_3.png"
    (base_info, data_view, aqi_max, aqi_min, aqi_mean) = hanyue.Weather_Reptile_Analysis_3_2(file1, file2)
    context = {}
    context['pic'] = '/' + file2
    context['base_info'] = base_info
    context['data_view'] = data_view
    context['aqi_max'] = aqi_max
    context['aqi_min'] = aqi_min
    context['aqi_mean'] = aqi_mean

    return render(request, "weather_analysize.html", context)

def get_city(request):
    return render(request, "get-city.html")

def show_weather(request):
    request.encoding = "utf-8"
    city = request.GET['city']
    weather = hanyue.get_city_aqi(city)
    return render(request, "show_weather.html", {"city":weather, "name":city})

def cluster(request):
    pictures = []
    pic_path = "static/img/"
    pictures.append(pic_path+ 'Cluster_show_samples_4_1.png')
    pictures.append(pic_path+ 'Cluster_Kmeans_4_2.png')
    pictures.append(pic_path+ 'Cluster_Anisotropicly_Distribution_show_4_31.png')
    pictures.append(pic_path+ 'Cluster_Anisotropicly_Distribution_cluster_4_32.png')
    pictures.append(pic_path+ 'Cluster_Circular_Distribution_show_4_41.png')
    pictures.append(pic_path+ 'Cluster_Six_Clustering_Algorithms_to_Circular_Distribution_Data_4_42.png')
    pictures.append(pic_path+ 'Cluster_Moon_shape_show_4_51.png')
    pictures.append(pic_path+ 'Cluster_Six_Clustering_Algorithms_to_Moon_shape_Data_4_52.png')
    pictures.append(pic_path+ 'Cluster_blobs_with_varied_variances_show_4_61.png')
    pictures.append(pic_path+ 'Cluster_Six_Clustering_Algorithms_to_blobs_with_varied_variances_Data_4_62.png')
    pictures.append(pic_path+ 'Cluster_anisotropy_Data_Show_4_71.png')
    pictures.append(pic_path+ 'Cluster_Six_Clustering_Algorithms_to_anisotropy_Data_4_72.png')
    pictures.append(pic_path+ 'Cluster_Random_Sample_Data_Show_4_81.png')
    pictures.append(pic_path+ 'Cluster_Six_Clustering_Algorithms_to_Random_Sample_Data_4_82.png')
    pictures.append(pic_path+ 'Cluster_Hierarchical_Clustering_4_9.png')

    # hanyue.Cluster_4(pictures)

    return render(request, "cluster_show.html", {'pics': pictures})

def cut_word_arg(request):
    return render(request, "cut_word_arg.html")

def cut_word(request):
    request.encoding = "utf-8"
    name = request.GET['name']
    bk = request.GET['bk']
    file = "static/files/" + name + "_" + bk + "_split_unattributed.txt"
    with open(file, "r", encoding="utf-8") as f:
        text = f.readlines()
        text = list(text)
    file1 = "static/files/" + name + "_" + bk + ".txt"
    with open(file1, "r", encoding="utf-8") as f:
        raw = list(f.readlines())
    return render(request, "cut_word.html", {'name':name, 'bk':bk, 'text':text, "raw":raw})

def info_extract_arg(request):
    return render(request, "info_extract_arg.html")

def info_extract(request):
    request.encoding = "utf-8"
    name = request.GET['name']
    bk = request.GET['bk']
    file = "static/files/" + name + "_" + bk + "_Extract_Job_Attribution_Value.txt"
    with open(file, "r", encoding="utf-8") as f:
        text = list(f.readlines())
    for i in range(len(text)):
        text[i] = eval(text[i])
    return render(request, "info_extract.html", {"text":text, "name":name, "bk":bk})




