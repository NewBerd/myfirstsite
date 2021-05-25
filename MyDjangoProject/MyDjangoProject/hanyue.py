from __future__ import print_function, unicode_literals
import csv
import time
import warnings
import numpy as np
import pandas as pd
import scipy as sp
import itertools
from itertools import cycle, islice

import matplotlib 
from matplotlib.pylab import rcParams
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import ListedColormap

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn import datasets
from sklearn.model_selection import KFold
import sklearn as sk
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_circles, make_classification, make_blobs
from sklearn.svm import SVC
from sklearn.gaussian_process.kernels import RBF
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from sklearn.multiclass import OneVsOneClassifier, OneVsRestClassifier
from sklearn.svm import SVR
from sklearn import cluster, datasets, mixture
from sklearn.neighbors import kneighbors_graph

from mlxtend.plotting import plot_decision_regions

import requests
from bs4 import BeautifulSoup


from bosonnlp import BosonNLP
import re

####################模块一############
def randomforest_1_1():
    # 实验一：对比不同模型的cross-validation结果
    # 用load_wine方法导入数据
    wine_data = datasets.load_wine()
    data_input = wine_data.data
    data_output = wine_data.target

    # 将数据集分为2/3的训练集和1/3的测试集
    X_train,X_test,y_train,y_test = train_test_split(data_input,
        data_output,test_size=0.33,random_state=42)
    
    # 1.结合K折交叉验证，使用随机森林方法观察准确率
    rf_class = RandomForestClassifier()
    rf_class.fit(X_train,y_train)
    y_rf_pred = rf_class.predict(X_test)
    normal_rf_accuracy = accuracy_score(y_test, y_rf_pred)*100
    crossv_rf_accuracy = cross_val_score(rf_class, data_input, 
        data_output, scoring='accuracy', cv=10).mean()*100
    # print('使用2/3的数据集作训练集，rf方法获得的准确率:{}%'.format(normal_rf_accuracy))
    # print('结合10折交叉验证，rf方法获得的准确率:{}%'.format(crossv_rf_accuracy))
    return normal_rf_accuracy, crossv_rf_accuracy

def logisticregression_1_1():
    # 实验一：对比不同模型的cross-validation结果
    # 用load_wine方法导入数据
    wine_data = datasets.load_wine()
    data_input = wine_data.data
    data_output = wine_data.target

    # 将数据集分为2/3的训练集和1/3的测试集
    X_train,X_test,y_train,y_test = train_test_split(data_input,
        data_output,test_size=0.33,random_state=42)
    # 2.结合K折交叉验证，使用支持向量机方法观察准确率
    lr_class = LogisticRegression()
    lr_class.fit(X_train,y_train)
    y_lr_pred = lr_class.predict(X_test)
    normal_lr_accuracy = accuracy_score(y_test, y_lr_pred)*100
    crossv_lr_accuracy = cross_val_score(lr_class, 
        data_input, data_output, scoring='accuracy', cv=10).mean()*100
    
    # print('使用2/3的数据集作训练集，svm方法获得的准确率:{}'.format(normal_svm_accuracy) + '%')
    # print('结合10折交叉验证，svm方法获得的准确率:{}'.format(crossv_svm_accuracy) + '%')
    return normal_lr_accuracy, crossv_lr_accuracy

def linersvc_1_1():
    # 实验一：对比不同模型的cross-validation结果
    # 用load_wine方法导入数据
    wine_data = datasets.load_wine()
    data_input = wine_data.data
    data_output = wine_data.target

    # 将数据集分为2/3的训练集和1/3的测试集
    X_train,X_test,y_train,y_test = train_test_split(data_input,
        data_output,test_size=0.33,random_state=42)
    # 3.结合K折交叉验证，使用逻辑回归方法观察准确率
    svm_class = svm.LinearSVC()
    svm_class.fit(X_train,y_train)
    y_svm_pred = svm_class.predict(X_test)
    normal_svm_accuracy = accuracy_score(y_test, y_svm_pred)*100
    crossv_svm_accuracy = cross_val_score(svm_class, 
        data_input, data_output, scoring='accuracy', cv=10).mean()*100
    # print('使用2/3的数据集作训练集，LR方法获得的准确率:{}%'.format(normal_lr_accuracy))
    # print('结合10折交叉验证，LR方法获得的准确率:{}%'.format(crossv_lr_accuracy))

    return normal_svm_accuracy, crossv_svm_accuracy

def Regulation_origin_1_2(figfile1, figfile2):
    #数据预处理
    x = np.array([1.4 * i * np.pi / 180 for i in range(0, 300, 4)])
    np.random.seed(20)  # 固定每次生成的随机数
    y = np.sin(x) + np.random.normal(0, 0.2, len(x))
    data = pd.DataFrame(np.column_stack([x, y]), columns=['x', 'y'])
    fig, ax = plt.subplots()
    ax.plot(data['x'], data['y'], '.')
    fig.savefig(figfile1)
    plt.close()

    for i in range(2, 16):  # power of 1 is already there
        colname = 'x_%d' % i  # new var will be x_power
        data[colname] = data['x'] ** i

    # 复杂度可变
    def linear_regression(data, power, models_to_plot):
        # initialize predictors:
        import matplotlib.pyplot as plt
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
    # 注意上行代码的columns不能写成column单数，画图无法画出来！
    # 定义作图的位置与模型的复杂度
    models_to_plot = {1: 231, 3: 232, 6: 233, 8: 234, 11: 235, 14: 236}

    # 画出来
    for i in range(1, 16):
        coef_matrix_simple.iloc[i - 1, 0:i + 2] = linear_regression(data, power=i, models_to_plot=models_to_plot)
    plt.savefig(figfile2)
    plt.close()

def Regulation_ridge_1_3(file):
    #L2正则化
    rcParams['figure.figsize'] = 12, 10
    def ridge_regression(data, predictors, alpha, models_to_plot={}):
        # Fit the model:1.初始化模型配置  2.模型拟合  3.模型预测
        import matplotlib.pyplot as plt
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

    # 定义作图的位置与模型的复杂度
    models_to_plot = {1e-15: 231, 1e-10: 232, 1e-4: 233, 1e-3: 234, 1e-2: 235, 5: 236}
    x = np.array([1.4 * i * np.pi / 180 for i in range(0, 300, 4)])
    np.random.seed(20)  # 固定每次生成的随机数
    y = np.sin(x) + np.random.normal(0, 0.2, len(x))
    data = pd.DataFrame(np.column_stack([x, y]), columns=['x', 'y'])
    for i in range(2, 16):  # power of 1 is already there
        colname = 'x_%d' % i  # new var will be x_power
        data[colname] = data['x'] ** i

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

    plt.savefig(file)
    plt.close()

def Regulation_lasso_1_4(file):
    #L1正则化
    rcParams['figure.figsize'] = 12, 10
    def lasso_regression(data, predictors, alpha, models_to_plot={}):
        # Fit the model
        import matplotlib.pyplot as plt
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

    x = np.array([1.4 * i * np.pi / 180 for i in range(0, 300, 4)])
    np.random.seed(20)  # 固定每次生成的随机数
    y = np.sin(x) + np.random.normal(0, 0.2, len(x))
    data = pd.DataFrame(np.column_stack([x, y]), columns=['x', 'y'])
    for i in range(2, 16):  # power of 1 is already there
        colname = 'x_%d' % i  # new var will be x_power
        data[colname] = data['x'] ** i

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

    plt.savefig(file)
    plt.close()



############模块2################
def SVM_show_sample_1(file):
    X, y = make_classification(n_features=2, n_redundant=0, n_informative=2, n_samples=200,
                               random_state=1, n_clusters_per_class=1)
    rng = np.random.RandomState(2)
    X += 2 * rng.uniform(size=X.shape)

    # 定义plot的colormap
    cm = plt.cm.RdBu
    cm_bright = ListedColormap(['#FF0000', '#0000FF'])

    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cm_bright)
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.savefig(file)
    plt.close()

def SVM_Linear_2(file):
    X, y = make_classification(n_features=2, n_redundant=0,
        n_informative=2, n_samples=200,random_state=1, n_clusters_per_class=1)
    rng = np.random.RandomState(2)
    X += 2 * rng.uniform(size=X.shape)

    # 线性核函数的支持向量机去训练样本
    C = 0.1
    clf = SVC(kernel="linear", C=C)

    # 在训练之前对数据进行normalization
    X = StandardScaler().fit_transform(X)

    clf.fit(X, y)
    y_pred = clf.predict(X)

    prec = precision_score(y_true=y, y_pred=y_pred, pos_label=1)
    rec = recall_score(y_true=y, y_pred=y_pred, pos_label=1)
    f1 = f1_score(y_true=y, y_pred=y_pred, pos_label=1)
    precision_score_ = "Precision score is : {:.2f}".format(prec)
    recall_score_ = "Recall score is : {:.2f}".format(rec)
    f1_score_ = "f1 score is : {:.2f}".format(f1)
    confusion_matrix_ = "Confusion matrix is :{}".format(confusion_matrix(y_pred=y_pred, y_true=y))

    plot_decision_regions(X, y, clf=clf, colors='orange,navy')
    plt.title("SVM with linear kernel")
    plt.savefig(file)
    plt.close()

    return precision_score_, recall_score_, f1_score_, confusion_matrix_

def SVM_RBF_Kernel_3(file):

    X, y = make_classification(n_features=2, n_redundant=0, n_informative=2,
                       n_samples=200, random_state=1, n_clusters_per_class=1)
    rng = np.random.RandomState(2)
    X += 2 * rng.uniform(size=X.shape)

    C = 0.1    # clf = SVC(kernel = "rbf", gamma = 2., C = C) kernel不选，默认就是rbf
    clf = SVC(gamma=2., C=C)

    clf.fit(X, y)
    y_pred = clf.predict(X)

    prec = precision_score(y_true=y, y_pred=y_pred, pos_label=1)
    rec = recall_score(y_true=y, y_pred=y_pred, pos_label=1)
    f1 = f1_score(y_true=y, y_pred=y_pred, pos_label=1)
    precision_score_ = "Precision score is : {:.2f}".format(prec)
    recall_score_ = "Recall score is : {:.2f}".format(rec)
    f1_score_ = "f1 score is : {:.2f}".format(f1)
    confusion_matrix_ = "Confusion matrix is :{}".format(confusion_matrix(y_pred=y_pred, y_true=y))

    plot_decision_regions(X, y, clf=clf, colors='orange,navy')
    plt.title("SVM with rbf kernel")
    plt.savefig(file)
    plt.close()

def SVM_Dynamic_Analysis_Parameter_4(file, figfile1, figfile2):
    X, y = make_classification(n_features=2, n_redundant=0, n_informative=2,
                        n_samples=200, random_state=1, n_clusters_per_class=1)
    rng = np.random.RandomState(2)
    X += 2 * rng.uniform(size=X.shape)

    C_list = [0.1, 1., 10.]
    gamma_list = [0.2, 2., 20.]

    plt.figure(figsize=(15, 5))
    for i, C in enumerate(C_list):
        clf = SVC(C=C)
        clf.fit(X, y)
        # y_pred = clf.predict(X)
        plt.subplot(1, len(C_list), i + 1)
        plot_decision_regions(X, y, clf=clf, colors='orange,navy')
        plt.title("SVM with rbf kernel, C = {:.4f}".format(C))
    plt.tight_layout()
    plt.savefig(file)
    plt.close()


    gamma_list = [0.05, 2., 20.]
    plt.figure(figsize=(15, 5))
    for i, gamma in enumerate(gamma_list):
        clf = SVC(gamma=gamma, C=1.0)
        clf.fit(X, y)
        # y_pred = clf.predict(X)
        plt.subplot(1, len(C_list), i + 1)
        plot_decision_regions(X, y, clf=clf, colors='orange,navy')
        plt.title("SVM with rbf kernel, gamma = {:.2f}".format(gamma))
    plt.tight_layout()
    plt.savefig(figfile1)
    plt.close()

    names = ["Linear SVM I", "Linear SVM II", "rbf SVM", "Poly SVM", "Sigmoid SVM"]
    models = [  SVC(kernel="linear", C=C),
                sk.svm.LinearSVC(C=C),
                SVC(kernel="rbf", gamma=2, C=C),  # or SVC(gamma = 2, C = 1)
                SVC(kernel="poly", degree=5, C=C),
                SVC(kernel="sigmoid", C=C)]

    figure = plt.figure(figsize=(10, 10))
    assert len(names) == len(models)
    for i, (clf_name, clf) in enumerate(zip(names, models)):
        plt.subplot((len(models) + 1) / 2, 2, i + 1)
        clf.fit(X, y)
        plot_decision_regions(X, y, clf, colors='orange,navy')
        plt.title(clf_name, fontsize=12)
    plt.tight_layout()
    plt.savefig(figfile2)
    plt.close()

def Multiclass_Classification_5(figfile1, figfile2):
    X, y = make_classification(n_features=2, n_redundant=0, n_informative=2, n_samples=200,
                               random_state=1, n_clusters_per_class=1)
    rng = np.random.RandomState(2)
    X += 2 * rng.uniform(size=X.shape)

    C_list = [0.1, 1., 10.]
    gamma_list = [0.2, 2., 20.]

    X2, Y2 = make_blobs(n_samples=200, n_features=2, centers=5, random_state=1, cluster_std=1)
    plt.scatter(X2[:, 0], X2[:, 1], c=Y2)
    plt.savefig(figfile1)
    plt.close()

    rbf_ovr = SVC(kernel='rbf', decision_function_shape="ovr")
    # rbf_ovo = SVC(kernel = 'rbf', decision_function_shape = "ovo")
    rbf_ovo = OneVsOneClassifier(SVC(kernel="rbf"))
    linear_ovr = SVC(kernel='linear', decision_function_shape="ovr")
    # linear_ovo = SVC(kernel = 'linear', decision_function_shape = "ovo")
    linear_ovo = OneVsOneClassifier(SVC(kernel="linear"))

    gs = gridspec.GridSpec(2, 2)
    fig = plt.figure(figsize=(10, 8))

    clfs = [rbf_ovr, rbf_ovo, linear_ovr, linear_ovo]
    names = ["RBF SVM OVR", "RBF SVM OVO", "Linear SVM OVR", "Linear SVM OVO"]

    for clf, lab, grd in zip(clfs, names, itertools.product([0, 1], repeat=2)):
        ax = plt.subplot(gs[grd[0], grd[1]])
        clf.fit(X2, Y2)
        fig = plot_decision_regions(X=X2, y=Y2, clf=clf, legend=2)
        plt.title(lab)
    plt.savefig(figfile2)
    plt.close()

def SVM_6_Support_Vector_Regression_6(figfile1, figfile2):
    X = np.sort(5 * np.random.rand(40, 1), axis=0)
    y = np.sin(X).ravel()

    # 人为增加一些噪声
    y[::5] += 3 * (0.5 - np.random.rand(8))

    plt.scatter(X, y)
    plt.xlabel("X")
    plt.ylabel("y")
    plt.savefig(figfile1)
    plt.close()
    # SVR 拟合数据
    C = 1e3
    svr_rbf = SVR(kernel='rbf', C=C, gamma=0.1)
    svr_lin = SVR(kernel='linear', C=C)
    svr_poly = SVR(kernel='poly', C=C, degree=2)
    y_rbf = svr_rbf.fit(X, y).predict(X)
    y_lin = svr_lin.fit(X, y).predict(X)
    y_poly = svr_poly.fit(X, y).predict(X)

    lw = 4
    plt.figure(figsize=(10, 5))
    plt.scatter(X, y, color='darkorange', label='data')
    plt.plot(X, y_rbf, color='navy', lw=lw, label='RBF model')
    plt.plot(X, y_lin, color='c', lw=lw, label='Linear model')
    plt.plot(X, y_poly, color='cornflowerblue', lw=lw, label='Polynomial model')
    plt.xlabel('data')
    plt.ylabel('target')
    plt.title('Support Vector Regression')
    plt.legend(loc='lower left')
    plt.savefig(figfile2)
    plt.close()

###########################***模块4***######################
def Weather_Web_Crawler_3_1(file):
    China_city_aqi_update = []
    def get_city_aqi_3_1_1(city_pinyin):
        """
            获取城市的AQI
        """
        url = 'http://pm25.in/' + city_pinyin
        r = requests.get(url, timeout=30)
        soup = BeautifulSoup(r.text, 'lxml')
        div_list = soup.find_all('div', {'class': 'span1'})
        city_aqi = []
        for i in range(8):
            div_content = div_list[i]
            caption = div_content.find('div', {'class', 'caption'}).text.strip()
            value = div_content.find('div', {'class', 'value'}).text.strip()
            city_aqi.append(value)
        return city_aqi

    def get_all_cities3_1_2():
        """
            获取所有的城市
        """
        url = 'http://www.pm25.in/'
        r = requests.get(url, timeout=30)
        soup = BeautifulSoup(r.text, 'lxml')

        city_list = []
        city_div = soup.find_all('div', {'class': 'bottom'})[1]
        city_link_list = city_div.find_all('a')
        for city_link in city_link_list:
            city_name = city_link.text
            city_pinyin = city_link['href'][1:]
            city_list.append((city_name, city_pinyin))
        return city_list
    
    city_list = get_all_cities()
    header = ['City', 'AQI', 'PM2.5/h', 'PM10/h', 'CO/1h', 'NO2/1h', 'O3/1h', 'O3/8h', 'SO2/1h']
    China_city_aqi_update.append(header)
    # 将信息录入csv文件
    with open(file, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)   # 左边写法正确，而f.writerow()的写法错误,是json文件格式
        writer.writerow(header)
        for i, city in enumerate(city_list):
            if (i+1) % 10 == 0:
                print('已处理{}条数据。(总共有{}条数据)'.format(i+1, len(city_list)))
            city_name = city[0]
            city_pinyin = city[1]
            city_aqi = get_city_aqi(city_pinyin)
            row = [city_name] + city_aqi
            China_city_aqi_update.append(row)
            writer.writerow(row)
    return China_city_aqi_update

def Weather_Reptile_Analysis_3_2(file1, file2):
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    aqi_data = pd.read_csv(file1)
    # print('基本信息：\n{}'.format(aqi_data.info))
    base_info = '基本信息：\n{}'.format(aqi_data.info)
    # print('数据预览：\n{}'.format(aqi_data.head(5)))
    data_view = '数据预览：\n{}'.format(aqi_data.head(5))
    clean_aqi_data = aqi_data[aqi_data['AQI'] > 0]
    # print('经过数据清洗后，AQI的最大值为：\n{}'.format(clean_aqi_data['AQI'].max()))
    # print('经过数据清洗后，AQI的最小值为：\n{}'.format(clean_aqi_data['AQI'].min()))
    # print('经过数据清洗后，AQI的均值为：\n{}'.format(clean_aqi_data['AQI'].mean()))
    aqi_max = '经过数据清洗后，AQI的最大值为：\n{}'.format(clean_aqi_data['AQI'].max())
    aqi_min = '经过数据清洗后，AQI的最小值为：\n{}'.format(clean_aqi_data['AQI'].min())
    aqi_mean = '经过数据清洗后，AQI的均值为：\n{}'.format(clean_aqi_data['AQI'].mean())
    # top50城市
    top50_cities = clean_aqi_data.sort_values(by=['AQI']).head(50)
    top50_cities.plot(kind='bar', x='City', y='AQI', title='空气质量最好的50个城市',
                      figsize=(20, 10))
    plt.savefig(file2)
    plt.close()

    return (base_info, data_view, aqi_max, aqi_min, aqi_mean)


############-----------模块5-----------#########

def Cluster_4(fig):
    #fig是一个文件名列表
    n_samples = 1500
    random_state = 170
    plt.figure(figsize=(5, 3))
    n_samples = 1500
    X, y = datasets.make_blobs(n_samples=n_samples, centers=3, random_state=170)

    plt.scatter(X[:, 0], X[:, 1])
    # plt.savefig('Cluster_show_samples_4_1.png')
    plt.savefig(fig[0])


    # Kmeans对cluster数目非常敏感
    plt.figure(figsize=(10, 8))

    ## 尝试对上面的数据做一下Kmeans 聚类，类的数目可以自己定义
    ## 比如
    n_clusters = [2, 3, 4, 5]
    for i, n in enumerate(n_clusters):
        plt.subplot(2, 2, i + 1)
        y_pred = cluster.KMeans(n_clusters=n, random_state=random_state).fit_predict(X)
        colors = np.array(list(islice(cycle(['#377eb8', '#ff7f00', '#4daf4a',
                                             '#f781bf', '#a65628', '#984ea3',
                                             '#999999', '#e41a1c', '#dede00']),
                                      int(max(y_pred) + 1))))
        plt.scatter(X[:, 0], X[:, 1], color=colors[y_pred])
        plt.title("{} cluster".format(n))
    # plt.savefig('Cluster_Kmeans_4_2.png')
    plt.savefig(fig[1])


    # Kmeans对数据分布的敏感度
    # 对数据进行一个转换，使得不再均匀分布， 有各向异性
    # Kmeans的前提条件是cluster是圆形区域的

    transformation = [[0.6, -0.636], [-0.40, 0.85]]
    X_aniso = np.dot(X, transformation)

    ## 对X_aniso 做Kmeans，看看有什么结论
    plt.scatter(X_aniso[:, 0], X_aniso[:, 1])
    # plt.savefig('Cluster_Anisotropicly_Distribution_show_4_31.png')
    plt.savefig(fig[2])


    ## 对X_aniso 做Kmeans，看看有什么结论
    y_pred = cluster.KMeans(n_clusters=3, random_state=random_state).fit_predict(X_aniso)
    colors = np.array(list(islice(cycle(['#377eb8', '#ff7f00', '#4daf4a',
                                         '#f781bf', '#a65628', '#984ea3',
                                         '#999999', '#e41a1c', '#dede00']),
                                  int(max(y_pred) + 1))))
    plt.scatter(X_aniso[:, 0], X_aniso[:, 1], color=colors[y_pred])
    plt.title("Anisotropicly Distributed Blobs")
    # plt.savefig('Cluster_Anisotropicly_Distribution_cluster_4_32.png')
    plt.savefig(fig[3])

    # 圆形分布的数据
    X, y = datasets.make_circles(n_samples=n_samples, factor=.5, noise=.05)
    X = StandardScaler().fit_transform(X)
    print("X shape: " + str(X.shape))
    print("y shape: " + str(y.shape))
    plt.figure(figsize=(5, 4))
    plt.scatter(X[:, 0], X[:, 1])
    # plt.savefig('Cluster_Circular_Distribution_show_4_41.png')
    plt.savefig(fig[4])



    # connectivity concept, kneighbors_graph
    X_test = np.array([[0], [3], [1], [5]])
    A = kneighbors_graph(X_test, n_neighbors=2, mode='connectivity', include_self=False)
    print(A.toarray())

    # 定义不同的clustering算法的训练及可视化函数
    def plot_cluster(X, y, params, save_name):
        # 利用kneighbors_graph， 建立 connectivity
        connectivity = kneighbors_graph(X, n_neighbors=params["n_neighbors"])
        # make connectivity symmetric
        connectivity = 0.5 * (connectivity + connectivity.T)

        # 定义kmeans，cluster数目根据传进来的参量
        kmeans = cluster.KMeans(n_clusters=params["n_clusters"])

        # 定义DBSCAN，eps和min_samples由传进来的params参量获得
        dbscan = cluster.DBSCAN(eps=params['eps'], min_samples=params["min_samples"])

        # 根据eulidean距离的average还是maximum得到 AgglomerativeClustering算法的两种average和complete linkage
        average_linkage = cluster.AgglomerativeClustering(linkage="average",
                                                          affinity="euclidean",
                                                          n_clusters=params['n_clusters'],
                                                          connectivity=connectivity)

        complete_linkage = cluster.AgglomerativeClustering(linkage="complete",
                                                           affinity="euclidean",
                                                           n_clusters=params['n_clusters'],
                                                           connectivity=connectivity)
        # GMM算法
        gmm = mixture.GaussianMixture(n_components=params['n_clusters'])

        # Spectral clustering的算法
        spectral = cluster.SpectralClustering(n_clusters=params['n_clusters'],
                                              affinity="nearest_neighbors")

        clustering_algorithms = (
            ('Kmeans', kmeans),
            ('DBSCAN', dbscan),
            ('Average linkage agglomerative clustering', average_linkage),
            ('Complete linkage agglomerative clustering', complete_linkage),
            ('Spectral clustering', spectral),
            ('GaussianMixture', gmm)
        )

        plt.figure(figsize=(10, 10))

        for i, (alg_name, algorithm) in enumerate(clustering_algorithms):
            t0 = time.time()
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                algorithm.fit(X)
            t1 = time.time()
            if hasattr(algorithm, 'labels_'):
                y_pred = algorithm.labels_.astype(np.int)
            else:
                y_pred = algorithm.predict(X)  # GMM

            plt.subplot(3, 2, i + 1)
            colors = np.array(list(islice(cycle(['#377eb8', '#ff7f00', '#4daf4a',
                                                 '#f781bf', '#a65628', '#984ea3',
                                                 '#999999', '#e41a1c', '#dede00']),
                                          int(max(y_pred) + 1))))
            plt.scatter(X[:, 0], X[:, 1], s=10, color=colors[y_pred])

            plt.xlim(-2.5, 2.5)
            plt.ylim(-2.5, 2.5)
            plt.xticks(())
            plt.yticks(())
            plt.title(alg_name)
            plt.text(.99, .01, ('time = %.2fs' % (t1 - t0)).lstrip('0'),
                     transform=plt.gca().transAxes, size=15,
                     horizontalalignment='right')
        plt.savefig(save_name)


    params = {'eps': .3, 'n_clusters': 4, "n_neighbors": 10, "min_samples": 10}
    plot_cluster(X, y, params=params,save_name=fig[5])
    #'Cluster_Six_Clustering_Algorithms_to_Circular_Distribution_Data_4_42.png'

    # moon shape
    X, y = datasets.make_moons(n_samples=n_samples, noise=.05)
    X = StandardScaler().fit_transform(X)
    print("X shape: " + str(X.shape))
    print("y shape: " + str(y.shape))

    plt.figure(figsize=(5, 4))
    plt.scatter(X[:, 0], X[:, 1])
    # plt.savefig('Cluster_Moon_shape_show_4_51.png')
    plt.savefig(fig[6])



    params = {'eps': .3, 'n_clusters': 3, "n_neighbors": 20, "min_samples": 10}
    plot_cluster(X, y, params=params, save_name=fig[7])
    # 'Cluster_Six_Clustering_Algorithms_to_Moon_shape_Data_4_52.png'

    # blobs with varied variances
    X, y = datasets.make_blobs(n_samples=n_samples,
                               cluster_std=[1.0, 2.5, 0.5],
                               random_state=random_state)

    X = StandardScaler().fit_transform(X)
    print("X shape: " + str(X.shape))
    print("y shape: " + str(y.shape))
    plt.figure(figsize=(5, 4))
    plt.scatter(X[:, 0], X[:, 1])
    # plt.savefig('Cluster_blobs_with_varied_variances_show_4_61.png')
    plt.savefig(fig[8])



    params = {'eps': 0.18, 'n_clusters': 3, "n_neighbors": 2, "min_samples": 5}
    plot_cluster(X, y, params=params, save_name=fig[9])
    # 'Cluster_Six_Clustering_Algorithms_to_blobs_with_varied_variances_Data_4_62.png'
    # anisotropy
    X, y = datasets.make_blobs(n_samples=n_samples, random_state=random_state)
    transformation = [[0.6, -0.6], [-0.4, 0.8]]
    X = np.dot(X, transformation)
    X = StandardScaler().fit_transform(X)
    print("X shape: " + str(X.shape))
    print("y shape: " + str(y.shape))

    plt.figure(figsize=(5, 4))
    plt.scatter(X[:, 0], X[:, 1])
    # plt.savefig('Cluster_anisotropy_Data_Show_4_71.png')
    plt.savefig(fig[10])



    params = {'eps': 0.18, 'n_clusters': 3, "n_neighbors": 2, "min_samples": 5}
    plot_cluster(X, y, params=params, save_name=fig[11])
    # 'Cluster_Six_Clustering_Algorithms_to_anisotropy_Data_4_72.png'

    # 对随机样本的聚类
    X, y = np.random.rand(n_samples, 2), None
    X = StandardScaler().fit_transform(X)

    plt.figure(figsize=(5, 4))
    plt.scatter(X[:, 0], X[:, 1], color='#377eb8')
    # plt.savefig('Cluster_Random_Sample_Data_Show_4_81.png')
    plt.savefig(fig[12])



    params = {'eps': 0.3, 'n_clusters': 3, "n_neighbors": 10, "min_samples": 5}
    plot_cluster(X, y, params=params, save_name=fig[13])
    # 'Cluster_Six_Clustering_Algorithms_to_Random_Sample_Data_4_82.png'

    # 层次聚类方法
    from scipy.cluster import hierarchy
    ytdist = np.array([662., 877., 255., 412., 996., 295., 468., 268.,
                       400., 754., 564., 138., 219., 869., 669.])
    Z = hierarchy.linkage(ytdist, method='average', metric="euclidean")
    plt.figure()
    dn = hierarchy.dendrogram(Z)
    # plt.savefig('Cluster_Hierarchical_Clustering_4_9.png')
    plt.savefig(fig[14])

#----------------爬虫-------------
def get_city_aqi(city_pinyin):
    """
        获取城市的AQI
    """
    url = 'http://pm25.in/' + city_pinyin
    r = requests.get(url, timeout=30)
    soup = BeautifulSoup(r.text, 'lxml')
    div_list = soup.find_all('div', {'class': 'span1'})
    # find_all()函数是找到目标字符串的所在位置，即找到该节点，但并不会返回节点里的内容，
    # 'div'和{'class': 'span1'}都是匹配的关键词，助于找到该节点位置

    # print(div_list)
    city_aqi = []
    for i in range(8):
        div_content = div_list[i]

        caption = div_content.find('div', {'class', 'caption'}).text.strip()
        # find()函数本身是找到节点位置，并不会深入获取节点的内容，
        # 使用.text函数是获取节点文本内容的函数，后面加上的strip()函数是去除获取内容里的所有空格
        value = div_content.find('div', {'class', 'value'}).text.strip()
        city_aqi.append((caption, value))
    return city_aqi


#---------------简历--------------
def Text_Segmentation_5_1():
    input_txt = open('static/files/方滨兴_互动百科.txt', 'r', encoding='utf-8')
    # 有的文件编码使用GBK形式，在读文件时需要再添加一个参数：encoding='utf-8'
    # 有的记事本文件编码使用ANSI,读文件添加encoding='utf-8'反而会报错

    lines = input_txt.readlines()
    input_txt.close()

    for line in lines:
        nlp = BosonNLP('QhCMB7FS.33943.0OYvhfw0JCx8')
        result = nlp.tag(line)[0]['word']
        output_txt = open('static/files/方滨兴_互动百科_split_unattributed.txt', mode='a', encoding='utf-8')
        # output_txt.write('{}\n'.format(result))             # 以列表字符串的形式写入
        output_txt.write('{}\n'.format(' '.join(result)))     # 以纯文本的形式写入
        output_txt.close()



job_list = []
personal_experience = []
# work_experience = []  改成放到函数中不会出现检查警告


def process_personal_experience(personal_experience_list):
    work_experience = []
    edu_element = ['人物 经历', '社会 任职', '学位', '出生', '学习', '就读', '博士', '硕士', '本科']
    for exp in personal_experience_list:
        count = 0
        for element in edu_element:
            if element not in exp:
                count += 1
                # print('yes')
        if count == 9:
            work_experience.append(exp)
            # print(work_experience)
    return work_experience


def read_file(filename1, filename2):
    with open(filename1, 'r', encoding='utf-8') as jobs:
        job_list = eval(jobs.read())
    with open(filename2, 'r', encoding='utf-8') as f:
        resume_txt = f.readlines()
    return job_list, resume_txt


def process_work_experience(work_experience_list):
    global job_list
    # with open('title_list.py', 'r', encoding='utf-8') as jobs:
    #     job_list = eval(jobs.read())
    time_unit_job_list = []
    for exp in work_experience_list:
        # 获取任职起始时间点
        # print(exp)
        begin_time_pre = re.findall(
            r'(\d*年|\d*年 \d*月|\d*年\d*月|\d*.\d*|\d*|\d*年 起|\d* 起|\d*年 \d*月 \d*日) (—|－|-|——|至|，).*?', exp)
        if begin_time_pre:
            begin_time = begin_time_pre[0][0]
        else:
            begin_time = []
        # print(begin_time)
        # 获取任职终止时间点
        end_time_pre = re.findall(r'.*(—|－|-|——|至) (\d*年 \d*月|\d*年|\d*.\d*|\d*)', exp)
        if end_time_pre:
            end_time = end_time_pre[0][1]
        else:
            end_time = ''
        job_out_list = []  # 该列表必须在循环里面，对每条工作经历文本单独处理
        job_txt1 = re.findall(r'.* (当选|担任|入选|任|任命 为|任命|成立|加入) (.*?) (,|、|，) (.*).*? (。|;|；)', exp)
        job_new_txt1 = []
        if job_txt1:
            for info in job_txt1[0]:
                if info not in ['成立', '加入', '担任', '当选', '入选', '任', '任命', '任命 为', ',', '、', '，', '。', ';', '；']:
                    # print(info)
                    job_new_txt1.append(info)
        # print(job_new_txt1)
        if job_new_txt1:
            for each in job_new_txt1:
                if each not in job_out_list:
                    # print(each)
                    job_out_list.append(each)
        # print(job_txt1)
        job_txt2 = re.findall(r'.* (当选|担任|入选|任|任命 为|任命|成立|加入) (.*?) (,|，|。|;|；)', exp)
        # print(job_txt2)
        job_new_txt2 = []
        if job_txt2:
            for info in job_txt2[0]:
                if info not in ['成立', '加入', '担任', '当选', '入选', '任', '任命', '任命 为', ',', '、', '，', '。', ';', '；']:
                    # print(info)
                    job_new_txt2.append(info)
        if job_new_txt2:
            for each in job_new_txt2:
                if each not in job_out_list:
                    # print(each)
                    job_out_list.append(each)
        job_txt3 = re.findall(r'.* 在 (.*?) 工作 (，|。|;|；) .* 担任(.*?) 。', exp)
        job_new_txt3 = []
        if job_txt3:
            for info in job_txt3[0]:
                if info not in ['成立', '加入', '担任', '当选', '入选', '任', '任命', '任命 为', ',', '、', '，', '。', ';', '；']:
                    # print(info)
                    job_new_txt3.append(info)
        if job_new_txt3:
            for each in job_new_txt3:
                if each not in job_out_list:
                    # print(each)
                    job_out_list.append(each)
        job_txt4 = re.findall(r'.* 加入 (.*?) (，|。|;|；) .* 担任 (.*?) 。', exp)
        job_new_txt4 = []
        if job_txt4:
            for info in job_txt4[0]:
                if info not in ['成立', '加入', '担任', '当选', '入选', '任', '任命', '任命 为', ',', '、', '，', '。', ';', '；']:
                    # print(info)
                    job_new_txt4.append(info)
        if job_new_txt4:
            for each in job_new_txt4:
                if each not in job_out_list:
                    # print(each)
                    job_out_list.append(each)
        # print(job_out_list)

        job_filter2_symbol_list = []
        for one in job_out_list:  # 对每一条任职信息都预处理一遍，过滤检查
            if '、' in one:  # 去除任职文本里的顿号
                job_filter2 = one.split('、')
                # print(job_filter2)
                for one_job in job_filter2:
                    one_job_ = one_job.strip()
                    if one_job_ not in job_filter2_symbol_list:
                        job_filter2_symbol_list.append(one_job_)
            else:
                if one.strip() not in job_filter2_symbol_list:
                    job_filter2_symbol_list.append(one.strip())
            # print(job_filter2_symbol_list)

        job_filter3_number_list = []
        for one_job in job_filter2_symbol_list:
            # 去除任职信息末尾的标注信息如:[7]
            # print(one_job)
            remove_number = re.findall(r'.* (\[ \d* \]).*', one_job)
            # print(remove_number)
            if remove_number:
                job_filter3_num = one_job.replace(remove_number[0], '')
                if job_filter3_num.strip() not in job_filter3_number_list:
                    job_filter3_number_list.append(job_filter3_num.strip())
            else:
                if one_job.strip() not in job_filter3_number_list:
                    job_filter3_number_list.append(one_job.strip())
        # print(job_filter3_number_list)

        job_filter4_years_list = []
        for one_ in job_filter3_number_list:
            # 去除任职信息末尾的任职年份区间，并更新任职时间如（ 2000年 — 2002年 ）
            # print(one_)
            # 更新准确的任职区间
            begin_time_pre = re.findall(
                r'(\d*年|\d*年 \d*月|\d*年\d*月|\d+.\d*|\d*|\d*年 起|\d* 起|\d*年 \d*月 \d*日) (—|－|-|——|至|，).*?', one_)
            if begin_time_pre:
                begin_time = begin_time_pre[0][0]
                # print(begin_time)
            else:
                begin_time_pre = re.findall(
                    r'(\d*年|\d*年 \d*月|\d*年\d*月|\d*.\d*|\d*|\d*年 起|\d* 起|\d*年 \d*月 \d*日) (—|－|-|——|至|，).*?', exp)
                if begin_time_pre:
                    begin_time = begin_time_pre[0][0]
                else:
                    begin_time = ''
            end_time_pre = re.findall(r'.*(—|－|-|——|至) (\d*年 \d*月|\d*年|\d*.\d*|\d*)', one_)
            if end_time_pre:
                end_time = end_time_pre[0][1]
                # print(end_time)
            else:
                end_time_pre = re.findall(r'.*(—|－|-|——|至) (\d*年 \d*月|\d*年|\d*.\d*|\d*)', exp)
                if end_time_pre:
                    end_time = end_time_pre[0][1]
                else:
                    end_time = ''
            remove1 = re.findall(r'.*(\（.*\d*年.* \）).*', one_)
            # print(remove1)
            if remove1:
                job_filter4_years = one_.replace(remove1[0], '')
                if job_filter4_years.strip() not in job_filter4_years_list:
                    # job_filter4_years_list.append(job_filter4_years.strip())
                    info = job_filter4_years.strip()
                else:
                    info = ''

            else:
                if one_ not in job_filter4_years_list:
                    job_filter4_years_list.append(one_.strip())
                    info = one_
                else:
                    info = ''
        # print(job_filter4_years_list)
# 弃用下面的循环，为了将类似‘教授级 高级 工程师（2000年—2002年）、主任（2002年—2006年）’
# 的信息上的任职年份起止时间点和特定的职称逐一对应上
        # for info in job_filter4_years_list:
            # job = []
            # institution = []
            if info:
                units = info.split(' ')
                # print(units)
                if len(units) >= 4:
                    unit_1 = units[-1]
                    unit_2 = units[-2]
                    unit_3 = units[-3]
                    unit_4 = units[-4]
                    unit_21 = unit_2 + ' ' + unit_1
                    unit_321 = unit_3 + ' ' + unit_2 + ' ' + unit_1
                    unit_4321 = unit_4 + ' ' + unit_3 + ' ' + unit_2 + ' ' + unit_1
                    if unit_4321 in job_list:
                        # print('职称_职位：{}'.format(unit_4321))
                        job = unit_4321
                        unit_institution = units[: -4]
                        institution = ' '.join(unit_institution)
                        # print('所在单位_机构：{}'.format(institution))
                    elif unit_321 in job_list:
                        # print('职称_职位：{}'.format(unit_1))
                        job = unit_321
                        unit_institution = units[: -3]
                        institution = ' '.join(unit_institution)
                        # print('所在单位_机构：{}'.format(institution))
                    elif unit_21 in job_list:
                        # print('职称_职位：{}'.format(unit_1))
                        job = unit_21
                        unit_institution = units[: -2]
                        institution = ' '.join(unit_institution)
                        # print('所在单位_机构：{}'.format(institution))
                    else:
                        if unit_1 in job_list:
                            job = unit_1
                            unit_institution = units[: -1]
                            institution = ' '.join(unit_institution)
                        else:
                            job = ''
                            unit_institution = units
                            institution = ' '.join(unit_institution)
                        # print('所在单位_机构：{}'.format(institution))
                elif len(units) == 3:
                    unit_1 = units[-1]
                    unit_2 = units[-2]
                    unit_3 = units[-3]
                    # print(unit_1, unit_2, unit_3)
                    unit_21 = unit_2 + ' ' + unit_1
                    unit_321 = unit_3 + ' ' + unit_2 + ' ' + unit_1
                    if unit_321 in job_list:
                        # print('职称_职位：{}'.format(unit_1))
                        job = unit_321
                        institution = ''
                        # print('所在单位_机构：{}'.format(institution))
                    elif unit_21 in job_list:
                        # print('职称_职位：{}'.format(unit_1))
                        job = unit_21
                        unit_institution = units[: -2]
                        institution = ' '.join(unit_institution)
                        # print('所在单位_机构：{}'.format(institution))
                    else:
                        if unit_1 in job_list:
                            job = unit_1
                            institution = ''
                        else:
                            job = ''
                            unit_institution = units
                            institution = ' '.join(unit_institution)
                        # print('所在单位_机构：{}'.format(institution))
                elif len(units) == 2:
                    unit_1 = units[-1]
                    unit_2 = units[-2]
                    # print(unit_1, unit_2)
                    unit_21 = unit_2 + ' ' + unit_1
                    if unit_21 in job_list:
                        # print('职称_职位：{}'.format(unit_1))
                        job = unit_21
                        institution = ''
                        # print('所在单位_机构：{}'.format(institution))
                    else:
                        if unit_1 in job_list:
                            job = unit_1
                            unit_institution = units[: -1]
                            institution = ' '.join(unit_institution)
                        else:
                            job = ''
                            unit_institution = units
                            institution = ' '.join(unit_institution)
                        # print('所在单位_机构：{}'.format(institution))
                else:
                    unit_1 = units[-1]
                    if unit_1 in job_list:
                        job = unit_1
                        institution = ''
                    else:
                        job = ''
                        unit_institution = units
                        institution = ' '.join(unit_institution)
                # print("{"+"\'起始时间\':\'{}\',\'终止时间\':\'{}\',\'所在单位\':\'{}\',\'职称\':\'{}\'".format(begin_time, end_time, institution, job)+"}")
                time_unit_job = str("{"+"\'起始时间\':\'{}\',\'终止时间\':\'{}\',\'所在单位\':\'{}\',\'职称\':\'{}\'".format(begin_time.replace(" ", ""), end_time.replace(" ", ""), institution.replace(" ", ""), job.replace(" ", ""))+"}")
                time_unit_job_list.append(time_unit_job)
    return time_unit_job_list


def Attribution_value_Extraction_5_2():
    global job_list
    # work_experience =[]
    # global resume_txt
    # 读入简历文件和任职列表
    info1_name=input('请输入想要查找的院士姓名：')
    info2_engine=input('请选择并输入本次搜索的数据来源（百度百科，搜狗百科，360百科，互动百科）：')

    job_list, resume_txt = read_file('title_list.py', info1_name+'_'+info2_engine+'_split_unattributed.txt')

    flag_store = False
    for line in resume_txt:
        if line == '人物 经历 ：\n':
            # (开始使用字典遍历的标志，将职称内容读到job_list中）
            flag_store = True     # (开始使用字典遍历的标志，将职称内容读到job_list中）

        if flag_store:
            personal_experience.append(line)

        if line == '社会 任职 ：\n':
            # (结束字典遍历的标志，表示职称内容已经全部读完到job_list中)
            flag_store = False
    # print(personal_experience)
    # 处理人物经历，获取工作经历
    work_experience = process_personal_experience(personal_experience)

    # 处理工作经历，获取任职多信息列表
    time_unit_job_list = process_work_experience(work_experience)
    # print(time_unit_job_list)
    with open(info1_name+'_'+info2_engine+'_Extract_Job_Attribution_Value.txt',mode='w',encoding='utf-8') as Extraction:
        for one in time_unit_job_list:
            Extraction.write(one+'\n')
    print('信息抽取完毕，请核实抽取信息！')







