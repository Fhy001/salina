#!/usr/bin/env python
# coding: utf-8

# In[1]:

#解压上传到AI Studio的压缩包
get_ipython().system('ls /home/aistudio/data')
import os
import zipfile
os.chdir('/home/aistudio/data/data100510')
extracting = zipfile.ZipFile('salt pan.zip')
extracting.extractall()


# In[2]:

#对数据集生成固定格式的列表，格式为：图片的路径 <Tab> 图片类别的标签
import json
import os

def create_data_list(data_root_path):
    with open(data_root_path + "test.list", 'w') as f:
        pass
    with open(data_root_path + "train.list", 'w') as f:
        pass
    # 所有类别的信息
    class_detail = []
    # 获取所有类别
    class_dirs = os.listdir(data_root_path)
    # 类别标签
    class_label = 0
    # 获取总类别的名称
    father_paths = data_root_path.split('/')
    while True:
        if father_paths[len(father_paths) - 1] == '':
            del father_paths[len(father_paths) - 1]
        else:
            break
    father_path = father_paths[len(father_paths) - 1]

    all_class_images = 0
    other_file = 0
    # 读取每个类别
    for class_dir in class_dirs:
        if class_dir == 'test.list' or class_dir == "train.list" or class_dir == 'readme.json':
            other_file += 1
            continue
        print('正在读取类别：%s' % class_dir)
        # 每个类别的信息
        class_detail_list = {}
        test_sum = 0
        trainer_sum = 0
        # 统计每个类别有多少张图片
        class_sum = 0
        # 获取类别路径
        path = data_root_path + "/" + class_dir
        # 获取所有图片
        img_paths = os.listdir(path)
        for img_path in img_paths:
            # 每张图片的路径
            name_path = class_dir + '/' + img_path
            # 如果不存在这个文件夹,就创建
            if not os.path.exists(data_root_path):
                os.makedirs(data_root_path)
            # 划分训练集和测试集，各个类别中每隔十张选取一张作为测试集，并将数据集生成固定格式列表。
            if class_sum % 10 == 0:
                test_sum += 1
                with open(data_root_path + "test.list", 'a') as f:
                    f.write(name_path + "\t%d" % class_label + "\n")
            else:
                trainer_sum += 1
                with open(data_root_path + "train.list", 'a') as f:
                    f.write(name_path + "\t%d" % class_label + "\n")
            class_sum += 1
            all_class_images += 1
        # 说明的json文件的class_detail数据
        class_detail_list['class_name'] = class_dir
        class_detail_list['class_label'] = class_label
        class_detail_list['class_test_images'] = test_sum
        class_detail_list['class_trainer_images'] = trainer_sum
        class_detail.append(class_detail_list)
        class_label += 1
    # 获取类别数量
    all_class_sum = len(class_dirs) - other_file
    # 说明的json文件信息
    readjson = {}
    readjson['all_class_name'] = father_path
    readjson['all_class_sum'] = all_class_sum
    readjson['all_class_images'] = all_class_images
    readjson['class_detail'] = class_detail
    jsons = json.dumps(readjson, sort_keys=True, indent=4, separators=(',', ': '))
    with open(data_root_path + "readme.json", 'w') as f:
        f.write(jsons)
    print('图像列表已生成')


# In[3]:


#生成图像的列表
if __name__ == '__main__':
    # 把生产的数据列表都放在自己的总类别文件夹中
    data_root_path = "salt pan/"
    create_data_list(data_root_path)



# In[6]:


#架构模型时需要用到的组件。
#包括BN层、常规卷积层、xavier初始化方法
#以及改进后的inception结构（salt-inception）和深度可分离卷积结构
import paddle.fluid as fluid
import numpy as np
import time
import math
import paddle
import paddle.fluid as fluid
import codecs
import logging
from paddle.fluid.initializer import MSRA
from paddle.fluid.initializer import Uniform
from paddle.fluid.param_attr import ParamAttr
from PIL import Image
from PIL import ImageEnhance
import os
import random
from multiprocessing import cpu_count
import numpy as np
import paddle
from PIL import Image 


#BN层
def conv_bn_layer(input, filter_size, num_filters, stride,
                  padding, channels=None, num_groups=1, act='relu', use_cudnn=True):
    conv = fluid.layers.conv2d(input=input,
                               num_filters=num_filters,
                               filter_size=filter_size,
                               stride=stride,
                               padding=padding,
                               groups=num_groups,
                               act=None,
                               use_cudnn=use_cudnn,
                               bias_attr=False)

    return fluid.layers.batch_norm(input=conv, act=act)



#常规卷积层
def conv_layer(
           input,
           num_filters,
           filter_size,
           stride=1,
           groups=1,
           act=None,
           name=None):
    channels = input.shape[1]
    stdv = (3.0 / (filter_size**2 * channels))**0.5
    param_attr = ParamAttr(
        initializer=fluid.initializer.Uniform(-stdv, stdv),
        name=name + "_weights")
    conv = fluid.layers.conv2d(
        input=input,
        num_filters=num_filters,
        filter_size=filter_size,
        stride=stride,
        padding=(filter_size - 1) // 2,
        groups=groups,
        act=act,
        param_attr=param_attr,
        bias_attr=False,
        name=name)
    return conv


def xavier(self, channels, filter_size, name):
    stdv = (3.0 / (filter_size**2 * channels))**0.5
    param_attr = ParamAttr(
        initializer=fluid.initializer.Uniform(-stdv, stdv),
        name=name + "_weights")
    
    return param_attr


#改进后的inception结构（salt-inception）
def inception(
          input,
          channels,
          filter1,
          filter3R,
          filter3,
          filter5R,
          filter5,
          filter7R,
          filter7,
          proj,
          name=None):
    conv1 = conv_layer(
        input=input,
        num_filters=filter1,
        filter_size=1,
        stride=1,
        act=None,
        name="inception_" + name + "_1x1")
        
    #由一个1*1卷积层和一个3*3卷积层组成salt-inception的支路
    conv3r = conv_layer(
        input=input,
        num_filters=filter3R,
        filter_size=1,
        stride=1,
        act=None,
        name="inception_" + name + "_3x3_reduce")
    conv3 = conv_layer(
        input=conv3r,
        num_filters=filter3,
        filter_size=3,
        stride=1,
        act=None,
        name="inception_" + name + "_3x3")
        
    #由一个1*1卷积层和2个3*3卷积层组成salt-inception的支路，由2个3*3卷积层代替5*5卷积层        
    conv5r = conv_layer(
        input=input,
        num_filters=filter5R,
        filter_size=1,
        stride=1,
        act=None,
        name="inception_" + name + "_5x5_reduce")
    conv5 = conv_layer(
        input=conv5r,
        num_filters=filter5R,
        filter_size=3,
        stride=1,
        act=None,
        name="inception_" + name + "_5x5")
    conv5 = conv_layer(
        input=conv5,
        num_filters=filter5,
        filter_size=3,
        stride=1,
        act=None,
        name="inception_" + name + "_5x5_2")

    #由一个1*1卷积层和3个3*3卷积层组成salt-inception的支路，由3个3*3卷积层代替7*7卷积层               
    conv7r = conv_layer(
        input=input,
        num_filters=filter7R,
        filter_size=1,
        stride=1,
        act=None,
        name="inception_" + name + "_7x7_reduce")
    conv7 = conv_layer(
        input=conv7r,
        num_filters=filter7R,
        filter_size=3,
        stride=1,
        act=None,
        name="inception_" + name + "_7x7")
    conv7 = conv_layer(
        input=conv7,
        num_filters=filter7R,
        filter_size=3,
        stride=1,
        act=None,
        name="inception_" + name + "_7x7_2")
    conv7 = conv_layer(
        input=conv7,
        num_filters=filter7,
        filter_size=3,
        stride=1,
        act=None,
        name="inception_" + name + "_7x7_3")
        
    pool = fluid.layers.pool2d(
        input=input,
        pool_size=3,
        pool_stride=1,
        pool_padding=1,
        pool_type='max')
    convprj = fluid.layers.conv2d(
        input=pool,
        filter_size=1,
        num_filters=proj,
        stride=1,
        padding=0,
        name="inception_" + name + "_3x3_proj",
        param_attr=ParamAttr(
            name="inception_" + name + "_3x3_proj_weights"),
        bias_attr=False)
    cat = fluid.layers.concat(input=[conv1, conv3, conv5, conv7, convprj], axis=1)
    cat = fluid.layers.relu(cat)
    return cat


#深度可分离卷积
def depthwise_separable(input, num_filters1, num_filters2, num_groups, stride, scale):
    depthwise_conv = conv_bn_layer(input=input,
                                   filter_size=3,
                                   num_filters=int(num_filters1 * scale),
                                   stride=stride,
                                   padding=1,
                                   num_groups=int(num_groups * scale),
                                   use_cudnn=False)

    pointwise_conv = conv_bn_layer(input=depthwise_conv,
                                   filter_size=1,
                                   num_filters=int(num_filters2 * scale),
                                   stride=1,
                                   padding=0)
    return pointwise_conv


# In[7]:


#salt-ConvNet卷积神经网络的结构
def net(input, class_dim, scale=1.0):
    # 224x224
    input = conv_bn_layer(input=input,
                          filter_size=3,
                          channels=3,
                          num_filters=int(32 * scale),
                          stride=2,
                          padding=1)
    

    # 112x112
    input = depthwise_separable(input=input,
                                num_filters1=32,
                                num_filters2=64,
                                num_groups=32,
                                stride=1,
                                scale=scale)
    
    input = depthwise_separable(input=input,
                                num_filters1=64,
                                num_filters2=128,
                                num_groups=64,
                                stride=2,
                                scale=scale)
                                
   
    
    
     # 56x56
    input = depthwise_separable(input=input,
                                num_filters1=128,
                                num_filters2=128,
                                num_groups=128,
                                stride=1,
                                scale=scale)

    input = depthwise_separable(input=input,
                                num_filters1=128,
                                num_filters2=256,
                                num_groups=128,
                                stride=2,
                                scale=scale)
    
    # 28x28
    input = depthwise_separable(input=input,
                                num_filters1=256,
                                num_filters2=256,
                                num_groups=256,
                                stride=1,
                                scale=scale)

    input = depthwise_separable(input=input,
                                num_filters1=256,
                                num_filters2=512,
                                num_groups=256,
                                stride=2,
                                scale=scale)
    
    # 14x14
    input = inception(input, 480, 192, 96, 208, 16, 24, 16, 24,64, "input")
    input1 = inception(input, 512, 160, 112, 224, 24, 32, 24, 32, 64, "input1")
    input1 = paddle.fluid.layers.concat(input=[input,input1], axis=1)
    input2 = inception(input1, 512, 128, 128, 256,  24, 32, 24, 32, 64, "input2")
    input2 = paddle.fluid.layers.concat(input=[input,input1,input2], axis=1)
    input3 = inception(input2, 512, 128, 128, 256,  24, 32, 24, 32, 64, "input3")
    input3 = paddle.fluid.layers.concat(input=[input,input1,input2,input3], axis=1)
  
   
    input = depthwise_separable(input=input3,
                                num_filters1=512,
                                num_filters2=1024,
                                num_groups=512,
                                stride=2,
                                scale=scale)
   
    # 7x7
    input = depthwise_separable(input=input,
                                num_filters1=1024,
                                num_filters2=1024,
                                num_groups=1024,
                                stride=1,
                                scale=scale)

    feature = fluid.layers.pool2d(input=input,
                                  pool_size=0,
                                  pool_stride=1,
                                  pool_type='avg',
                                  global_pooling=True)

    net = fluid.layers.fc(input=feature,
                          size=class_dim,
                          act='softmax')
    return net


# In[8]:


# 训练图片的预处理
import random 
from PIL import Image
def train_mapper(sample):
    img_path, label, crop_size, resize_size = sample
    img = Image.open(img_path)
    # 统一图片大小
    img = img.resize((resize_size, resize_size), Image.ANTIALIAS)
    # 随机水平翻转
    r1 = random.random()
    if r1 > 0.5:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
    # 随机垂直翻转
    r2 = random.random()
    if r2 > 0.5:
        img = img.transpose(Image.FLIP_TOP_BOTTOM)
    # 随机角度翻转
    r3 = random.randint(-3, 3)
    img = img.rotate(r3, expand=False)
    # 随机裁剪
    r4 = random.randint(0, int(resize_size - crop_size))
    r5 = random.randint(0, int(resize_size - crop_size))
    box = (r4, r5, r4 + crop_size, r5 + crop_size)
    img = img.crop(box)
    # 把图片转换成numpy值
    img = np.array(img).astype(np.float32)
    # 转换成CHW
    img = img.transpose((2, 0, 1))
    # 转换成BGR
    img = img[(2, 1, 0), :, :] / 255.0
    return img, int(label)



# 获取训练的reader
def train_r(train_list_path, crop_size, resize_size):
    father_path = os.path.dirname(train_list_path)

    def reader():
        with open(train_list_path, 'r') as f:
            lines = f.readlines()
            # 打乱图像列表
            np.random.shuffle(lines)
            # 开始获取每张图像和标签
            for line in lines:
                img, label = line.split('\t')
                img = os.path.join(father_path, img)
                yield img, label, crop_size, resize_size

    return paddle.reader.xmap_readers(train_mapper, reader, cpu_count(), 102400)


# 测试图片的预处理
def test_mapper(sample):
    img, label, crop_size = sample
    img = Image.open(img)
    # 统一图像大小
    img = img.resize((crop_size, crop_size), Image.ANTIALIAS)
    # 转换成numpy值
    img = np.array(img).astype(np.float32)
    # 转换成CHW
    img = img.transpose((2, 0, 1))
    # 转换成BGR
    img = img[(2, 1, 0), :, :] / 255.0
    return img, int(label)


# 测试的图片reader
def test_r(test_list_path, crop_size):
    father_path = os.path.dirname(test_list_path)

    def reader():
        with open(test_list_path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                img, label = line.split('\t')
                img = os.path.join(father_path, img)
                yield img, label, crop_size

    return paddle.reader.xmap_readers(test_mapper, reader, cpu_count(), 1024)




# In[9]:


#进行模型的训练
import os
import shutil
import paddle as paddle
import paddle.fluid as fluid
from multiprocessing import cpu_count
import numpy as np
import os

crop_size = 224
resize_size = 250
# 获取划分好的训练集和数据集，注意传入的.list文件的路径是否正确
try:
    os.chdir('/home/aistudio/data/data100510')
except:
    pass
train_reader = paddle.batch(reader=train_r('salt pan/train.list', crop_size, resize_size), batch_size=32)
test_reader = paddle.batch(reader=test_r('salt pan/test.list', crop_size), batch_size=32)


# 定义网络
def vgg_test_net(image, type_size):  
    def conv_block(ipt, num_filter, groups, dropouts):
        return fluid.nets.img_conv_group(
            input=ipt, # 具有[N，C，H，W]格式的输入图像
            pool_size=2,
            pool_stride=2,
            conv_num_filter=[num_filter] * groups, # 过滤器个数
            conv_filter_size=3, # 过滤器大小
            conv_act='relu',
            conv_with_batchnorm=True, # 表示在 Conv2d Layer 之后是否使用 BatchNorm
            conv_batchnorm_drop_rate=dropouts,# 表示 BatchNorm 之后的 Dropout Layer 的丢弃概率
            pool_type='max') # 最大池化

    conv1 = conv_block(image, 64, 2, [0.0, 0])
    conv2 = conv_block(conv1, 128, 2, [0.0, 0])
    conv3 = conv_block(conv2, 256, 3, [0.0, 0.0, 0])
    conv4 = conv_block(conv3, 512, 3, [0.0, 0.0, 0])
    conv5 = conv_block(conv4, 512, 3, [0.0, 0.0, 0])

    drop = fluid.layers.dropout(x=conv2, dropout_prob=0.5)
    fc1 = fluid.layers.fc(input=drop, size=512, act=None)
    
    bn = fluid.layers.batch_norm(input=fc1, act='relu')
    drop2 = fluid.layers.dropout(x=bn, dropout_prob=0.0)
    fc2 = fluid.layers.fc(input=drop2, size=512, act=None)
    # predict = fluid.layers.fc(input=fc1, size=type_size, act='softmax')
    predict = fluid.layers.fc(input=fc1, size=type_size)
    return predict



# 定义输入层
paddle.enable_static()
image = fluid.layers.data(name='image', shape=[3, crop_size, crop_size], dtype='float32')
label = fluid.layers.data(name='label', shape=[1], dtype='int64')


# 调用构建好的网络模型salt-ConvNet,并对11种盐田进行分类。
predict = net(image, 11)#疑似出错！！！！！！！！！！！

# 获取损失函数(交叉熵函数)和准确率函数
cost = fluid.layers.cross_entropy(input=predict, label=label)
avg_cost = fluid.layers.mean(cost)
acc = fluid.layers.accuracy(input=predict, label=label)

# 获取训练和测试程序
test_program = fluid.default_main_program().clone(for_test=True)

# 定义优化方法为Adam
optimizer = fluid.optimizer.AdamOptimizer(learning_rate=1e-3,
                                          regularization=fluid.regularizer.L2DecayRegularizer(1e-4))
opts = optimizer.minimize(avg_cost)





# 定义一个使用GPU的执行器
place = fluid.CUDAPlace(0)
# place = fluid.CPUPlace()
exe = fluid.Executor(place)
# 进行参数初始化
exe.run(fluid.default_startup_program())

# 定义输入数据维度
feeder = fluid.DataFeeder(place=place, feed_list=[image, label])




# In[10]:


# 训练30轮
for pass_id in range(30):
    # 进行训练
    for batch_id, data in enumerate(train_reader()):
        train_cost, train_acc = exe.run(program=fluid.default_main_program(),
                                        feed=feeder.feed(data),
                                        fetch_list=[avg_cost, acc])

        # 每100个batch打印一次信息
        if batch_id % 10 == 0:
            print('Pass:%d, Batch:%d, Cost:%0.5f, Accuracy:%0.5f' %
                  (pass_id, batch_id, train_cost[0], train_acc[0]))

    # 进行测试
    test_accs = []
    test_costs = []
    for batch_id, data in enumerate(test_reader()):
        test_cost, test_acc = exe.run(program=test_program,
                                      feed=feeder.feed(data),
                                      fetch_list=[avg_cost, acc])
        test_accs.append(test_acc[0])
        test_costs.append(test_cost[0])
    # 求测试结果的平均值
    test_cost = (sum(test_costs) / len(test_costs))
    test_acc = (sum(test_accs) / len(test_accs))
    print('Test:%d, Cost:%0.5f, Accuracy:%0.5f' % (pass_id, test_cost, test_acc))
    
    # 保存预测模型
    save_path = 'infer_model/'
    # 删除旧的模型文件
    shutil.rmtree(save_path, ignore_errors=True)
    # 创建保持模型文件目录
    os.makedirs(save_path)
    # 保存预测模型
    fluid.io.save_inference_model(save_path, feeded_var_names=[image.name], target_vars=[predict], executor=exe)


# In[11]:


import cv2
from PIL import Image
import random 
# data = cv2.imread("/home/aistudio/data/data100510/salt pan/Salt5_level16/8 (38).jpg")
img = Image.open("/home/aistudio/data/data100510/salt pan/Salt5_level16/8 (38).jpg")
print(img)
# 统一图片大小
img = img.resize((resize_size, resize_size), Image.ANTIALIAS)
# 随机水平翻转
r1 = random.random()
if r1 > 0.5:
    img = img.transpose(Image.FLIP_LEFT_RIGHT)
# 随机垂直翻转
r2 = random.random()
if r2 > 0.5:
    img = img.transpose(Image.FLIP_TOP_BOTTOM)
# 随机角度翻转
r3 = random.randint(-3, 3)
img = img.rotate(r3, expand=False)
# 随机裁剪
r4 = random.randint(0, int(resize_size - crop_size))
r5 = random.randint(0, int(resize_size - crop_size))
box = (r4, r5, r4 + crop_size, r5 + crop_size)
img = img.crop(box)
# 把图片转换成numpy值
img = np.array(img).astype(np.float32)
        # 转换成CHW
img = img.transpose((2, 0, 1))
        # 转换成BGR
img = img[(2, 1, 0), :, :] / 255.0
print(img)


# 加载好训练的模型，对某一特定输入的盐田图片进行测试

# In[12]:


import paddle.fluid as fluid
from PIL import Image
import numpy as np

# 创建执行器
place = fluid.CPUPlace()
exe = fluid.Executor(place)
exe.run(fluid.default_startup_program())

# 保存预测模型路径
save_path = 'infer_model/'
# 从模型中获取预测程序、输入数据名称列表、分类器
[infer_program, feeded_var_names, target_var] = fluid.io.load_inference_model(dirname=save_path, executor=exe)

# 预处理图片
def load_image(file):
    img = Image.open(file)
    # 统一图像大小
    img = img.resize((224, 224), Image.ANTIALIAS)
    # 转换成numpy值
    img = np.array(img).astype(np.float32)
    # 转换成CHW
    img = img.transpose((2, 0, 1))
    # 转换成BGR
    img = img[(2, 1, 0), :, :] / 255.0
    img = np.expand_dims(img, axis=0)
    return img

# 获取图片数据，可以更改imgpath的值，实现对不同的图片进行预测。
imgpath = "/home/aistudio/salt3_level12.jpg"
img = load_image(imgpath)

# 执行预测
result = exe.run(program=infer_program,
                 feed={feeded_var_names[0]: img},
                 fetch_list=target_var)

# 显示图片并输出结果最大的label
lab = np.argsort(result)[0][0][-1]

names = ['Salt3_level12', 'Salt4_level12', 'Salt4_level14','Salt5_level11', 'Salt5_level12','Salt5_level13','Salt5_level14',
'Salt5_level15','Salt5_level16','Salt7_level12','Salt7_level16']

print('图片路径为：%s,  预测结果标签为：%d， 预测结果为：%s， 概率为：%f' % (imgpath, lab, names[lab], result[0][0][lab]))
