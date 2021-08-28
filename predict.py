# -*- coding: utf-8 -*-
# @Time    : 2021/08/15
# @Author  : Z.J
# @File    : predict.py
# @Software: vs code
from tensorflow.keras.preprocessing import image
import numpy as np
import os
import tensorflow as tf
import train
from tensorflow.python.training.checkpoint_management import latest_checkpoint

path = "./data/scidb_mineral_data/test/"  # 预测图片的路径
path_save_model = './model/data_scidb_model_InceptionResNetV2.h5'  # 保存的模型的路径
checkpoint_path = 'ckpt/transfer_{epoch:02d}-{val_acc:.2f}.h5'  # 检查点路径
checkpoint_root = os.path.dirname(checkpoint_path)  # 检查点文件根目录
image_size = 150  # 图片格式(150,150)
# 建立标签字典，便于输出结果
label_dict = {
    '0': '十字石',
    '1': '斜长石',
    '2': '普通辉石',
    '3': '橄榄石',
    '4': '石榴子石',
    '5': '红柱石',
    '6': '角闪石',
    '7': '鲕粒'
}


def loadWeights():
    """
    读取保存的权重数据，需先构建网络结构一致的新模型
    """
    base_model = train.PowerTransferMode()
    model = base_model.InceptionV3_model(
        nb_classes=5,
        img_rows=image_size,
        img_cols=image_size,
        is_plot_model=False
    )
    # 从检查点恢复权重
    saved_weights = './ckpt/transfer_50-1.00.h5'
    # latest_weights = tf.train.latest_checkpoint(checkpoint_root)  只对ckpt格式文件有用！
    model.load_weights(saved_weights)
    return model


def loadModel():
    """读取全部模型数据"""
    model = tf.keras.models.load_model('model/data_scidb_model_InceptionResNetV2.h5')
    return model

def predict(dir,model):
    for img_name in os.listdir(dir):
        img_path = dir+img_name
        img = image.load_img(img_path, target_size=(299, 299))
        # 保持输入格式一致
        x = image.img_to_array(img) / 255
        # 变为四维数据
        x = np.expand_dims(x, axis=0)
        # 预测
        result = model.predict(x)
        # 返回最大概率值的索引，类型是张量
        index = tf.argmax(result, axis=1)
        print(str(int(index)))
        print(img_name, '======================>', label_dict[str(int(index))])

if __name__ == '__main__':
    model = loadModel()
    print(model.summary())
    for data_dir in os.listdir(path):
        dataPath = path + data_dir + '/'
        predict(dataPath,model)
