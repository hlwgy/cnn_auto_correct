# %%
import tensorflow as tf
import numpy as np
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
import pathlib
import cv2

# %% 构建模型
def create_model():
    model = Sequential([
        layers.experimental.preprocessing.Rescaling(1./255, input_shape=(24, 24, 1)),
        layers.Conv2D(24, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(32, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Dropout(0.2),
        layers.Flatten(),
        layers.Dense(96, activation='relu'),
        layers.Dense(15)]
    )
    
    model.compile(optimizer='adam',
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=['accuracy'])

    return model

# %% 训练数据
def train():
    # 统计文件夹下的所有图片数量
    data_dir = pathlib.Path('dataset')
    batch_size = 64
    img_width = 24
    img_height = 24

    # 从文件夹下读取图片，生成数据集
    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset='training',
        seed=123,
        color_mode="grayscale",
        image_size=(img_height, img_width),
        batch_size=batch_size
    )

    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="validation",
        seed=123,
        color_mode="grayscale",
        image_size=(img_height, img_width),
        batch_size=batch_size
    )

    # 数据集的分类，对应dataset文件夹下有多少图片分类
    class_names = train_ds.class_names
    # 保存数据集分类
    np.save("checkpoint/class_name.npy", class_names)

    # 数据集缓存处理
    AUTOTUNE = tf.data.experimental.AUTOTUNE
    train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
    # 创建模型
    model = create_model()
    # 训练模型，epochs=10，所有数据集训练10遍
    model.fit(train_ds,validation_data=val_ds,epochs=20)
    # 保存训练后的权重
    model.save_weights('checkpoint/char_checkpoint')

# %% 预测
def predict(model, imgs, class_name):
    label_dict = {0: '0', 1: '1', 2: '2', 3: '3', 4: '4', 5: '5', 6: '6', 7: '7', 8: '8', 9: '9', 10: '=', 11: '+', 12: '-', 13: '×', 14: '÷'}
    # 预测图片，获取预测值
    predicts = model.predict(imgs) 
    results = [] # 保存结果的数组
    for predict in predicts: #遍历每一个预测结果
        index = np.argmax(predict) # 寻找最大值
        result = class_name[index] # 取出字符
        results.append(label_dict[int(result)])
    return results


# %% 
if __name__ == '__main__':

    train()
    
    # model = create_model()
    # # 加载前期训练好的权重
    # model.load_weights('checkpoint/char_checkpoint')
    # # 读出图片分类
    # class_name = np.load('checkpoint/class_name.npy')
    # print(class_name)
    # img1=cv2.imread('img1.png',0) 
    # img2=cv2.imread('img2.png',0) 
    # img3=cv2.imread('img3.png',0)
    # img4=cv2.imread('img4.png',0)
    # img5=cv2.imread('img5.png',0)
    # img6=cv2.imread('img6.png',0)
    # imgs = np.array([img1,img2,img3,img4,img5,img6])
    # results = predict(model, imgs, class_name)
    # print(results)


# %%
