import numpy as np
import cv2
import os
import keras

class Dataset:
    def __init__(self):
        self.path = "../data/normalized-imgs/"
        # 数据集
        self.images = []
        self.labels = []
        # 训练集
        self.train_images = None
        self.train_labels = None
        # 验证集
        self.valid_images = None
        self.valid_labels = None
        # 测试集
        self.test_images = None
        self.test_labels = None
        # 类别数
        self.class_num = 0
        # 输入维度
        self.input_shape = None

    def my_atoi(self, num_string):
        if num_string == '':
            return 0
        else:
            try:
                f = float(num_string)
                i = int(f)
            except:
                return 0
            else:
                return i

    def load_dataset(self):
        img_set_id_list = []
        for img_set_id in os.listdir(self.path):
            img_set_id_list.append(img_set_id)

        for person in img_set_id_list:
            current_path = os.path.join(self.path, person)
            for fpath, dirname, fnames in os.walk(current_path):
                for fname in fnames:
                    if fname.endswith('.jpg'):
                        face_path = fpath + '/' + fname
                        image = cv2.imread(face_path)
                        self.images.append(image)
                        #self.labels.append(int(person[-1]))
                        self.labels.append(self.my_atoi(person))
        self.images = np.array(self.images)
        self.labels = np.array(self.labels)
        # print(self.images.shape)
        self.class_num = max(self.labels) + 1
        # print(self.labels)

        img_rows = self.images.shape[1]
        img_cols = self.images.shape[2]
        img_channel = self.images.shape[3]
        self.input_shape = (img_rows, img_cols, img_channel)

    def split_dataset(self, data, test_ratio, valid_ratio):
        # 设置随机数种子，保证每次生成的结果都是一样的
        np.random.seed(50)
        # permutation随机生成0-len(data)随机序列
        random_list = np.random.permutation(len(data))

        test_set_size = int(len(data) * test_ratio)
        valid_set_size = int(len(data) * valid_ratio)
        test_set = random_list[:test_set_size]
        valid_set = random_list[test_set_size:test_set_size + valid_set_size]
        train_set = random_list[test_set_size + valid_set_size:]
        return data[train_set], data[test_set], data[valid_set]

    def prepare_dataset(self):
        # 按照交叉验证的原则区分数据集
        train_labels, test_labels, valid_labels = self.split_dataset(self.labels, 0.1, 0.1)
        train_images, test_images, valid_images = self.split_dataset(self.images, 0.1, 0.1)
        # 将标签数据转换为one-hot编码向量
        train_labels = keras.utils.to_categorical(train_labels, self.class_num)
        valid_labels = keras.utils.to_categorical(valid_labels, self.class_num)
        test_labels = keras.utils.to_categorical(test_labels, self.class_num)

        train_images = train_images.astype('float32')
        valid_images = valid_images.astype('float32')
        test_images = test_images.astype('float32')

        # 对样本进行归一化
        train_images /= 255
        valid_images /= 255
        test_images /= 255

        self.train_images = train_images
        self.valid_images = valid_images
        self.test_images = test_images
        self.train_labels = train_labels
        self.valid_labels = valid_labels
        self.test_labels = test_labels


if __name__ == '__main__':
    dataset = Dataset()
    dataset.load_dataset()
    print(dataset.labels)
    # dataset.prepare_dataset()
