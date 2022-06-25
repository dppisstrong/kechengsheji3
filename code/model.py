import numpy as np
import tensorflow
import keras
from keras.optimizers import SGD

from load_data import Dataset
from keras.preprocessing.image import ImageDataGenerator

class Model:
    def __init__(self):
        self.model = None

    def build_model(self, dataset):
        model = keras.models.Sequential()

        model.add(keras.layers.Convolution2D(32, (3, 3), padding='same',
                                             input_shape=dataset.input_shape,
                                             activation='relu'))

        model.add(keras.layers.Convolution2D(32, (3, 3), activation='relu'))

        model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
        model.add(keras.layers.Dropout(0.25))

        model.add(keras.layers.Convolution2D(64, (3, 3), padding='same', activation='relu'))

        model.add(keras.layers.Convolution2D(64, (3, 3), activation='relu'))

        model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
        model.add(keras.layers.Dropout(0.25))

        model.add(keras.layers.Flatten())
        model.add(keras.layers.Dense(512, activation='relu'))

        model.add(keras.layers.Dropout(0.5))
        model.add(keras.layers.Dense(dataset.class_num, activation='softmax'))

        self.model = model
        self.model.summary()

    def train_model(self, dataset, batch_size=20, epoch=10):
        sgd = SGD(lr=0.0007, decay=1e-6, momentum=0.9, nesterov=True)
        self.model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
        datagen = keras.preprocessing.image.ImageDataGenerator(
            featurewise_center=False,  # 是否使输入数据去中心化（均值为0），
            samplewise_center=False,  # 是否使输入数据的每个样本均值为0
            featurewise_std_normalization=False,  # 是否数据标准化（输入数据除以数据集的标准差）
            samplewise_std_normalization=False,  # 是否将每个样本数据除以自身的标准差
            zca_whitening=False,  # 是否对输入数据施以ZCA白化
            rotation_range=20,  # 数据提升时图片随机转动的角度(范围为0～180)
            width_shift_range=0.2,  # 数据提升时图片水平偏移的幅度（单位为图片宽度的占比，0~1之间的浮点数）
            height_shift_range=0.2,  # 同上，只不过这里是垂直
            horizontal_flip=True,  # 是否进行随机水平翻转
            vertical_flip=False)  # 是否进行随机垂直翻转
        datagen.fit(dataset.train_images)
        self.model.fit_generator(datagen.flow(dataset.train_images, dataset.train_labels,
                                 batch_size=batch_size),
                                 # samples_per_epoch=dataset.train_images.shape[0],
                                 steps_per_epoch=dataset.train_images.shape[0] / batch_size,
                                 epochs=epoch,
                                 validation_data=(dataset.valid_images, dataset.valid_labels))

    def evaluate_model(self, dataset):
        score = self.model.evaluate(dataset.test_images, dataset.test_labels, verbose=1)
        print("%s: %.2f%%" % (self.model.metrics_names[1], score[1] * 100))

    def save_model(self, model_path):
        self.model.save(model_path)

    # def load_model(self, model_path):
    #     self.model = keras.models.load_model(model_path)
    #
    # def face_predict(self, image):
    #     image = np.array(image)
    #     image = image.reshape((1, 64, 64, 3))
    #     image = image.astype('float32')
    #     image /= 255
    #     # 给出输入属于各个类别的概率，我们是二值类别，则该函数会给出输入图像属于0和1的概率各为多少
    #     predict_probability = self.model.predict_proba(image)
    #     # 给出类别预测：0或者1
    #     result = self.model.predict_classes(image)
    #     # 返回类别预测结果
    #     # return max(predict_probability[0]), result[0]
    #     return predict_probability[0], result[0]


if __name__ == '__main__':
    dataset = Dataset()
    dataset.load_dataset()
    dataset.prepare_dataset()
    model = Model()
    model.build_model(dataset)
    model.train_model(dataset)
    model.evaluate_model(dataset)
    model.save_model('../data/model/model.h5')
