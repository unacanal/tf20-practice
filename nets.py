import tensorflow as tf
from tensorflow.keras.layers import Conv2D, BatchNormalization, Layer, ReLU

# SRCNN
class SRCNN(tf.keras.Model): # inherit
    def __init__(self):
        super(SRCNN, self).__init__() # 상위 클래스의 init함수를 실행

        self.conv1 = Conv2D(filters=64,
                      kernel_size=(9, 9),
                      padding='same',
                      kernel_initializer=tf.initializers.GlorotUniform())

        self.act1 = ReLU() # Conv2D의 인자로 줘도 됨

        self.conv2 = Conv2D(filters=32,
                       kernel_size=(1, 1),
                       padding='same',
                       kernel_initializer=tf.initializers.GlorotUniform())

        self.act2 = ReLU()

        self.conv3 = Conv2D(filters=1,
                       kernel_size=(5, 5),
                       padding='same',
                       kernel_initializer=tf.initializers.GlorotUniform())
                       # 마지막 레이어에서는 손실이 잃어나면 안돼서 activation function안씀

        # self.trainable_variables

    def call(self, inputs, training=None, mask=None):
        y1 = self.conv1(inputs)
        y2 = self.act1(y1)
        y3 = self.conv2(y2)
        y4 = self.act2(y3)
        y5 = self.conv3(y4)
        y6 = y5 + inputs

        return y6, y5


class ResUnit(Layer):
    def __init__(self):
        super(ResUnit, self).__init__()

        self.c1 = Conv2D(filters=64,
                         kernel_size=(3, 3),
                         padding='same',
                         activation='relu',
                         kernel_initializer=tf.initializers.he_normal())

        self.c2 = Conv2D(filters=64,
                         kernel_size=(3, 3),
                         padding='same',
                         kernel_initializer=tf.initializers.he_normal())

        self.batch_norm = BatchNormalization()

    def call(self, inputs, **kwargs):
        y1 = self.c1(inputs)
        y2 = self.batch_norm(y1)
        y3 = self.c2(y2)
        y4 = y3 + inputs
        return y4


class ComplexModel(tf.keras.Model):
    def __init__(self):
        super(ComplexModel, self).__init__()

