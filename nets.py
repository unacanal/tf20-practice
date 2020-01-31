import tensorflow as tf
from tensorflow.keras.layers import Conv2D, BatchNormalization, Layer, ReLU

class ivpl24(tf.keras.Model):
    def __init__(self):
        super(ivpl24, self).__init__()

        self.layer_list = []
        self.layer_list.append(BatchNormalization())

        for i in range(23):
            conv = Conv2D(filters=64,
                          kernel_size=(5,5),
                          padding='same',
                          activation='relu',
                          kernel_initializer=tf.initializers.he_normal())
            self.layer_list.append(conv)

        conv = Conv2D(filters=1,
                      kernel_size=(5, 5),
                      padding='same',
                      kernel_initializer=tf.initializers.he_normal())

        self.layer_list.append(conv)

    def call(self, inputs, training=None, mask=None):
        y = inputs
        for layer in self.layer_list:
            y = layer(y)

        return y + inputs

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

        self.c1 = Conv2D(filters=64,
                         kernel_size=(9, 9),
                         padding='same',
                         activation='relu',
                         kernel_initializer=tf.initializers.he_normal())

        self.c1r1 = ResUnit()
        self.c1r2 = ResUnit()
        self.c1r3 = ResUnit()


        self.c2 = Conv2D(filters=64,
                         kernel_size=(7, 7),
                         padding='same',
                         activation='relu',
                         kernel_initializer=tf.initializers.he_normal())

        self.c2r1 = ResUnit()
        self.c2r2 = ResUnit()
        self.c2r3 = ResUnit()
        self.c2r4 = ResUnit()

        self.c3 = Conv2D(filters=64,
                         kernel_size=(5, 5),
                         padding='same',
                         activation='relu',
                         kernel_initializer=tf.initializers.he_normal())

        self.c3r1 = ResUnit()
        self.c3r2 = ResUnit()
        self.c3r3 = ResUnit()
        self.c3r4 = ResUnit()
        self.c3r5 = ResUnit()

        self.c4 = Conv2D(filters=64,
                         kernel_size=(3, 3),
                         padding='same',
                         activation='relu',
                         kernel_initializer=tf.initializers.he_normal())
        self.c5 = Conv2D(filters=32,
                         kernel_size=(3, 3),
                         padding='same',
                         activation='relu',
                         kernel_initializer=tf.initializers.he_normal())
        self.c6 = Conv2D(filters=1,
                         kernel_size=(3, 3),
                         padding='same',
                         kernel_initializer=tf.initializers.he_normal())

    def call(self, inputs, training=None, mask=None):
        # 1st path
        y1 = self.c1(inputs)

        y1r1 = self.c1r1(y1)

        y1r2 = self.c1r2(y1r1)
        y1r2 = y1r2 + y1 # 바로 이전 레이어는 ResUnit에서 더하므로 안 더해도 됨

        y1r3 = self.c1r3(y1r2)
        y1r3 = y1r3 + y1r1 + y1

        # 2nd path
        y2 = self.c2(inputs)

        y2r1 = self.c2r1(y2)

        y2r2 = self.c2r2(y2r1)
        y2r2 = y2r2 + y2

        y2r3 = self.c2r3(y2r2)
        y2r3 = y2r3 + y2r1 + y2

        y2r4 = self.c2r4(y2r3)
        y2r4 = y2r4 + y2r2 + y2r1 + y2

        # 3rd path
        y3 = self.c3(inputs)

        y3r1 = self.c3r1(y3)

        y3r2 = self.c3r2(y3r1)
        y3r2 = y3r2 + y3

        y3r3 = self.c3r3(y3r2)
        y3r3 = y3r3 + y3r1 + y3

        y3r4 = self.c3r4(y3r3)
        y3r4 = y3r4 + y3r2 + y3r1 + y3

        y3r5 = self.c3r5(y3r4)
        y3r5 = y3r5 + y3r3 + y3r2 + y3r1 + y3

        y4 = tf.concat((y1r3, y2r4, y3r5), axis=-1) # 192개의 feature

        y5 = self.c4(y4)
        y6 = self.c5(y5)
        y7 = self.c6(y6)

        return y7 + inputs