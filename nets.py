import tensorflow as tf
from tensorflow.keras.layers import Conv2D, BatchNormalization, Layer, ReLU

# 막 만드는 네트워크
class ivpl24(tf.keras.Model): # inherit
    def __init__(self):
        super(ivpl24, self).__init__() # 상위 클래스의 init함수를 실행

        self.layer_list = []
        self.layer_list.append(BatchNormalization())
        for i in range(23):
            conv = Conv2D(filters=64,
                          kernel_size=(5, 5),
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