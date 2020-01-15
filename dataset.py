# Dataset
# Load YUV sequences
# Create low-quality file using bicubic-interpolation
# 패치 단위로 분할해서 별도의 파일로 저장
import glob
import os
import numpy as np
import cv2
import h5py
import tensorflow as tf

class Dataset:
    def __init__(self, flags):
        self.flags = flags

        # [] 리스트
        # () 튜플
        # {} 딕셔너리

        train_set = self.load_h5()
        self.train_set = tf.data.Dataset.from_tensor_slices(train_set)
        self.train_set = self.train_set.shuffle(1000).batch(flags.batch)

    def load_h5(self):
        if not os.path.exists('./train.h5'):
            self.create_h5()
        with h5py.File('./train.h5', 'r') as hf:
            # shape = [psize, psize] -> [psize, psize, 1]
            input_ = np.asarray(hf.get('input_'))
            label_ = np.asarray(hf.get('label_'))

            return np.expand_dims(input_, axis=-1), np.expand_dims(label_, axis=-1)

    def create_h5(self):
        yuv_list = glob.glob(os.path.join(self.flags.data, "*.yuv"))
        file = yuv_list[0]
        # name_가로x세로_framerate.yuv
        # 정규식을 다루는 패키지 = import re # 추후에 공부해보기
        fname = os.path.basename(file)  # only file name
        seg = fname.split('_')
        wh = seg[1].split('x')
        w = int(wh[0])
        h = int(wh[1])
        fp = open(file, 'rb')
        y = np.fromfile(fp, np.uint8, w * h).reshape((h, w))
        u = np.fromfile(fp, np.uint8, w * h // 4).reshape((h // 2, w // 2))  # // : 소수점 버림
        v = np.fromfile(fp, np.uint8, w * h // 4).reshape((h // 2, w // 2))
        # 이미지의 해상도 줄임
        small_y = cv2.resize(y, None, fx=1 / 3, fy=1 / 3)
        cubic_y = cv2.resize(small_y, (w, h), interpolation=cv2.INTER_CUBIC)
        # normalize 0~1
        ny = y / 255.
        ncubic_y = cubic_y / 255.
        # TODO: 테스트 코드
        # cv2.imwrite('cubic.jpg', cubic_y)
        input_list = []
        label_list = []
        # range(start, end, step)
        for i in range(0, h - self.flags.psize + 1, self.flags.psize):
            for j in range(0, w - self.flags.psize + 1, self.flags.psize):
                # slicing [from:to, from:to]
                input_ = cubic_y[i:i + self.flaself.gs.psize, j:j + self.flags.psize]  # 쪼개져 나온 패치
                label_ = y[i:i + self.flags.psize, j:j + self.flags.psize]

                input_list.append(input_)
                label_list.append(label_)
        input_ = np.asarray(input_list)
        label_ = np.asarray(label_list)
        """ hf = h5py.File('./train.h5', 'w')
                이렇게 하면 프로그램 종료시에 hf가 없어지지만 with구문을 이용하면 해당 부분에서만 사용됨
                """
        with h5py.File('./train.h5', 'w') as hf:
            hf.create_dataset('input_', data=input_)
            hf.create_dataset('label_', data=label_)




