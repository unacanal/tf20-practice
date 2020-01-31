# image enhancement
import argparse
import time
from skimage.metrics import peak_signal_noise_ratio as psnr
from dataset import *


@tf.function # GPU로 돌리라는 속성 정의
def train_step(model, loss_fn, opt, batch):
    input_ = batch[0]
    label_ = batch[1]

    with tf.GradientTape() as tape:
        prediction = model(input_) # nets.py의 SRCNN.call(inputs) # SRCNN일때는 residual 추가
        loss = loss_fn(prediction, label_)

    grads = tape.gradient(loss, model.trainable_variables) # weight와 bias에 loss를 반영하는 부분
    opt.apply_gradients(zip(grads, model.trainable_variables)) # 모든 optimizer가 가지고 있는 함수 # zip은 따로 공부하셈
    return loss, prediction #, residual

def main(flags):
    # 데이터셋 클래스
    ds = Dataset(flags)

    # 모델 구성
    mod = __import__('nets', fromlist=['*'])
    model = getattr(mod, flags.model) # SRCNN
    model = model()

    # back-prop = Adam
    opt = tf.optimizers.Adam(1e-4)

    # L2 or MSE
    loss_fn = tf.losses.MeanAbsoluteError()

    model.compile(optimizer=opt, loss=loss_fn)

    shape = [None, flags.psize, flags.psize, flags.channel]

    train_loss = tf.metrics.Mean()


    dir_name = flags.model + time.strftime('-%Y%m%d-%H%M%S', time.localtime(time.time())) # string float time
    log_train = './log/' + dir_name + '/train/'
    train_summary_writer = tf.summary.create_file_writer(log_train)
    tf.summary.trace_on() # 그래프 저장

    count = 0
    for epoch in range(flags.epoch):
        for step, batch in enumerate(ds.train_set):
            count += 1

            loss, prediction = train_step(model, loss_fn, opt, batch) # 이 batch가 dataset.py의 train_set에서 오는 것입니다. # residual
            train_loss(loss)
            #train_loss(model.train_on_batch(batch[0], batch[1])) # [0] : input_, [1] : label_

            print("[Step: %10d] Loss: %f" % (count, train_loss.result()))

            if count == 1:
                with train_summary_writer.as_default():
                    tf.summary.trace_export(flags.model, step=0)
                    train_summary_writer.flush()

            if count % 100 == 0: # 100번마다 한 번 씩
                clipped = tf.clip_by_value(prediction, 0., 1.) # prediction은 0과1사이가 아닐수있음. 0보다 작은값은0, 1보다큰값은1로 만듬(rescaling이 아님)
                with train_summary_writer.as_default():
                    tf.summary.scalar('loss', train_loss.result(), step=count)
                    tf.summary.image('input', batch[0], step=count)
                    tf.summary.image('label', batch[1], step=count)
                    tf.summary.image('prediction', clipped, step=count)
                    # tf.summary.image('residual', residual, step=count)

                    train_summary_writer.flush()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="매개변수 목록",
        epilog="바위"
    )
    parser.add_argument('--model', type=str, choices=["ivpl24", "SRCNN", "ComplexModel"],
                        help="신경망 모델")
    parser.add_argument('--data', type=str, default="/home/ivpl-d14/Dataset/",
                        help="YUV 파일이 저장된 위치")
    parser.add_argument('--psize', type=int, default=45,
                        help="이미지 패치 사이즈")
    parser.add_argument('--batch', type=int, default=128,
                        help="배치 사이즈")
    parser.add_argument('--channel', type=int, default=1,
                        help="이미지 컬러 채널")
    parser.add_argument('--epoch', type=int, default=3000,
                        help="학습 반복 횟수")

    flags, _ = parser.parse_known_args()
    main(flags)