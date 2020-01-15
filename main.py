import argparse
from dataset import *

def main(flags):
    # 데이터셋 클래스
    ds = Dataset(flags)

    # 모델 구성
    mod = __import__('nets', fromlist=['*'])
    model = getattr(mod, flags.model) # --> ivpl24()
    model = model()

    opt = tf.optimizers.Adam(1e-4)
    loss_fn = tf.losses.MeanAbsoluteError()

    model.compile(optimizer=opt, loss=loss_fn)

    shape = [None, flags.psize, flags.psize, flags.channel]
    #model.build(input_shape=[shape, shape]) # 입력에 대한 shape, label에 대한 shape

    #model.summary()

    train_loss = tf.metrics.Mean()

    for epoch in range(flags.epoch):
        for step, batch in enumerate(ds.train_set):
            train_loss(model.train_on_batch(batch[0], batch[1]))

        print("Loss: %f" % (train_loss.result()))

    #pass
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="매개변수 목록",
        epilog="바위"
    )
    parser.add_argument('--model', type=str, choices=["ivpl24"],
                        help="신경망 모델")
    parser.add_argument('--data', type=str, default="./data",
                        help="YUV 파일이 저장된 위치")
    parser.add_argument('--psize', type=int, default=8,
                        help="이미지 패치 사이즈")
    parser.add_argument('--batch', type=int, default=128,
                        help="배치 사이즈")
    parser.add_argument('--channel', type=int, default=1,
                        help="이미지 컬러 채널")
    parser.add_argument('--epoch', type=int, default=3000,
                        help="학습 반복 횟수")

    flags, _ = parser.parse_known_args()
    main(flags)