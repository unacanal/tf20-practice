import argparse
from dataset import *

def main(flags):
    ds = Dataset(flags)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="매개변수 목록",
        epilog="잘가유"
    )
    parser.add_argument('--data', type=str, default="./data",
                        help="YUV 파일이 저장된 위치")
    parser.add_argument('--psize', type=int, default=8,
                        help="이미지 패치 사이즈")
    parser.add_argument('--batch', type=int, default=128,
                        help="배치 사이즈")

    flags, _ = parser.parse_known_args()
    main(flags)