# import torch

# # CUDA 사용 가능 여부 확인
# if torch.cuda.is_available():
#     print("CUDA is available. GPU support is enabled.")
#     # 현재 GPU 장치 이름 출력
#     print("Current GPU Device:", torch.cuda.get_device_name(torch.cuda.current_device()))
#     # CUDA 버전 출력
#     print("CUDA Version:", torch.version.cuda)
# else:
#     print("CUDA is not available. No GPU support.")

import tensorflow as tf

# GPU 사용 가능 여부 확인
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        # 필요한 GPU 메모리만큼만 할당하기 위해 메모리 성장을 허용
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("GPUs available:", gpus)
    except RuntimeError as e:
        # 프로그램 시작 후에 메모리 성장을 설정할 수 없을 때 발생
        print(e)
else:
    print("No GPUs found.")
