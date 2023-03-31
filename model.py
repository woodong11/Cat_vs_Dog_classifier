# 모델 구성에 필요
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data


# 미리 훈련된, pytorch가 제공하는 모델들 불러오는 라이브러리
import torchvision.models as models

''' 사전 훈련된 ResNet 불러오기 '''
# 2개 이진분류하는 모델
pretrained_model = models.resnet50(pretrained=True) # pretrained=True: 사전 학습된 모델 불러오기 
print("model load successed")


''' 내가 직접 ResNet 만들기 '''

