import torch
import torchvision

x = torch.rand(5, 3)
print(x)

print('Cuda')
cuda = torch.cuda.is_available()

print(cuda)

weights = torchvision.models.video.MViT_V2_S_Weights.DEFAULT
net = torchvision.models.video.mvit_v2_s(weights=weights)

print('Versão do PyTorch: ')
print(torch.__version__)

