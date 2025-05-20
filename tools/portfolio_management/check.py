import torch

if torch.cuda.is_available():
    x = torch.rand(3, 3).cuda()
    print("CUDA Tensor:")
    print(x)
else:
    print("CUDA not available!")
