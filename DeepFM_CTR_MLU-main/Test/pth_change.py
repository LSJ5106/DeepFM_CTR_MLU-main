import torch

state_dict = torch.load("checkpoint.pth", map_location=torch.device('cpu'))	#xxx.pth或者xxx.pt就是你想改掉的权重文件
torch.save(state_dict, "old.pth", _use_new_zipfile_serialization=False)
