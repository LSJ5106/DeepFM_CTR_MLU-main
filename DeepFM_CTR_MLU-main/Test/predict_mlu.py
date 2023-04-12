import torch
from torch.autograd import Variable

from trainer import Trainer
from network import WideDeep
from criteo_loader import getTestData, getTrainData
import torch.utils.data as Data
import torch_mlu
import torch_mlu.core.mlu_model as ct
import torch_mlu.core.mlu_quantize as mlu_quantize
import torchvision.models as models
import platform  # 计时
import time
import numpy as np

ct.set_core_version("MLU270")
torch.set_grad_enabled(False)

# 模型参数，不动
widedeep_config = \
    {
        'deep_dropout': 0,
        'embed_dim': 8,  # 用于控制稀疏特征经过Embedding层后的稠密特征大小
        'hidden_layers': [256, 128, 64],
        'num_epoch': 10,
        'batch_size': 32,
        'lr': 1e-3,
        'l2_regularization': 1e-4,
        'device_id': 0,
        'use_cuda': False,
        'train_file': '../Data/criteo/processed_data/train_set.csv',
        'fea_file': '../Data/criteo/processed_data/fea_col.npy',
        'validate_file': '../Data/criteo/processed_data/val_set.csv',
        'test_file': '../Data/criteo/processed_data/test_set.csv',
        'model_name': '../TrainedModels/WideDeep.model'
    }


def quantize_model():
    # 数据集读取
    training_data, training_label, dense_features_col, sparse_features_col = getTrainData(widedeep_config['train_file'],
                                                                                          widedeep_config['fea_file'])
    test_data = getTestData(widedeep_config['test_file'])
    test_dataset = Data.TensorDataset(torch.tensor(test_data).float())

    # dataiter = iter(test_dataset)
    # data = dataiter.__next__()
    # data_tensor = torch.tensor([item.detach().numpy() for item in data])


    # 读取原始模型
    model = WideDeep(widedeep_config, dense_features_cols=dense_features_col,
                     sparse_features_cols=sparse_features_col).cpu()
    # 转量化模型
    quantized_model = mlu_quantize.quantize_dynamic_mlu(
        model, {'firstconv': False}, dtype='int16', gen_quant=True)

    # 执行推理，同时生成量化数据
    input_tensor = torch.tensor(test_data).float()    #自带转tensor方法
    output = quantized_model(input_tensor)

    # 保存量化模型参数
    quantized_model_path = './recom_quantize_state.pth'
    torch.save(quantized_model.state_dict(), quantized_model_path)
    print("\nSuccessfully Save recom_quantize_state.pth\n")

    return model, input_tensor, quantized_model_path


# 在线逐层推理
def online_inference(model, input_tensor, quantized_model_path):

    input_tensor_float = torch.tensor(input_tensor).float()
    #FLoatTensor转LongTensor

    # input_tensor_long = torch.LongTensor(input_tensor.cpu().numpy())

    ct.set_core_number(4)
    # input_tensor = Variable(input_tensor, requires_grad=False).long()

    # 转为量化模型
    quantized_model = mlu_quantize.quantize_dynamic_mlu(model)

    # 读取量化数据
    quantized_model.load_state_dict(torch.load(quantized_model_path))
    print("\nSuccessfully Load Quantized_Model\n")

    # 把模型和数据放到MLU设备上，执行推理
    quantized_model_mlu = quantized_model.to(ct.mlu_device())
    # print("\ninput_tensor = ", input_tensor, "\t", type(input_tensor), "\n")

    data_mlu = input_tensor_float.to(ct.mlu_device())
    # print("\ndata_mlu = ", data_mlu, "\n")

    output = quantized_model_mlu(data_mlu)
    print("\nSuccessfully Get quantized_model_mlu Output\n")
    print("\nOutput = ", output, "\n")

#在线融合推理
def online_fusion(model, input_tensor, quantized_model_path):

    #转换为量化模型
    quantized_model = mlu_quantize.quantize_dynamic_mlu(model)
    quantized_model.load_state_dict(torch.load(quantized_model_path))
    quantized_model_mlu = quantized_model.to(ct.mlu_device())

    input_tensor = input_tensor.long()
    print("\nSuccessfully Get Long Tensor\n")

    data_mlu = input_tensor.to(ct.mlu_device())
    print("\nSuccessfully Get data for MLU\n")

    #转换为带静态图的跟踪模型
    traced_quantized_model_mlu = torch.jit.trace(
        quantized_model_mlu, data_mlu, check_trace=False
    )
    print("\nSuccessfully Trace\n")

    #执行推理
    output = traced_quantized_model_mlu(data_mlu)

    print("\nSuccessfully Output traced_quantized_model_mlu\n")

def main():
    # 模型量化
    model, input_tensor, quantized_model_path = quantize_model()
    # 在线逐层推理
    # online_inference(model, input_tensor, quantized_model_path)
    #在线融合推理
    online_fusion(model, input_tensor, quantized_model_path)

if __name__ == "__main__":
    main()
