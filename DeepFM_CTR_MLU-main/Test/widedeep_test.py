import torch
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

# ct.set_core_number(1)
ct.set_core_version("MLU270")
torch.set_grad_enabled(False)
import numpy as np

mean = [0.485, 0.456, 0.406]
# std = [1 / 255]
std = [0.229, 0.224, 0.225]

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

if __name__ == "__main__":
    ####################################################################################
    # WideDeep 模型
    ####################################################################################
    training_data, training_label, dense_features_col, sparse_features_col = getTrainData(widedeep_config['train_file'],
                                                                                          widedeep_config['fea_file'])
    # train_dataset = Data.TensorDataset(torch.tensor(training_data).float(), torch.tensor(training_label).float())
    test_data = getTestData(widedeep_config['test_file'])
    test_dataset = Data.TensorDataset(torch.tensor(test_data).float())

    model = WideDeep(widedeep_config, dense_features_cols=dense_features_col,
                        sparse_features_cols=sparse_features_col).cpu()

    model_path = "old.pth"

    # wideDeep.load_state_dict(torch.load(model_path), False)

    # 加载pth预训练模型
    pretrained_dict = torch.load(model_path, map_location="cpu")
    model_dict = model.state_dict()
    # 重新制作预训练的权重，主要是减去参数不匹配的层
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if (k in model_dict and 'fc' not in k)}
    # 更新权重
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    # # 有两个层参数不一样，不能用，切掉
    # ckpt = torch.load(model_path)
    # # ckpt = torch.load(model_path, map_location='cpu')
    # ckpt.pop("0.bias")
    # ckpt.pop("0.weight")
    # wideDeep.load_state_dict(ckpt, False)

    ####################################################################################
    # 模型训练阶段
    ####################################################################################
    # # 实例化模型训练器
    # trainer = Trainer(model=wideDeep, config=widedeep_config)
    # 训练
    # trainer.train(train_dataset)
    # 保存模型
    # torch.save(wideDeep.state_dict(), "./checkpoint.pth")

    # 加载保存的权重



    # 模型量化
    quantized_model = torch_mlu.core.mlu_quantize.quantize_dynamic_mlu(
        model, dtype='int8', gen_quant = True)

    #在CPU上进行推理，生成量化值
    input_tensor = torch.tensor(test_data).float()
    quantized_model(input_tensor)

    #保存量化模型
    model_path = "quantize_widedeep.pth"
    torch.save(quantized_model.state_dict(), model_path)
    print("\nSuccessfully Save quantized net\n")

    net = mlu_quantize.quantize_dynamic_mlu(model)
    net.load_state_dict(torch.load(model_path))
    net_mlu = net.to(ct.mlu_device())

    print("\n\nSuccessfuuly Load Quantized Model\n\n")

    # 转为LongTensor
    # input_tensor = input_tensor.cpu()
    # input_tensor = torch.LongTensor(input_tensor.numpy())
    input_tensor_mlu = input_tensor.to(ct.mlu_device())
    print("input_tensor.to(ct.mlu_device()) = ", type(input_tensor.to(ct.mlu_device())))

    output = net_mlu(input_tensor_mlu)

    # print("\nSuccessfully Predict quantized net\n")
    # net_quantization = mlu_quantize.quantize_dynamic_mlu(
    #     wideDeep, {'mean': mean, 'std': std, 'firstconv': True}, dtype='int8', gen_quant=True)
    # net_quantization.eval()
    # torch.save(net_quantization.state_dict(), 'test_quantization.pth')
    #
    # net = mlu_quantize.quantize_dynamic_mlu(wideDeep)
    # net.load_state_dict(torch.load('test_quantization.pth'))
    # net_mlu = net.to(ct.mlu_device())


    ####################################################################################
    # 模型测试阶段
    ####################################################################################
    # wideDeep.eval()
    # if widedeep_config['use_cuda']:
    #     wideDeep.loadModel(map_location=lambda storage, loc: storage.cuda(widedeep_config['device_id']))
    #     resNet = wideDeep.cuda()
    # else:
    #     wideDeep.loadModel(map_location=torch.device('cpu'))

    # 融合，生成静态图
    # ct.set_core_number(1)
    # ct.save_as_cambricon("int8_accuracy")
    #
    # trace_input = torch.tensor(test_data).float()
    #
    # # 从Float Tensor 转成Long Tensor
    # # trace_input = torch.LongTensor(trace_input.numpy())
    #
    # trace_input = trace_input.to(ct.mlu_device())
    #
    # print("\nSuccessfully Trace input\n\n")

    # quantized_net = torch.jit.trace(net_mlu, trace_input, check_trace=False)
    #
    # print("\n\nSuccessfully RongHe\n\n")

    # y_pred_probs = quantized_net(trace_input)

    print("\n\nSuccessfully output\n\n")
    print("Output = ", output, "\t", type(output))

    # y_pred = torch.where(y_pred_probs > 0.5, torch.ones_like(y_pred_probs), torch.zeros_like(y_pred_probs))
    # print("Test Data CTR Predict...\n ", y_pred.view(-1))
    index = torch.argmax(output)
    print("Predict = ", output[index])
