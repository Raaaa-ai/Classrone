aloha


# 学习总结
## 一、Git版本控制
学习了Git基础工作流：clone、add、commit、push。commit仅在本地保存快照，push同步到远程仓库。
学会创建分支`for_fun`，在独立分支开发，不污染主线main。分支合并时，如果两个分支修改同一文件会产生冲突，需要手动解决。同时可以通过commit哈希回访历史提交，查看项目过去的状态。

## 二、HuggingFace预训练模型推理
HuggingFace提供transformers、datasets库，可以快速加载预训练ResNet-50模型和MNIST数据集。ResNet在ImageNet大数据集预训练，直接在陌生数据集MNIST推理，没有微调的情况下预测效果很差。
推理时需要使用`model.eval()`和`torch.no_grad()`关闭梯度，加速推理、节省内存。

## 三、图像预处理与Resize
预训练ResNet要求输入是224×224的三通道RGB图像。MNIST原始图片是28×28单通道灰度图，需要两步预处理：
1. 将灰度图转换成RGB三通道；
2. 使用Resize将图片显式缩放至模型要求的224×224尺寸。
AutoImageProcessor可以后续做归一化与张量转换。图像尺寸、通道必须匹配模型要求，否则模型无法正常推理。