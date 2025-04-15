# 项目名称  
语义分割模型比较 (PSPNet, DeepLabV3, CCNet)  

## 项目结构  
- models/ : 各种分割模型的实现  
- dataloader.py : 数据加载与数据增强  
- train.py : 训练脚本  
- evaluate.py : 评估脚本  

## 运行步骤  
1. 安装依赖  
```bash  
pip install -r requirements.txt 
2.模型训练
 python train.py  
3.评估模型
python evaluate.py  