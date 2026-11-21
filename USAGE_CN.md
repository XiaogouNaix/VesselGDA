# 使用方法

1. 在本目录安装 `requirements.txt` 中的依赖。
2. 准备 FIVES 源模型权重和目标域数据，文件结构见 [README.md](README.md)。
3. 修改 `configs/exp_gnn_pseudo_learn_2d/fundus_to_dias.yaml` 或 `fundus_to_dsca.yaml` 中的路径和参数。
4. 按 README 顺序执行伪标签生成、Frangi 前处理、图构建、输入检查和适应训练。
5. 训练轮数由配置固定指定，训练期间不比较测试标签，不按测试集选择模型；最终保存 `state_dict_model_final.pt`。
6. 用 `save_val_epochs` 指定合并需要的阶段预测。此步骤仅保存预测，不计算测试指标。
7. 填写合并脚本中的 `fd_pred`、`fd_pred_with_gnn` 两个空路径并运行，或通过命令行传入目录。

超参数直接在配置文件中调整。示例数据仅演示目录格式，源权重和完整数据集需要另行准备。
