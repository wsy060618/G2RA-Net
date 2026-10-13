# SliceRel-U-Net

G2RA-Net是本项目的最终命名，其原名为SliceRel-U-Net。本仓库包含 SliceRel-U-Net 的模型、训练和评估代码。输入是同一病例中连续的 `S` 张切片，输出是每张切片的分割结果。

## 1. 安装

建议使用 Python 3.10 或 3.11。先根据显卡和 CUDA 版本安装 PyTorch，再安装本项目：

```bash
python -m venv .venv
```

Windows：

```bash
.venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -e .
```

Linux 或 macOS：

```bash
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

运行模型检查：

```bash
python scripts/smoke_test.py
```

出现下面的输出说明模型能够完成前向和反向计算：

```text
SliceRel-U-Net smoke test passed: (1, 3, 4, 32, 32)
```

## 2. 准备数据

支持 `.npy`、只含一个数组的 `.npz`、`.nii` 和 `.nii.gz`。每个图像和标签都应是形状一致的三维体数据。

建议按病例划分训练集、验证集和测试集。同一受试者的所有数据必须放在同一个集合中，测试集只用于最终评估。

将数据清单保存为 `data/manifest.csv`：

```csv
volume_id,image_path,label_path,split,spacing_0,spacing_1,spacing_2
case001,images/case001.nii.gz,labels/case001.nii.gz,train,1.0,1.0,1.0
case002,images/case002.nii.gz,labels/case002.nii.gz,val,1.0,1.0,1.0
case003,images/case003.nii.gz,labels/case003.nii.gz,test,1.0,1.0,1.0
```

需要满足以下条件：

- `image_path` 和 `label_path` 相对于 CSV 所在目录，也可以填写绝对路径。
- `split` 按 `train`、`val` 和 `test` 填写。
- 标签必须是 `0` 到 `num_classes - 1` 的整数，其中 `0` 是背景。
- 每个体数据沿切片轴的数量不能少于 `model.slice_count`。
- NIfTI 的空间信息从文件头读取；数组数据可用 `spacing_0` 至 `spacing_2` 填写体素间距，省略时按 `1.0` 处理。

## 3. 修改配置

复制 `configs/example.yaml`，为每组实验单独保存一份配置。开始训练前至少检查：

```yaml
model:
  num_classes: 36       # 包含背景类
  slice_count: 3        # 每个输入窗口的连续切片数
  variant: slicerel_unet

data:
  manifest: data/manifest.csv
  slice_axis: 2         # 0、1 或 2；NIfTI 在转为 RAS 后使用该轴
  group_stride: 1
  target_size: [224, 224]

training:
  output_dir: runs/oasis_s3
```

`target_size` 的高和宽建议能被 `16` 整除。为了让所有切片都被评估，`group_stride` 不应大于 `slice_count`。`adjacency_mlp_hidden: null` 会自动使用 `slice_count²`，改变切片数时不需要再手动修改该参数。

训练损失为：

```text
CE + Dice + 0.1 × HausdorffDTLoss
```

对应的正权重可以在配置文件的 `loss` 部分修改。训练前应固定数据划分、预处理、损失权重和随机种子，测试过程中不要再根据测试结果调整参数。

## 4. 训练

在仓库根目录运行：

```bash
python scripts/train.py --config configs/example.yaml
```

输出目录中包含：

- `best.pt`：验证阶段前景 Dice 最高的检查点；
- `last.pt`：最后一个 epoch 的检查点；
- `history.json`：每个 epoch 的训练损失、验证损失和验证 Dice。

训练中断后可从 `last.pt` 继续：

```bash
python scripts/train.py --config configs/example.yaml --resume runs/example/last.pt
```

继续训练时必须使用相同的模型结构配置。若修改了类别数、切片数、主干或模块设置，应开始一个新实验并使用新的输出目录。

## 5. 测试

训练完成后，用验证集选出的 `best.pt` 评估测试集：

```bash
python scripts/evaluate.py --config configs/example.yaml --checkpoint runs/example/best.pt --split test --output runs/example/test_metrics.json
```

相邻窗口有重叠时，程序先对同一切片的 logits 取平均，再计算每个病例、每个前景类别的 Dice，最后报告病例平均结果。没有出现在真实标签中的类别不会计入该病例的平均值。

## 6. 组织实验

主实验使用：

```yaml
variant: slicerel_unet
```

如需检查各模块的作用，可以保持数据划分和其余参数不变，只修改 `variant`：

| 配置值 | 使用的结构 |
| --- | --- |
| `unet` | U-Net |
| `gsrm_only` | U-Net + GSRM |
| `csag_only` | U-Net + CS-AG |
| `slicerel_unet` | U-Net + GSRM + CS-AG |

不同切片数、不同模块组合和不同随机种子应使用独立的配置文件及输出目录，避免检查点和日志相互覆盖。最终记录每次实验所用的配置文件、随机种子和 `best.pt` 对应的测试结果。


