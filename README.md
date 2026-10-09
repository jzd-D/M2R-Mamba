# M2R-Mamba：实验代码与复现

本仓库提供 M2R-Mamba 的模型实现、数据预处理、训练、评估和结果核验。实验包括 PTB-XL 主对比与消融，以及 MIT-BIH 七个基线和最终校准候选。

## 内容与边界

- `src/`：模型、数据处理和指标实现。PTB/基线使用 `models/m2r_mamba.py`；MIT 最终候选使用 `models/m2r_mamba_mit.py`，后者保留中心心搏读出。二者模型算法按原冻结源码保留。
- `experiments/`：三个冻结阶段的训练入口和完整配置；75 个 PTB 运行、35 个 MIT 基线运行、5 个 MIT 最终候选运行，共 115 个独立训练配置。PTB 主对比与消融共用的完整模型只训练一次。
- `scripts/`：数据准备、串行启动、分阶段评估、MIT 校准、结果重算和完整性检查。
- `reference_results/`：精简的逐种子结果、原数据划分审计、统计结果、原运行环境和来源摘要。

本快照保留最终训练与校准方案，以及核验结果所需的精简记录。不包含原始 ECG、处理后数组、权重、逐样本预测、训练日志、缓存、虚拟环境、IDE/SSH 配置、凭据、历史搜索和自动调度脚本。

## 1. 无 GPU 的快速核验

从本仓库根目录执行；Python 标准库即可运行下面两条检查，不会训练模型：

```bash
python scripts/verify_snapshot.py
python scripts/run.py all --dry-run
```

根据原始逐种子数值重算均值及样本标准差：

```bash
python scripts/summarize.py
```

输出到 `reproduced/summary/`。该目录存在时会拒绝覆盖，可用 `--output` 指定新的目录。冻结结果与新训练输出分开保存。

## 2. 训练环境

训练面向 Linux + NVIDIA CUDA；原冻结镜像的实际版本已核对：Python 3.11.10、PyTorch 2.5.0+cu124、CUDA 12.4、Mamba-SSM 2.3.1、causal-conv1d 1.6.1。其他直接依赖固定在 `requirements.txt`，完整核对记录见 `reference_results/runtime.json`。它记录的是运行镜像的软件版本，不是可下载的镜像地址，也不是跨硬件逐位一致的保证。

在新的 Python 3.11 环境中按顺序安装：

```bash
python -m pip install torch==2.5.0 --index-url https://download.pytorch.org/whl/cu124
python -m pip install -r requirements.txt
python -m pip install causal-conv1d==1.6.1 --no-build-isolation
python -m pip install mamba-ssm==2.3.1 --no-build-isolation
```

Mamba/CUDA 扩展必须与 PyTorch、CUDA 和编译环境兼容；若需要源码编译，使用具备 CUDA 工具链的环境。[Mamba 官方安装说明](https://github.com/state-spaces/mamba)。本次导出没有在新的空白环境重新安装所有依赖或重跑 115 次 GPU 训练。

环境安装后可运行 `python scripts/check_environment.py --models`，核对已安装版本、23 种独立模型配置的参数量及普通 CNN/RNN 基线的 CPU 前向。本快照已在原冻结镜像的独立 CPU 容器中通过该检查。

## 3. 数据准备

分别从 [PTB-XL 1.0.3](https://physionet.org/content/ptb-xl/1.0.3/) 和 [MIT-BIH 1.0.0](https://physionet.org/content/mitdb/1.0.0/) 获取原始公开数据，遵守数据源条款并引用相应数据集文献。数据不随此代码包分发。

将 PTB 的 `ptbxl_database.csv`、`scp_statements.csv` 和 `records100/` 放在 `data/raw/ptbxl/`；MIT 的 `.dat`、`.hea`、`.atr` 文件放在 `data/raw/mitbih/`。`data/raw/` 和 `data/processed/` 均被 Git 忽略。

```bash
python scripts/prepare_data.py ptbxl --raw data/raw/ptbxl
python scripts/prepare_data.py mitbih --raw data/raw/mitbih
```

PTB 使用 100 Hz、官方折 1–8/9/10，保留原程序的全部记录处理；MIT 使用 N/S/V/F、两路波形加八路 RR 特征、五心搏上下文，训练集排除与测试记录 202 来自同一受试者的记录 201；验证记录为 106/119/223/230。归一化只使用训练数据。

准备完成后程序强制比对冻结划分的样本数、类别计数和样本 ID 摘要。预期 PTB 的训练/验证/测试为 17418/2183/2198，MIT 为 40097/8858/49603。不匹配时应先排查数据版本，避免混用不同划分的结果。缓存已经存在时拒绝覆盖；另用 `--processed` 指定全新的输出目录。准备数据需要数 GB 磁盘与足够内存，原始数据目录只读。

## 4. 训练与测试

所有命令默认**串行**运行，每次仅一个训练进程。GPU 选择由 `CUDA_VISIBLE_DEVICES` 控制，程序不会探测或停止其他人的任务。完整超参数和种子以各阶段 `manifest.json` 为准。

冻结配置保留原实验的显存分配上限（GPU 容量的 15%），在较小 GPU 上可能不足；可在自己的副本中调整 `gpu_memory_fraction` 这一资源参数。完整比较中的较大基线需要比主模型更多显存。

先做一个真正的 GPU 小规模检查（需要已准备数据）：

```bash
python scripts/run.py ptb_main --job ptbxl_m2r_full_s1 --smoke --output reproduced/smoke_check
```

只训练主模型五个种子时，逐个使用 `--job ptbxl_m2r_full_s1` 至 `s5`；完整实验复现用：

```bash
python scripts/run.py standard
python scripts/run.py ptb_ablation
python scripts/run.py ptb_ablation --phase evaluate
python scripts/run.py mit_final
python scripts/run.py mit_final --phase evaluate
python scripts/calibrate_mit.py init
python scripts/calibrate_mit.py select
python scripts/calibrate_mit.py evaluate
python scripts/summarize.py --from-runs reproduced --output reproduced/new_summary
```

`standard` 包含 60 个原 PTB 主对比/消融运行及 35 个 MIT 基线运行，按原训练器在每次验证集选出最佳模型后测试。`ptb_ablation` 只训练新增的 15 个消融运行；`mit_final` 训练最终候选的 5 个种子。后两个阶段必须完成全部训练，冻结全部检查点后，再单独测试。也可用 `ptb_main`（40 次）或 `mit_baselines`（35 次）选择子集。`--seeds` 可用于调试子集；计算五种子汇总时必须保留全部五个种子。

默认数据位于 `data/processed/`，新输出位于 `reproduced/`。启动器的 `--data` / `--output` 可修改位置；直接执行阶段统计和校准程序时，用环境变量 `M2R_DATA_ROOT` / `M2R_OUTPUT_ROOT` 指向同样的路径。运行目录已存在时拒绝覆盖，避免覆盖已完成或中断的结果。

主数据集统计和推理成本（耗时，不属于快速核验）：

```bash
python experiments/standard/aggregate.py --bootstrap
python experiments/ptb_ablation/aggregate.py
python experiments/standard/benchmark.py
```

患者重采样需要训练产生的逐样本测试预测；最小包只保留原重采样结果。时延测量需要空闲 GPU，脚本检测到其他计算进程会跳过；不同硬件的时延不可直接要求一致。

## 实验解释与发布范围

保留全部种子，标准差使用 `ddof=1`；不隐藏较差消融。PTB 是主研究，MIT 为另外训练的第二任务适用性检查，不是跨数据集权重迁移。MIT 最终候选追加了调参预算，与七个基线的搜索预算不同，且既有测试结果曾被查看。原验证集校准选择的 F 类 logit 惩罚为 1.5；重训练时应重新在验证集选择，不得为了匹配原表而看测试集调参。选择规则和限制完整保存在校准脚本中。

这份快照不包含许可证授权文件，因为现有归档中没有可直接沿用的项目许可证；依赖和数据仍遵守各自许可证，项目许可证由作者决定。可以直接把本目录作为仓库根目录提交；`.gitignore` 会排除后续数据、模型权重、缓存和训练日志。
