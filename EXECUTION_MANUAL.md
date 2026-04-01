CS336 Assignment 1 执行手册（按章节精炼）

目标概览
- 从零实现：BPE 分词器、Transformer LM 组件、训练基础设施、解码与实验
- 代码入口：tests/adapters.py 中的 run_* / get_* 钩子
- 测试入口：tests/test_*.py

第 2 章：BPE Tokenizer

待实现函数清单（adapters.py）
- run_train_bpe
- get_tokenizer

对应测试文件
- tests/test_train_bpe.py
- tests/test_tokenizer.py
- tests/test_data.py（可能包含与数据/分词相关的辅助测试）

执行要点
- 使用字节级 BPE：先 UTF-8 bytes，再做 pre-tokenization
- 预分词正则使用 GPT-2 方案，计数用 re.finditer
- 处理 special tokens：先 split，再分别预分词，避免跨特殊 token 合并
- merge 规则：频次并列时按字典序选择更大的 pair
- 训练输出：vocab（id->bytes）与 merges（bytes pair 列表，保持创建顺序）
- tokenizer 支持 encode、decode、encode_iterable，decode 用 errors="replace"

实验建议
- TinyStories 上先用小样本调试，再训练 10k vocab
- OpenWebText 上用 32k vocab，注意内存与时间
- 记录吞吐（bytes/s）与压缩率（bytes/token）

第 3 章：Transformer 组件

待实现函数清单（adapters.py）
- run_linear
- run_embedding
- run_swiglu
- run_silu
- run_rmsnorm
- run_softmax
- run_scaled_dot_product_attention
- run_multihead_self_attention
- run_rope
- run_multihead_self_attention_with_rope
- run_transformer_block
- run_transformer_lm

对应测试文件
- tests/test_nn_utils.py（Linear / Embedding / RMSNorm / SiLU / Softmax / Swiglu）
- tests/test_model.py（Attention / RoPE / TransformerBlock / TransformerLM）
- tests/test_data.py（可能包含形状与数值一致性测试）

执行要点
- 线性层与 Embedding：不使用 nn.Linear/nn.Embedding
- RMSNorm：先 float32 计算，再回转原 dtype
- Softmax：减去最大值，避免溢出
- SDPA：支持可选 mask，False 位置填 -inf
- MHA：QKV 三次投影，head 维度作为 batch 维
- RoPE：只作用于 Q/K，支持 token_positions 切片
- TransformerBlock：pre-norm + residual
- TransformerLM：Embedding -> N 层 block -> final norm -> lm head

实验建议
- 用单 batch 过拟合检查梯度与数值稳定性
- 用 einsum / rearrange 保持张量形状一致

第 4 章：损失与优化

待实现函数清单（adapters.py）
- run_cross_entropy
- get_adamw_cls
- run_get_lr_cosine_schedule
- run_gradient_clipping

对应测试文件
- tests/test_optimizer.py
- tests/test_nn_utils.py（可能包含 softmax / cross entropy）

执行要点
- Cross-Entropy：log-sum-exp 稳定化，按 batch 均值返回
- AdamW：维护 m/v 一阶二阶矩，按算法做 bias-correction 与 decoupled weight decay
- LR schedule：warmup + cosine anneal + floor
- Gradient clipping：全参数合并 L2 范数，按比例缩放梯度

实验建议
- 学习率是最关键超参，先做 sweep 找稳定边界
- 记录 loss 曲线与 wallclock

第 5 章：数据与训练基础设施

待实现函数清单（adapters.py）
- run_get_batch
- run_save_checkpoint
- run_load_checkpoint

对应测试文件
- tests/test_data.py
- tests/test_serialization.py

执行要点
- get_batch：随机采样起点，构造 (x, y) 形状 (B, T)
- checkpoint：保存 model/optimizer state_dict 与 iteration
- 大数据用 np.memmap 或 np.load(mmap_mode="r")

实验建议
- CPU / MPS 环境用更小 tokens 数
- 检查 I/O 与 dataloader 是否成为瓶颈

第 6 章：解码与生成

待实现函数清单
- 解码函数（按手册要求实现：温度采样 + top-p）

对应测试文件
- 无强制测试文件（根据 writeup/实验提交检查）

执行要点
- 只取最后一个 token 的 logits 进行下一步采样
- 支持最大生成长度与遇到 <|endoftext|> 停止

实验建议
- 调整 temperature 与 top-p 获取流畅文本

第 7 章：实验与消融

需要完成的实验建议（无直接测试）
- TinyStories：学习率、batch size sweep
- 生成样本文本并分析质量
- 消融：去 RMSNorm、改 post-norm、去 RoPE、SwiGLU vs SiLU
- OpenWebText：同配置训练并对比 TinyStories loss
- 可选 leaderboard：在 1.5h 预算内优化

快速执行顺序建议
1) 先实现 BPE 训练与 tokenizer，跑 tests/test_train_bpe.py 与 tests/test_tokenizer.py
2) 实现基础算子与注意力，跑 tests/test_nn_utils.py 与 tests/test_model.py
3) 实现优化器与 loss，跑 tests/test_optimizer.py
4) 实现数据与 checkpoint，跑 tests/test_data.py 与 tests/test_serialization.py
5) 训练脚本与生成/实验记录
