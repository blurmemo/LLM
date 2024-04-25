## OpenCompass

## 特点

- 大语言模型、多模态模型评测框架

- 开源可复现：提供公平、公开、可复现的大模型评测方案
- 全面的能力维度：五大维度设计，提供 70+ 个数据集约 40 万题的的模型评测方案，全面评估模型能力
- 丰富的模型支持：已支持 20+ HuggingFace 及 API 模型
- 分布式高效评测：一行命令实现任务分割和分布式评测，数小时即可完成千亿模型全量评测
- 多样化评测范式：支持零样本、小样本及思维链评测，结合标准型或对话型提示词模板，轻松激发各种模型最大性能
- 灵活化拓展：想增加新模型或数据集？想要自定义更高级的任务分割策略，甚至接入新的集群管理系统？OpenCompass 的一切均可轻松扩展

## 评测对象

语言大模型、多模态大模型

## 工具架构



[![图片](https://private-user-images.githubusercontent.com/148421775/321935556-705924f8-01b0-48f2-bca7-c660445b013f.png?jwt=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3MTQwMzQ4NjQsIm5iZiI6MTcxNDAzNDU2NCwicGF0aCI6Ii8xNDg0MjE3NzUvMzIxOTM1NTU2LTcwNTkyNGY4LTAxYjAtNDhmMi1iY2E3LWM2NjA0NDViMDEzZi5wbmc_WC1BbXotQWxnb3JpdGhtPUFXUzQtSE1BQy1TSEEyNTYmWC1BbXotQ3JlZGVudGlhbD1BS0lBVkNPRFlMU0E1M1BRSzRaQSUyRjIwMjQwNDI1JTJGdXMtZWFzdC0xJTJGczMlMkZhd3M0X3JlcXVlc3QmWC1BbXotRGF0ZT0yMDI0MDQyNVQwODQyNDRaJlgtQW16LUV4cGlyZXM9MzAwJlgtQW16LVNpZ25hdHVyZT04YWIyZjdjMTA3M2ZlMzVmMWJkM2YzMzYxMjI4NmU1MDY0OWJmYTAxNzhkYzYyMTNmYTEwYzU3YmNiODZhNjlkJlgtQW16LVNpZ25lZEhlYWRlcnM9aG9zdCZhY3Rvcl9pZD0wJmtleV9pZD0wJnJlcG9faWQ9MCJ9.MBVga1rpuOZX92D8mhVXmioUtXiifD1VXNMxgHz5kAc)](https://private-user-images.githubusercontent.com/148421775/321935556-705924f8-01b0-48f2-bca7-c660445b013f.png?jwt=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3MTQwMzQ4NjQsIm5iZiI6MTcxNDAzNDU2NCwicGF0aCI6Ii8xNDg0MjE3NzUvMzIxOTM1NTU2LTcwNTkyNGY4LTAxYjAtNDhmMi1iY2E3LWM2NjA0NDViMDEzZi5wbmc_WC1BbXotQWxnb3JpdGhtPUFXUzQtSE1BQy1TSEEyNTYmWC1BbXotQ3JlZGVudGlhbD1BS0lBVkNPRFlMU0E1M1BRSzRaQSUyRjIwMjQwNDI1JTJGdXMtZWFzdC0xJTJGczMlMkZhd3M0X3JlcXVlc3QmWC1BbXotRGF0ZT0yMDI0MDQyNVQwODQyNDRaJlgtQW16LUV4cGlyZXM9MzAwJlgtQW16LVNpZ25hdHVyZT04YWIyZjdjMTA3M2ZlMzVmMWJkM2YzMzYxMjI4NmU1MDY0OWJmYTAxNzhkYzYyMTNmYTEwYzU3YmNiODZhNjlkJlgtQW16LVNpZ25lZEhlYWRlcnM9aG9zdCZhY3Rvcl9pZD0wJmtleV9pZD0wJnJlcG9faWQ9MCJ9.MBVga1rpuOZX92D8mhVXmioUtXiifD1VXNMxgHz5kAc)

- 模型层：大模型评测所涉及的主要模型种类，OpenCompass 以基座模型和对话模型作为重点评测对象
- 能力层：OpenCompass 从本方案从通用能力和特色能力两个方面来进行评测维度设计。在模型通用能力方面，从语言、知识、理解、推理、安全等多个能力维度进行评测。在特色能力方面，从长文本、代码、工具、知识增强等维度进行评测
- 方法层：OpenCompass 采用客观评测与主观评测两种评测方式。客观评测能便捷地评估模型在具有确定答案（如选择，填空，封闭式问答等）的任务上的能力，主观评测能评估用户对模型回复的真实满意度，OpenCompass 采用基于模型辅助的主观评测和基于人类反馈的主观评测两种方式
- 工具层：OpenCompass 提供丰富的功能支持自动化地开展大语言模型的高效评测。包括分布式评测技术，提示词工程，对接评测数据库，评测榜单发布，评测报告生成等诸多功能

## 评测方法

- 客观评测+主观评测
- 针对具有确定性答案的能力维度和场景，通过构造丰富完善的评测集，对模型能力进行综合评价
- 针对体现模型能力的开放式或半开放式的问题、模型安全问题等，采用主客观相结合的评测方式

### 客观评测

- 定量指标比较模型的输出与标准答案的差异，并根据结果衡量模型的性能

- 由于大语言模型输出自由度较高，在评测阶段，对其输入和输出作一定的规范和设计，尽可能减少噪声输出在评测阶段的影响

- 引导模型按照一定的模板输出答案

  - 采用提示词工程 （prompt engineering）和语境学习（in-context learning）

  - 评测方式

    - 判别式评测：该评测方式基于将问题与候选答案组合在一起，计算模型在所有组合上的困惑度（perplexity），并选择困惑度最小的答案作为模型的最终输出
      - 例如，若模型在 问题? 答案1 上的困惑度为 0.1，在 问题? 答案2 上的困惑度为 0.2，最终我们会选择 答案1 作为模型的输出

    - 生成式评测：该评测方式主要用于生成类任务，如语言翻译、程序生成、逻辑分析题等
      - 使用问题作为模型的原始输入，并留白答案区域待模型进行后续补全，通常还需要对其输出进行后处理，以保证输出满足数据集的要求

### 主观评测

- 借助受试者的主观判断对具有对话能力的大语言模型进行能力评测
- 基于模型的能力维度构建主观测试问题集合，并将不同模型对于同一问题的不同回复展现给受试者，收集受试者基于主观感受的评分
  - 真实人类专家的主观评测与基于模型打分的主观评测相结合的方式开展模型能力评估
  - 采用单模型回复满意度统计和多模型满意度比较两种方式

### 概览

![图片](https://private-user-images.githubusercontent.com/148421775/321935388-276fa169-4937-4ef0-9620-1f7863054863.png?jwt=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3MTQwMzQ4NjQsIm5iZiI6MTcxNDAzNDU2NCwicGF0aCI6Ii8xNDg0MjE3NzUvMzIxOTM1Mzg4LTI3NmZhMTY5LTQ5MzctNGVmMC05NjIwLTFmNzg2MzA1NDg2My5wbmc_WC1BbXotQWxnb3JpdGhtPUFXUzQtSE1BQy1TSEEyNTYmWC1BbXotQ3JlZGVudGlhbD1BS0lBVkNPRFlMU0E1M1BRSzRaQSUyRjIwMjQwNDI1JTJGdXMtZWFzdC0xJTJGczMlMkZhd3M0X3JlcXVlc3QmWC1BbXotRGF0ZT0yMDI0MDQyNVQwODQyNDRaJlgtQW16LUV4cGlyZXM9MzAwJlgtQW16LVNpZ25hdHVyZT04OTBlNDJmY2NiY2U2YWEzMjg0ZGI4NTZiMDk0Njk1YTY2YmM4ZWExYjEwM2EwY2VhNGRkZDllNGU4ZTI1MzIzJlgtQW16LVNpZ25lZEhlYWRlcnM9aG9zdCZhY3Rvcl9pZD0wJmtleV9pZD0wJnJlcG9faWQ9MCJ9.UBz1ZJcVM_s65jSz8NYrudgbGz_ogVdZlof4PIp9GCY)

- 阶段：配置 -> 推理 -> 评估 -> 可视化

  - 配置：这是整个工作流的起点
    - 选择要评估的模型和数据集，还可以选择评估策略、计算后端等，并定义显示结果的方式。

  - 推理与评估：对模型和数据集进行并行推理和评估；推理阶段主要是让模型从数据集产生输出，而评估阶段则是衡量这些输出与标准答案的匹配程度
    - 拆分为多个同时运行的“任务”以提高效率；如果计算资源有限，这种策略可能会使评测变得更慢
    - 了解该问题及解决方案，可以参考 FAQ: 效率

  - 可视化：评估完成后，将结果整理成易读的表格，并将其保存为 CSV 和 TXT 文件
    - 可激活飞书状态上报功能，此后可以在飞书客户端中及时获得评测状态报告



## 实战

### env

```
cuda 11.7 10% A100 * 1
```

### install

```
studio-conda -o internlm-base -t opencompass
source activate opencompass
git clone -b 0.2.4 https://github.com/open-compass/opencompass
cd opencompass
pip install -e .
```

**如果pip install -e .安装未成功**

```
pip install -r requirements.txt
```

有部分第三方功能,如代码能力基准测试 HumanEval 以及 Llama 格式的模型评测,可能需要额外步骤才能正常运行，如需评测，详细步骤请参考安装指南

### 数据准备

解压评测数据集到 data/ 处

```
cp /share/temp/datasets/OpenCompassData-core-20231110.zip /root/opencompass/
unzip OpenCompassData-core-20231110.zip
```

#### 查看支持的数据集和模型

```
# 列出所有跟 InternLM 及 C-Eval 相关的配置
python tools/list_configs.py internlm ceval
```

### 评测

- 评测 InternLM2-Chat-1.8B 模型在 C-Eval 数据集上的性能
- --debug 模式下，任务将按顺序执行，并实时打印输出

```
# bug
pip install protobuf
```

```
python run.py --datasets ceval_gen --hf-path /share/new_models/Shanghai_AI_Laboratory/internlm2-chat-1_8b --tokenizer-path /share/new_models/Shanghai_AI_Laboratory/internlm2-chat-1_8b --tokenizer-kwargs padding_side='left' truncation='left' trust_remote_code=True --model-kwargs trust_remote_code=True device_map='auto' --max-seq-len 1024 --max-out-len 16 --batch-size 2 --num-gpus 1 --debug
```

- 命令解析

```
python run.py
--datasets ceval_gen \
--hf-path /share/new_models/Shanghai_AI_Laboratory/internlm2-chat-1_8b \  # HuggingFace 模型路径
--tokenizer-path /share/new_models/Shanghai_AI_Laboratory/internlm2-chat-1_8b \  # HuggingFace tokenizer 路径（如果与模型路径相同，可以省略）
--tokenizer-kwargs padding_side='left' truncation='left' trust_remote_code=True \  # 构建 tokenizer 的参数
--model-kwargs device_map='auto' trust_remote_code=True \  # 构建模型的参数
--max-seq-len 1024 \  # 模型可以接受的最大序列长度
--max-out-len 16 \  # 生成的最大 token 数
--batch-size 2  \  # 批量大小
--num-gpus 1  # 运行模型所需的 GPU 数量
--debug
```

**遇到错误mkl-service + Intel(R) MKL MKL_THREADING_LAYER=INTEL is incompatible with libgomp.so.1 ... 解决方案**

```
export MKL_SERVICE_FORCE_INTEL=1
#或
export MKL_THREADING_LAYER=GNU
```



## 自定义数据集客主观评测

- 详细的客观评测参见 https://opencompass.readthedocs.io/zh-cn/latest/advanced_guides/new_dataset.html
- 详细的主观评测参见 https://opencompass.readthedocs.io/zh-cn/latest/advanced_guides/subjective_evaluation.html
- 评估步骤：https://opencompass.readthedocs.io/zh-cn/latest/advanced_guides/needleinahaystack_eval.html

### 简介

- 方式：打分，对战，多模型评测等。

### 数据污染评估简介

- 数据污染是指本应用在下游测试任务重的数据出现在了大语言模型 (LLM) 的训练数据中，从而导致在下游任务 (例如，摘要、自然语言推理、文本分类) 上指标虚高，无法反映模型真实泛化能力的现象
-  由于数据污染的源头是出现在 LLM 所用的训练数据中，因此最直接的检测数据污染的方法就是将测试数据与训练数据进行碰撞，然后汇报两者之间有多少语料是重叠出现的，经典的 GPT-3 论文中的表 C.1 会报告了相关内容
- 开源社区往往只会公开模型参数而非训练数据集，在此种情况下 如何判断是否存在数据污染问题或污染程度如何，这些问题还没有被广泛接受的解决方案

### 评估步骤

https://opencompass-cn.readthedocs.io/zh-cn/latest/advanced_guides/contamination_eval.html

### 大海捞针测试简介

- 大海捞针测试（灵感来自 NeedleInAHaystack）是指通过将关键信息随机插入一段长文本的不同位置，形成大语言模型 (LLM) 的Prompt，通过测试大模型是否能从长文本中提取出关键信息，从而测试大模型的长文本信息提取能力的一种方法，可反映LLM长文本理解的基本能力

### 数据集介绍

- Skywork/ChineseDomainModelingEval 数据集收录了 2023 年 9 月至 10 月期间发布的高质量中文文章，涵盖了多个领域。这些文章确保了公平且具有挑战性的基准测试

- 该数据集包括特定领域的文件：

  - zh_finance.jsonl 金融

  - zh_game.jsonl 游戏

  - zh_government.jsonl 政务

  - zh_movie.jsonl 电影

  - zh_tech.jsonl 技术

  - zh_general.jsonl 综合 这些文件用于评估LLM对不同特定领域的理解能力