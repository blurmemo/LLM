### pip换源

- 参考[PyPI 软件仓库镜像使用帮助 - MirrorZ Help (cernet.edu.cn)](https://help.mirrors.cernet.edu.cn/pypi/)

- 或者

  - 临时使用镜像源安装

    - ```
      pip install -i https://mirrors.cernet.edu.cn/pypi/web/simple some-package
      ```

  - 设置 `pip` 默认镜像源，升级 `pip` 到最新的版本 (>=10.0.0) 后进行配置

    - ```
      python -m pip install --upgrade pip
      
      # 临时镜像升级
      python -m pip install -i https://mirrors.cernet.edu.cn/pypi/web/simple --upgrade pip
      
      pip config set global.index-url   https://mirrors.cernet.edu.cn/pypi/web/simple
      ```



### conda换源

- 修改 `.condarc`文件

  - Linux: `${HOME}/.condarc`
  - macOS: `${HOME}/.condarc`
  - Windows: `C:\Users\<YourUserName>\.condarc`
    - Windows 无法直接创建名为 `.condarc` 的文件，需先执行 `conda config --set show_channel_urls yes` 生成该文件后再修改

- ```
  channels:
  - defaults
  show_channel_urls: true
  default_channels:
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/r
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/msys2
  custom_channels:
  conda-forge: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
  pytorch: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
  ```

  

### 模型下载

#### Hungging Face

- 配置Hugging Face官方提供的huggingface-cli命令行工具依赖

```
pip install -U huggingface_hub
```

- 创建下载文件py

  - resume-download：断点续下

  - local-dir：本地存储路径
  - linux 环境下需要填写绝对路径

```
import os
# 下载模型
os.system('huggingface-cli download --resume-download internlm/internlm2-chat-7b --local-dir your_path')

# 另外一种方式下载
import os 
from huggingface_hub import hf_hub_download  # Load model directly 

hf_hub_download(repo_id="internlm/internlm2-7b", filename="config.json")
```



#### ModelScope

- 配置依赖

```
pip install modelscope==1.9.5
pip install transformers==4.35.2
```

- 使用snapshot_download函数下载模型
  - 第一个参数为模型名称，参数cache_dir为模型的下载路径
  - 下载路径最好为绝对路径

```
import torch
from modelscope import snapshot_download, AutoModel, AutoTokenizer
import os
model_dir = snapshot_download('Shanghai_AI_Laboratory/internlm2-chat-7b', cache_dir='your path', revision='master')
```



#### OpenXLab

- 指定模型仓库的地址、下载的文件的名称、文件所需下载的位置等，直接下载模型权重文件
- 使用download函数导入模型中心的模型

```
import torch
import os
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel
base_path = './local_files'
os.system('apt install git')
os.system('apt install git-lfs')
os.system(f'git clone https://code.openxlab.org.cn/Usr_name/repo_name.git {base_path}')
os.system(f'cd {base_path} && git lfs pull')
```



### 软链接清除方法

- 建立安全链接后删除命令

```
unlink link_name

# example
# 删除软链接 /root/demo/internlm2-chat-7b
cd /root/demo/
unlink internlm2-chat-7b
```