import argparse
import yaml
import os
import time
import torch.backends.cudnn 

# 解析命令行参数
parser = argparse.ArgumentParser(description='Model Configuration')
parser.add_argument('--config', type=str, default='config.yaml', help='Path to the configuration file')

args = parser.parse_args()
# 加载 YAML 文件配置, 将 YAML 配置转化为命令行参数的默认值
with open(args.config, 'r') as file:
    yaml_config = yaml.safe_load(file)

for key, value in yaml_config.items():
    parser.add_argument(f'--{key}', default=value, type=type(value))

config = parser.parse_args()

if config.device == 'cuda':
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    
if config.model == 'cnn':
    from model import CNN as MyModel
elif config.model == 'rnn':
    from model import RNN as MyModel
elif config.model == 'transformer':
    from model import Transformer as MyModel
else: 
    print('No modelframe to train!')
    os._exit()

time_flag = True # 防止跨分钟训练导致重命名错误
T = None
def save_file(path: str):
    os.makedirs(config.folder) if not os.path.exists(config.folder) else True
    global time_flag,T
    if time_flag:
        T = time.localtime()
        time_flag = False
    time_str = time.strftime(r"%Y-%m-%d_%H.%M_", T)
    return os.path.join(config.folder, time_str + path)

def save_model(loss, accuracy=None):
    if accuracy is None:
        path = f'loss{loss:.4f}_model.ckpt'
    else:
        path = f'accuracy{accuracy:.3f}_model.ckpt'
    return save_file(path)