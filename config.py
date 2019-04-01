import argparse
import collections
import os
from inspect import currentframe

from easydict import EasyDict as edict

frame = currentframe().f_back
while frame.f_code.co_filename.startswith('<frozen'):
    frame = frame.f_back
import_from = frame.f_code.co_filename
eval_mode = 0 if 'eval' not in import_from else 1
train_mode = 0 if 'train' not in import_from else 1
attack_mode = 0 if 'attack' not in import_from else 1

config = edict(d=collections.OrderedDict())
# attack related
config.attack_network = "0"
config.step_size = 16.0
config.max_epsilon = 16.0
config.num_steps = 1
config.universal = False

# train related
config.lr = 0.001
config.lr_decay_points = [0, ]
config.steps_per_epoch = 5000
config.max_epoch = 50
config.batch = 32  # total batch size default is for 4 GPUs
config.optimizer = 'adam'  # 'momentum'
config.zero_init = True
config.GPU_ID = '0,1,2,3'

# region norm related
config.w_group_num = 299
config.h_group_num = 1

# eval related
config.test_network = "012"

# misc
config.batch_size = 20  # batch size for attack
config.report_step = 100
config.overwrite = False
config.img_num = 5000
config.img_val_num = 500

# data related
config.train_list_filename = 'data/list/train_list.txt'
config.test_list_filename = 'data/list/test_list.txt'
config.ground_truth_file = 'data/valid_gt.csv'
config.img_dir = 'data/val_data/'
config.checkpoint_path = os.path.join(os.path.dirname(__file__), 'data/checkpoints')
config.exp = 'RHP'
config.exp_train = '299x1-16'

parser = argparse.ArgumentParser(description='config')
for key, value in config.items():
    if type(value) is bool:
        parser.add_argument("--" + key, action='store_' + str(not value).lower())
    elif type(value) is list:
        parser.add_argument("--" + key, action='append', default=value, type=type(value[0]))
    else:
        parser.add_argument("--" + key, type=type(value), default=value)
args = parser.parse_args()
for key, value in args.__dict__.items():
    config[key] = value

network_pool = ["inception_v3", "inception_v4", "inception_resnet_v2"]  # 0, 1, 2
test_network_pool = ["ens3_inception_v3", "ens4_inception_v3", "ens_inception_resnet_v2"]  # 0, 1, 2

config.attack_networks = [
    network_pool[ord(index) - ord('a') + 10] if index >= 'a' else network_pool[int(index)] for index
    in config.attack_network]
config.test_networks = [
    test_network_pool[ord(index) - ord('a') + 10] if index >= 'a' else test_network_pool[int(index)]
    for index in config.test_network]

config.result_dir = '../result/{:s}_{:s}_att{:s}'.format(config.exp, config.exp_train,
                                                         config.attack_network)
config.RHP_savepath = '../ckpt/{:s}_att{:s}'.format(config.exp_train, config.attack_network)
if config.universal:
    config.result_dir += '_U'

if train_mode:
    if not os.path.exists(config.RHP_savepath):
        os.makedirs(config.RHP_savepath)
    else:
        assert config.overwrite, "{:s}".format(config.RHP_savepath)
elif attack_mode:
    if not os.path.exists(config.result_dir):
        os.makedirs(config.result_dir)
    else:
        assert config.overwrite, "{:s}".format(config.result_dir)
else:
    assert eval_mode

assert config.batch_size > 1
os.environ['CUDA_VISIBLE_DEVICES'] = config.GPU_ID
print(config)
