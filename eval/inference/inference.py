from PIL import Image
import requests
import matplotlib.pyplot as plt
import numpy as np
import jax
import jax.numpy as jnp
import tensorflow as tf
from io import BytesIO
import h5py
import os
import json
from octo.model.octo_model import OctoModel
from tqdm import tqdm


os.environ['CUDA_VISIBLE_DEVICES'] = '1'

HDF5_PATH_LIST = \
[f'/data1/zhuxiaopei/put_orange_paperbox_val/episode_{i}.hdf5' for i in range(42, 51)]
MODEL_PATH = '/data/zhuxiaopei/ckpt/octo_cobot/orange_lowlr_20240505_110550/'
STEP = 30000

def MSE(y_true, y_pred):
    return jnp.mean((y_true - y_pred) ** 2)

class OctoInput():
    
    def __init__(self, hdf5_path, statistics_path):
        self.hdf5_path = hdf5_path
        self.normalizer = Normalizer(statistics_path)
        self.instruction, self.qpos, self.image_primary, self.image_wrist_left, self.image_wrist_right, self.action\
            = self.make_frame_list()
        
    def make_frame_list(self) -> list:
        # 1. 加载hdf5
        with h5py.File(self.hdf5_path, 'r') as f:
            
            action = jnp.array(f['action'])
            self.length = len(action)
            qpos = jnp.array(f['observations']['qpos'])
            
            image_primary = np.stack([
                Image.open(BytesIO(
                    f['observations']['images']['cam_high'][i]
                )) for i in range(len(f['observations']['images']['cam_high']))
            ])[:,:,:,[2,1,0]]
            image_wrist_left = np.stack([
                Image.open(BytesIO(
                    f['observations']['images']['cam_left_wrist'][i]
                )) for i in range(len(f['observations']['images']['cam_left_wrist']))
            ])[:,:,:,[2,1,0]]
            image_wrist_right = np.stack([
                Image.open(BytesIO(
                    f['observations']['images']['cam_right_wrist'][i]
                )) for i in range(len(f['observations']['images']['cam_right_wrist']))
            ])[:,:,:,[2,1,0]]
            
            instruction = np.array(f['instruction'])[()].decode('utf-8')

        # 2. 将图片padding+resize
        image_primary = tf.image.resize_with_pad(image_primary, 256, 256)
        image_wrist_left = tf.image.resize_with_pad(image_wrist_left, 128, 128)
        image_wrist_right = tf.image.resize_with_pad(image_wrist_right, 128, 128)
        
        # 3. 加载统计值，对qpos进行归一化
        qpos = self.normalizer.preprocess_qpos(qpos)
        
        # 4. 转换数据格式
        image_primary = jnp.expand_dims(jnp.array(image_primary), axis = 0)
        image_wrist_left = jnp.expand_dims(jnp.array(image_wrist_left), axis = 0)
        image_wrist_right = jnp.expand_dims(jnp.array(image_wrist_right), axis = 0)
        qpos = jnp.expand_dims(qpos, axis = 0)
        
        # 5. 返回数据
        return instruction, qpos, image_primary, image_wrist_left, image_wrist_right, action
    
    def make_input_with_label(self, window_size, future_size) -> list[tuple]:
        input_list = []
        label_list = []
        pad_mask = []
        W = window_size + future_size
        for i in range(0, self.length - W):
            pad_mask = jnp.expand_dims(jnp.array([True for _ in range(window_size)]), axis = 0)
            input_list.append({
                'proprio': self.qpos[:,i:i+window_size],
                'image_primary': self.image_primary[:,i:i+window_size],
                'image_wrist_left': self.image_wrist_left[:,i:i+window_size],
                'image_wrist_right': self.image_wrist_right[:,i:i+window_size],
                'pad_mask': pad_mask
            })
            label_list.append(self.action[i+window_size:i+W])
        return self.instruction, input_list, label_list
    
class Normalizer():
    
    def __init__(self, statistics_path):
        self.statistics_path = statistics_path
        with open(statistics_path, 'r') as f:
            self.statistics = json.load(f)
        self.qpos_std = jnp.array(self.statistics['proprio']['std'])
        self.qpos_mean = jnp.array(self.statistics['proprio']['mean'])
        self.action_std = jnp.array(self.statistics['action']['std'])
        self.action_mean = jnp.array(self.statistics['action']['mean'])
        
    def preprocess_qpos(self, qpos):
        qpos_mean = jnp.broadcast_to(self.qpos_mean, qpos.shape)
        qpos_std = jnp.broadcast_to(self.qpos_std, qpos.shape)
        return (qpos - qpos_mean) / qpos_std
    
    def postprocess_action(self, raw_action):
        action_mean = jnp.broadcast_to(self.action_mean, raw_action.shape)
        action_std = jnp.broadcast_to(self.action_std, raw_action.shape)
        return raw_action * action_std + action_mean
    
if __name__ == '__main__':
    model = OctoModel.load_pretrained(MODEL_PATH, STEP)
    print(f'model loaded from {MODEL_PATH} at step {STEP}')
    total_mse = 0
    total_cnt = 0
    for hdf5 in HDF5_PATH_LIST:
        tot_mse = 0
        tot_cnt = 0
        octo_input = OctoInput(hdf5_path=hdf5, \
            statistics_path='/data1/zhuxiaopei/put_orange_paperbox_train/dataset_statistics.json')
        instruction, input_list, label_list = octo_input.make_input_with_label(window_size=2, future_size=4)
        print(f'hdf5: {hdf5.rsplit("/", 1)[-1]}, "{instruction}"')
        tasks = model.create_tasks(texts = [instruction])
        # tasks = model.create_tasks(texts = ['Pick up the minoral water bottle and place it on the laptop.'])
        for input, label in tqdm(zip(input_list, label_list), desc = f'inferring {hdf5.rsplit("/", 1)[-1]}'):
            tot_cnt += 1
            total_cnt += 1
            prediction = model.sample_actions(observations = input, tasks = tasks, rng=jax.random.PRNGKey(0))
            pred_action = octo_input.normalizer.postprocess_action(prediction)[0]
            # print('pred_action:', pred_action)
            # print('label:', label)
            tot_mse += MSE(pred_action, label)
        print('MSE:', tot_mse / tot_cnt)
        total_mse += tot_mse
    print('AVG MSE:', total_mse / total_cnt)