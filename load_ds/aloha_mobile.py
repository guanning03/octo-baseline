import tensorflow as tf
import tensorflow_datasets as tfds
from utils import clean_task_instruction, quaternion_to_euler
import tensorflow as tf
import h5py
import numpy as np
from tqdm import tqdm
import os
import imageio
import concurrent.futures
import fnmatch
import cv2

def get_all_hdf5s(root_dir):
    num_files = 0
    for root, dirs, files in os.walk(root_dir):
        for filename in fnmatch.filter(files, '*.hdf5'):
            num_files += 1
    return num_files

def decode_img(img):
    return cv2.cvtColor(cv2.imdecode(np.frombuffer(img, np.uint8), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)

def dataset_generator(root_dir):
    for root, dirs, files in os.walk(root_dir):
        for filename in fnmatch.filter(files, '*.hdf5'):
            filepath = os.path.join(root, filename)
            with h5py.File(filepath, 'r') as f:
                action = f['action'][:]
                base_action = f['base_action'][:]
                qpos = f['observations']['qpos'][:]
                qvel = f['observations']['qvel'][:]
                cam_high = f['observations']['images']['cam_high'][:]
                cam_left_wrist = f['observations']['images']['cam_left_wrist'][:]
                cam_right_wrist = f['observations']['images']['cam_right_wrist'][:]
                instruction = f['instruction'][()]
                num_episodes = action.shape[0]
                steps = []
                for i in range(num_episodes):
                    step = {
                                'action': action[i],
                                'base_action':base_action[i],
                                'qvel': qvel[i],
                                'qpos': qpos[i],
                                'cam_high': decode_img(cam_high[i]),
                                'cam_left_wrist': decode_img(cam_left_wrist[i]),
                                'cam_right_wrist': decode_img(cam_right_wrist[i]),
                                'instruction': instruction,
                                'terminate_episode': i == (num_episodes - 1)
                            }
                    steps.append(step)

                steps_dataset = tf.data.Dataset.from_generator(
                            lambda: iter(steps),
                            output_signature={
                                'action': tf.TensorSpec(shape=(14,), dtype=tf.float32),
                                'base_action': tf.TensorSpec(shape=(2,), dtype=tf.float32),
                                'qpos': tf.TensorSpec(shape=(14,), dtype=tf.float32),
                                'qvel': tf.TensorSpec(shape=(14,), dtype=tf.float32),
                                'instruction':tf.TensorSpec(shape=(), dtype=tf.string),
                                'cam_high': tf.TensorSpec(shape=(None,None,3), dtype=tf.uint8),
                                'cam_left_wrist': tf.TensorSpec(shape=(None,None,3), dtype=tf.uint8),
                                'cam_right_wrist': tf.TensorSpec(shape=(None,None,3), dtype=tf.uint8),
                                'terminate_episode': tf.TensorSpec(shape=(), dtype=tf.bool),
                            }
                    )
                yield {'steps': steps_dataset}

def load_dataset():
    root_dir = '/data1/zhuxiaopei/aloha_mobile/aloha_mobile_shrimp_truncated/splited/'
    dataset = tf.data.Dataset.from_generator(
        lambda: dataset_generator(root_dir),
        output_signature={
            'steps': tf.data.DatasetSpec(
                element_spec = {
                    'action': tf.TensorSpec(shape=(14,), dtype=tf.float32),
                    'base_action': tf.TensorSpec(shape=(2,), dtype=tf.float32),
                    'qpos': tf.TensorSpec(shape=(14,), dtype=tf.float32),
                    'qvel': tf.TensorSpec(shape=(14,), dtype=tf.float32),
                    'instruction':tf.TensorSpec(shape=(), dtype=tf.string),
                    'cam_high': tf.TensorSpec(shape=(None,None,3), dtype=tf.uint8),
                    'cam_left_wrist': tf.TensorSpec(shape=(None,None,3), dtype=tf.uint8),
                    'cam_right_wrist': tf.TensorSpec(shape=(None,None,3), dtype=tf.uint8),
                    'terminate_episode': tf.TensorSpec(shape=(), dtype=tf.bool),
                }
            )
        }
    )
    return dataset

if __name__ == '__main__':
    dataset = load_dataset()
    print('successfully loaded dataset')
    print(type(dataset))
    for item in dataset.take(1):
        steps = item['steps']
        for idx,step in enumerate(steps):
            print(f'step {idx}:')
            print(step['action'].shape)
            print(step['base_action'])
            print(step['qpos'].shape)
            print(step['qvel'].shape)
            print(step['instruction'])
            print(step['cam_high'].shape)
            print(step['cam_left_wrist'].shape)
            print(step['cam_right_wrist'].shape)
            print(step['terminate_episode'])
            exit()