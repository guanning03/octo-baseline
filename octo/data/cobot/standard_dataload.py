import tensorflow as tf
import tensorflow_datasets as tfds
from ..utils.load_utils import clean_task_instruction, quaternion_to_euler
from ..utils.format import pytree_display
import tensorflow as tf
import h5py
import numpy as np
from tqdm import tqdm
import os
import imageio
import concurrent.futures
import fnmatch
from dlimp.dataset import _wrap
import cv2
import sys

def load_dataset_from_hdf5(hdf5_path):
    def load_and_parse(filename):
        with h5py.File(filename, 'r') as f:
            # Read and preprocess each component of the dataset
            action = np.array(f['action'][:])
            base_action = np.array(f['base_action'][:])
            qpos = np.array(f['observations']['qpos'][:])
            qvel = np.array(f['observations']['qvel'][:])
            cam_high = np.array([x.tobytes() for x in f['observations']['images']['cam_high'][:]])
            cam_left_wrist = np.array([x.tobytes() for x in f['observations']['images']['cam_left_wrist'][:]])
            cam_right_wrist = np.array([x.tobytes() for x in f['observations']['images']['cam_right_wrist'][:]])
            instruction = np.array(f['instruction'][()].decode('utf-8'))
            num_episodes = action.shape[0]

            # Generate terminate_episode flags
            terminate_episode = [bool(i == num_episodes - 1) for i in range(num_episodes)]

            # Create a TensorFlow dataset from numpy arrays
            dataset = tf.data.Dataset.from_tensor_slices({
                "action": action,
                "base_action": base_action,
                "qpos": qpos,
                "qvel": qvel,
                "observation": {
                    "cam_high": cam_high,
                    "cam_left_wrist": cam_left_wrist,
                    "cam_right_wrist": cam_right_wrist
                },
                "instruction": [instruction] * num_episodes,  # Broadcast instruction to match episodes
                "terminate_episode": terminate_episode
            })

            # Batch the entire dataset from one file
            return dataset.batch(10**10)

    all_files = []
    all_datasets = []
    
    for root, _, files in os.walk(hdf5_path):
        all_files.extend([os.path.expanduser(os.path.join(root, filename)) 
                          for filename in fnmatch.filter(files, '*.hdf5')])

    for filename in tqdm(all_files, desc = "Loading HDF5 files"):
            filepath = os.path.join(root, filename)
            dataset = load_and_parse(filepath)
            all_datasets.append(dataset)

    # Use flat_map to flatten all datasets into one
    complete_dataset = tf.data.Dataset.from_tensor_slices(all_datasets).flat_map(lambda x: x)

    return complete_dataset

def load_dataset_from_tfrecords(tfrecord_path):
    def _parse_function(proto):
        keys_to_features = {
            'action': tf.io.FixedLenFeature([], tf.string),
            'base_action': tf.io.FixedLenFeature([], tf.string),
            'qpos': tf.io.FixedLenFeature([], tf.string),
            'qvel': tf.io.FixedLenFeature([], tf.string),
            'cam_high': tf.io.FixedLenFeature([], tf.string),
            'cam_left_wrist': tf.io.FixedLenFeature([], tf.string),
            'cam_right_wrist': tf.io.FixedLenFeature([], tf.string),
            'instruction': tf.io.FixedLenFeature([], tf.string),
            # 'terminate_episode': tf.io.FixedLenFeature([], tf.int64)
        }

        parsed_features = tf.io.parse_single_example(proto, keys_to_features)
        
        ret = {
            "action": tf.io.parse_tensor(parsed_features['action'], out_type=tf.float32),
            "base_action": tf.io.parse_tensor(parsed_features['base_action'], out_type=tf.float32),
            "qpos": tf.io.parse_tensor(parsed_features['qpos'], out_type=tf.float32),
            "qvel": tf.io.parse_tensor(parsed_features['qvel'], out_type=tf.float32),
            "observation": {
                "cam_high": parsed_features['cam_high'],
                "cam_left_wrist": parsed_features['cam_left_wrist'],
                "cam_right_wrist": parsed_features['cam_right_wrist']
            },
            "instruction": parsed_features['instruction'],
            # "terminate_episode": tf.cast(parsed_features['terminate_episode'], tf.int64)
        }
        
        num_of_episodes = tf.shape(ret['action'])[0]
        terminate_episode = tf.concat([tf.zeros(num_of_episodes - 1, dtype=tf.int64), tf.ones(1, dtype=tf.int64)], axis=0)
        ret["terminate_episode"] = terminate_episode

        return ret
        
    def load_and_parse(filename):
        raw_dataset = tf.data.TFRecordDataset(filename)
        parsed_dataset = raw_dataset.map(_parse_function).batch(10**10)
        return parsed_dataset

    all_files = []
    all_datasets = []
    
    for root, _, files in os.walk(tfrecord_path):
        all_files.extend([os.path.expanduser(os.path.join(root, filename)) 
                          for filename in fnmatch.filter(files, '*.tfrecord')])
        
    for filename in tqdm(all_files, desc = "Loading TFRecord files"):
            filepath = os.path.join(root, filename)
            dataset = load_and_parse(filepath)
            all_datasets.append(dataset)

    # 使用 flat_map 来扁平化处理所有数据集
    complete_dataset = tf.data.Dataset.from_tensor_slices(all_datasets).flat_map(lambda x: x)

    return complete_dataset

