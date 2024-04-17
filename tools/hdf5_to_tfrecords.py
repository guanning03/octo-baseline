import tensorflow as tf
import h5py
import os
import fnmatch
import cv2
import numpy as np
from tqdm import tqdm

# def decode_img(img):
#     return cv2.cvtColor(cv2.imdecode(np.frombuffer(img, np.uint8), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)

# def decode_all_imgs(imgs):
#     return [decode_img(img) for img in imgs]

def _bytes_feature(value):
    """返回一个bytes_list."""
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()  # BytesList 不会从 EagerTensor 中解包一个字符串。
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def serialize_example(action, base_action, qpos, qvel, cam_high, cam_left_wrist, cam_right_wrist, instruction):
    feature = {
        'action': _bytes_feature(tf.io.serialize_tensor(action)),
        'base_action': _bytes_feature(tf.io.serialize_tensor(base_action)),
        'qpos': _bytes_feature(tf.io.serialize_tensor(qpos)),
        'qvel': _bytes_feature(tf.io.serialize_tensor(qvel)),
        'cam_high': _bytes_feature(cam_high),
        'cam_left_wrist': _bytes_feature(cam_left_wrist),
        'cam_right_wrist': _bytes_feature(cam_right_wrist),
        'instruction': _bytes_feature(instruction),
        # 'terminate_episode': _bool_feature(terminate_episode)
    }
    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()

def write_tfrecords(root_dir, out_dir):
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    for root, dirs, files in os.walk(root_dir):
        for filename in tqdm(fnmatch.filter(files, '*.hdf5'), desc="处理文件"):
            filepath = os.path.join(root, filename)
            with h5py.File(filepath, 'r') as f:
                output_dir = os.path.join(out_dir, os.path.relpath(root, root_dir))
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                print(f"正在写入TFRecords到 {output_dir}")
                tfrecord_path = os.path.join(output_dir, filename.replace('.hdf5', '.tfrecord'))
                with tf.io.TFRecordWriter(tfrecord_path) as writer:
                    num_episodes = f['action'].shape[0]
                    for i in range(num_episodes):
                        action = f['action'][i]
                        base_action = f['base_action'][i]
                        qpos = f['observations']['qpos'][i]
                        qvel = f['observations']['qvel'][i]
                        cam_high = f['observations']['images']['cam_high'][i].tobytes()
                        cam_left_wrist = f['observations']['images']['cam_left_wrist'][i].tobytes()
                        cam_right_wrist = f['observations']['images']['cam_right_wrist'][i].tobytes()
                        instruction = f['instruction'][()]
                        # terminate_episode = i == num_episodes - 1
                        serialized_example = serialize_example(action, base_action, qpos, qvel, cam_high, cam_left_wrist, cam_right_wrist, instruction)
                        writer.write(serialized_example)
                    print(f"TFRecords已写入 {tfrecord_path}")
    print(f"所有TFRecords已写入 {out_dir}")

import argparse

parser = argparse.ArgumentParser(description='Convert hdf5 files to tfrecords')
parser.add_argument('--input_dir', type=str, help='Root directory containing hdf5 files')
parser.add_argument('--output_dir', type=str, help='Output directory for tfrecords')
root_dir = parser.parse_args().input_dir
output_dir = parser.parse_args().output_dir

write_tfrecords(root_dir, output_dir)