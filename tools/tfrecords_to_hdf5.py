import tensorflow as tf
import h5py
import numpy as np
import os

def _parse_function(proto):
    # Define your tfrecord again. It is normal to define it twice
    keys_to_features = {
        'action': tf.io.FixedLenFeature([], tf.string),
        'base_action': tf.io.FixedLenFeature([], tf.string),
        'qpos': tf.io.FixedLenFeature([], tf.string),
        'qvel': tf.io.FixedLenFeature([], tf.string),
        'cam_high': tf.io.FixedLenFeature([], tf.string),
        'cam_left_wrist': tf.io.FixedLenFeature([], tf.string),
        'cam_right_wrist': tf.io.FixedLenFeature([], tf.string),
        'instruction': tf.io.FixedLenFeature([], tf.string),
        # 'terminate_episode': tf.io.FixedLenFeature([], tf.bool)
    }
    
    # Load one example
    parsed_features = tf.io.parse_single_example(proto, keys_to_features)
    
    # Turn your parsed tensors into their original shape
    parsed_features['action'] = tf.io.parse_tensor(parsed_features['action'], out_type=tf.float32)
    parsed_features['base_action'] = tf.io.parse_tensor(parsed_features['base_action'], out_type=tf.float32)
    parsed_features['qpos'] = tf.io.parse_tensor(parsed_features['qpos'], out_type=tf.float32)
    parsed_features['qvel'] = tf.io.parse_tensor(parsed_features['qvel'], out_type=tf.float32)
    # Images and strings do not need to be reshaped.
    return parsed_features

def read_tfrecords(tfrecords_path):
    # Create a dataset from the TFRecord file
    dataset = tf.data.TFRecordDataset([tfrecords_path])
    # Map the parser over the dataset
    parsed_dataset = dataset.map(_parse_function)
    return parsed_dataset

def convert_tfrecords_to_hdf5(tfrecords_dir, hdf5_output_dir):
    if not os.path.exists(hdf5_output_dir):
        os.makedirs(hdf5_output_dir)
    
    # Iterate through all tfrecord files
    for root, dirs, files in os.walk(tfrecords_dir):
        for filename in files:
            if filename.endswith('.tfrecord'):
                tfrecord_path = os.path.join(root, filename)
                parsed_dataset = read_tfrecords(tfrecord_path)
                
                hdf5_filename = filename.replace('.tfrecord', '.hdf5')
                hdf5_path = os.path.join(hdf5_output_dir, hdf5_filename)
                
                with h5py.File(hdf5_path, 'w') as hdf5_file:
                    # Initialize lists to collect the data
                    actions = []
                    base_actions = []
                    qpos_list = []
                    qvel_list = []
                    cam_highs = []
                    cam_left_wrists = []
                    cam_right_wrists = []
                    instructions = []
                    
                    for features in parsed_dataset:
                        actions.append(features['action'].numpy())
                        base_actions.append(features['base_action'].numpy())
                        qpos_list.append(features['qpos'].numpy())
                        qvel_list.append(features['qvel'].numpy())
                        cam_highs.append(np.frombuffer(features['cam_high'].numpy(), dtype=np.uint8))
                        cam_left_wrists.append(np.frombuffer(features['cam_left_wrist'].numpy(), dtype=np.uint8))
                        cam_right_wrists.append(np.frombuffer(features['cam_right_wrist'].numpy(), dtype=np.uint8))
                        instructions.append(features['instruction'].numpy().decode('utf-8'))
                    
                    # Create datasets for each group
                    hdf5_file.create_dataset('action', data=np.array(actions))
                    hdf5_file.create_dataset('base_action', data=np.array(base_actions))
                    obs_group = hdf5_file.create_group('observations')
                    obs_group.create_dataset('qpos', data=np.array(qpos_list))
                    obs_group.create_dataset('qvel', data=np.array(qvel_list))
                    images_group = obs_group.create_group('images')
                    images_group.create_dataset('cam_high', data=np.array(cam_highs))
                    images_group.create_dataset('cam_left_wrist', data=np.array(cam_left_wrists))
                    images_group.create_dataset('cam_right_wrist', data=np.array(cam_right_wrists))
                    hdf5_file.create_dataset('instruction', data=np.array(instructions, dtype='S'))

                    print(f"Converted {tfrecord_path} to {hdf5_path}")
                    
import argparse
parser = argparse.ArgumentParser(description='Convert tfrecords files to hdf5')
parser.add_argument('--input_dir', type=str, help='Root directory containing tfrecords files')
parser.add_argument('--output_dir', type=str, help='Output directory for hdf5 files')
args = parser.parse_args()
root_dir = args.input_dir
output_dir = args.output_dir

convert_tfrecords_to_hdf5(root_dir, output_dir)
