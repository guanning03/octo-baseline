import tensorflow as tf
import tensorflow_datasets as tfds
from ..utils.load_utils import clean_task_instruction, quaternion_to_euler
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

def decode_all_imgs(imgs):
    return [decode_img(img) for img in imgs]

def dataset_generator():
    root_dir = '/mnt/d/aloha/aloha_mobile/'
    dsets = []
    # if os.path.exists(os.path.join(root_dir, 'dataset')):
    #     print(f"directly loading from {root_dir}")
    #     return datasets.load_from_disk(os.path.join(root_dir, 'dataset'))
    for root, dirs, files in os.walk(root_dir):
        for filename in fnmatch.filter(files, '*.hdf5'):
            filepath = os.path.join(root, filename)
            with h5py.File(filepath, 'r') as f:
                action = f['action'][:]
                base_action = f['base_action'][:]
                qpos = f['observations']['qpos'][:]
                qvel = f['observations']['qvel'][:]
                cam_high = decode_all_imgs(f['observations']['images']['cam_high'][:])
                cam_left_wrist = decode_all_imgs(f['observations']['images']['cam_left_wrist'][:])
                cam_right_wrist = decode_all_imgs(f['observations']['images']['cam_right_wrist'][:])
                instruction = f['instruction'][()]
                num_episodes = action.shape[0]
            steps = []
            for i in range(num_episodes):
                step = {
                            'action': action[i],
                            'base_action': base_action[i],
                            'qvel': qvel[i],
                            'qpos': qpos[i],
                            'cam_high': cam_high[i],
                            'cam_left_wrist': cam_left_wrist[i],
                            'cam_right_wrist': cam_right_wrist[i],
                            'instruction': instruction,
                            'terminate_episode': i == num_episodes - 1
                        }
                steps.append(step)
            yield({'steps': [steps]})
            # save_dir = os.path.join('./temp', filename.split('.')[0])
            # dset = Dataset.from_dict({'steps': [steps]}).save_to_disk(save_dir)
            # print(f"Loaded {filename}")
            # dset = datasets.load_from_disk(save_dir)
            # dsets.append(dset)
            # dsets = datasets.concatenate_datasets(dsets)
            # # save the dataset
            # dsets.save_to_disk(os.path.join(root_dir, 'dataset'))

def stash_image_into_observation(step):
    step['observation'] = {'cam_high': [], 'cam_left_wrist': [], 'cam_right_wrist':[]}
    step['observation']['cam_high'] = step['cam_high']
    step['observation']['cam_left_wrist'] = step['cam_left_wrist']
    step['observation']['cam_right_wrist'] = step['cam_right_wrist']
    return step

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
        'terminate_episode': tf.io.FixedLenFeature([], tf.int64)
    }

    parsed_features = tf.io.parse_single_example(proto, keys_to_features)

    action = tf.io.parse_tensor(parsed_features['action'], out_type=tf.float32)
    base_action = tf.io.parse_tensor(parsed_features['base_action'], out_type=tf.float32)
    qpos = tf.io.parse_tensor(parsed_features['qpos'], out_type=tf.float32)
    qvel = tf.io.parse_tensor(parsed_features['qvel'], out_type=tf.float32)
    cam_high = tf.io.parse_tensor(parsed_features['cam_high'], out_type=tf.uint8)
    cam_left_wrist = tf.io.parse_tensor(parsed_features['cam_left_wrist'], out_type=tf.uint8)
    cam_right_wrist = tf.io.parse_tensor(parsed_features['cam_right_wrist'], out_type=tf.uint8)
    instruction = parsed_features['instruction']
    terminate_episode = tf.cast(parsed_features['terminate_episode'], tf.int64)
    action = tf.reshape(action, [14])
    base_action = tf.reshape(base_action, [2])
    qpos = tf.reshape(qpos, [14])
    qvel = tf.reshape(qvel, [14])
    cam_high = tf.reshape(cam_high, [480, 640, 3])
    cam_left_wrist = tf.reshape(cam_left_wrist, [480, 640, 3])
    cam_right_wrist = tf.reshape(cam_right_wrist, [480, 640, 3])

    return {
        "action": action,
        "base_action": base_action,
        "qpos": qpos,
        "qvel": qvel,
        'observation':{
        "cam_high": cam_high,
        "cam_left_wrist": cam_left_wrist,
        "cam_right_wrist": cam_right_wrist
        },
        "instruction": instruction,
        "terminate_episode": terminate_episode
    }

def dataset_generator_from_tfrecords(tfrecord_path):
    for root, dirs, files in os.walk(tfrecord_path):
        for filename in fnmatch.filter(files, '*.tfrecord'):
            filepath = os.path.join(root, filename)
            raw_dataset = tf.data.TFRecordDataset(filepath)
            dataset = raw_dataset.map(_parse_function)
            yield {
                'steps': dataset
            }

def load_dataset(path):
    dataset = tf.data.Dataset.from_generator(
        lambda: dataset_generator_from_tfrecords(path),
        output_signature={
            'steps': tf.data.DatasetSpec(
                element_spec={
                    'action': tf.TensorSpec(shape=(14), dtype=tf.float32),
                    'base_action': tf.TensorSpec(shape=(2), dtype=tf.float32),
                    'qpos': tf.TensorSpec(shape=(14), dtype=tf.float32),
                    'qvel': tf.TensorSpec(shape=(14), dtype=tf.float32),
                    'observation': {
                        'cam_high': tf.TensorSpec(shape=(480, 640, 3), dtype=tf.uint8),
                        'cam_left_wrist': tf.TensorSpec(shape=(480, 640, 3), dtype=tf.uint8),
                        'cam_right_wrist': tf.TensorSpec(shape=(480, 640, 3), dtype=tf.uint8),
                    },
                    'instruction': tf.TensorSpec(shape=(), dtype=tf.string),
                    'terminate_episode': tf.TensorSpec(shape=(), dtype=tf.int64)
                }
            )
        }
    )

    return dataset

def terminate_act_to_bool(terminate_act: tf.Tensor) -> tf.Tensor:
    """
    Convert terminate action to a boolean, where True means terminate.
    """
    return tf.where(tf.equal(terminate_act, tf.constant(0.0, dtype=tf.float32)),tf.constant(False),tf.constant(True))


def process_step(step: dict) -> dict:
    """
    Unify the action format and clean the task instruction.

    DO NOT use python list, use tf.TensorArray instead.
    """
    # Convert raw action to our action
    old_action = step['action']
    step['action'] = {}
    action = step['action']
    step['action']['terminate'] = step['terminate_episode']
    # act-plus-plus/utils.py at main · MarkFzp/act-plus-plus
    left_eef_delta_pos = old_action[:3]
    left_eef_delta_ang = quaternion_to_euler(old_action[3:7])
    right_eef_delta_pos = old_action[7:10]
    right_eef_delta_ang = quaternion_to_euler(old_action[10:14])

    base_delta_pos = step['base_action'][:1]
    base_delta_ang = step['base_action'][1:]
    base_action = tf.concat([base_delta_pos, base_delta_ang], axis=0)
    # # No base found
    # # Concatenate the action
    arm_action = tf.concat([left_eef_delta_pos, left_eef_delta_ang, right_eef_delta_pos, right_eef_delta_ang], axis=0)
    action['arm_concat'] = arm_action
    action['base_concat'] = base_action
    # # Write the action format
    action['format'] = tf.constant(
        "left_eef_delta_pos_x,left_eef_delta_pos_y,left_eef_delta_pos_z,left_eef_delta_angle_roll,left_eef_delta_angle_pitch,left_eef_delta_angle_yaw,right_eef_delta_pos_x,right_eef_delta_pos_y,right_eef_delta_pos_z,right_eef_delta_angle_roll,right_eef_delta_angle_pitch,right_eef_delta_angle_yaw,base_vel_y,base_angular_vel")

    state = step['observation']
    left_qpos = step['qpos'][:6]
    left_gripper_open = step['qpos'][6:7]
    right_qpos = step['qpos'][7:13]
    right_gripper_open = step['qpos'][13:14]
    left_qvel = step['qvel'][:6]
    left_gripper_joint_vel = step['qvel'][6:7]
    right_qvel = step['qvel'][7:13]
    right_gripper_joint_vel = step['qvel'][13:14]

    state['arm_concat'] = tf.concat([left_qpos, left_qvel, left_gripper_open, right_qpos, right_qvel, right_gripper_open,left_gripper_joint_vel,right_gripper_joint_vel], axis=0)
    # # Write the state format
    state['format'] = tf.constant(
        "left_arm_joint_0_pos,left_arm_joint_1_pos,left_arm_joint_2_pos,left_arm_joint_3_pos,left_arm_joint_4_pos,left_arm_joint_5_pos,left_arm_joint_0_vel,left_arm_joint_1_vel,left_arm_joint_2_vel,left_arm_joint_3_vel,left_arm_joint_4_vel,left_arm_joint_5_vel,left_gripper_open,right_arm_joint_0_pos,right_arm_joint_1_pos,right_arm_joint_2_pos,right_arm_joint_3_pos,right_arm_joint_4_pos,right_arm_joint_5_pos,right_arm_joint_0_vel,right_arm_joint_1_vel,right_arm_joint_2_vel,right_arm_joint_3_vel,right_arm_joint_4_vel,right_arm_joint_5_vel,right_gripper_open,left_gripper_joint_0_vel,right_gripper_joint_0_vel")

    # Clean the task instruction
    # Define the replacements (old, new) as a dictionary
    replacements = {
        '_': ' ',
        '1f': ' ',
        '4f': ' ',
        '-': ' ',
        '50': ' ',
        '55': ' ',
        '56': ' ',
        ';': ' '    # Separtor used in outer format
    }
    instr = step['instruction']
    instr = clean_task_instruction(instr, replacements)
    step['observation']['natural_language_instruction'] = instr

    return step


# if __name__ == "__main__":
#     import tensorflow_datasets as tfds
#     from data.openx_embod.utils import dataset_to_path

#     DATASET_DIR = '/mnt/d/aloha/'
#     DATASET_NAME = 'dataset'
#     # Load the dataset
#     dataset = load_dataset()
#     for data in dataset.take(1):
#         for step in data['steps'].take(1):
#             from IPython import embed; embed()
#             print(step)