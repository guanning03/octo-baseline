from octo.data.utils.format import channel_transform
import dlimp as dl
import tensorflow as tf
from typing import Tuple

def standardize_fn(traj: dict) -> dict:
    traj['observation']['qpos'] = traj['qpos']
    traj['observation']['qvel'] = traj['qvel']
    del traj['qpos']
    del traj['qvel']
    for key in traj['observation']:
        if key.startswith('image_'):
            for i in range(len(traj['observation'][key])):
                traj['observation'][key][i] = channel_transform(traj['observation'][key][i])
    return traj
    
def resize_image_with_padding(image: tf.Tensor, size: Tuple[int, int]) -> tf.Tensor:
    assert image.dtype == tf.uint8, "Input image tensor should be of type uint8"
    original_height, original_width = tf.shape(image)[0], tf.shape(image)[1]
    target_height, target_width = size

    target_aspect_ratio = tf.cast(target_width, tf.float32) / tf.cast(target_height, tf.float32)
    image_aspect_ratio = tf.cast(original_width, tf.float32) / tf.cast(original_height, tf.float32)

    if image_aspect_ratio < target_aspect_ratio:
        # Image is taller, pad width
        new_width = tf.cast(tf.round(tf.cast(original_height, tf.float32) * target_aspect_ratio), tf.int32)
        pad_width = (new_width - original_width) // 2
        pad_height = 0
        image = tf.image.pad_to_bounding_box(
            image, 0, pad_width, original_height, new_width
        )
    else:
        # Image is wider, pad height
        new_height = tf.cast(tf.round(tf.cast(original_width, tf.float32) / target_aspect_ratio), tf.int32)
        pad_height = (new_height - original_height) // 2
        pad_width = 0
        image = tf.image.pad_to_bounding_box(
            image, pad_height, 0, new_height, original_width
        )
    image = dl.transforms.resize_image(image, size=size)
    return image