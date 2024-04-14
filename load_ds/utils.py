import tensorflow as tf
from tensorflow_graphics.geometry.transformation.euler import from_quaternion
from tensorflow_graphics.geometry.transformation.quaternion import from_euler


def dataset_to_path(dataset_name: str, dir_name: str) -> str:
    """
    Return the path to the dataset.
    """
    if dataset_name == 'robo_net' or \
        dataset_name == 'cmu_playing_with_food' or \
        dataset_name == 'droid':
        version = '1.0.0'
    elif dataset_name == 'language_table':
        version = '0.0.1'
    elif dataset_name == 'nyu_door_opening_surprising_effectiveness':
        version = ''
    elif dataset_name == 'cmu_play_fusion':
        version=''
    elif dataset_name=='berkeley_gnm_recon':
        version=''
    else:
        version = '0.1.0'
    return f'{dir_name}/{dataset_name}/{version}'


def clean_task_instruction(
        task_instruction: tf.Tensor, replacements: dict) -> tf.Tensor:
    """
    Clean up the natural language task instruction.
    """
    # Create a function that applies all replacements
    def apply_replacements(tensor):
        for old, new in replacements.items():
            tensor = tf.strings.regex_replace(tensor, old, new)
        return tensor
    # Apply the replacements and strip leading and trailing spaces
    cleaned_task_instruction = apply_replacements(task_instruction)
    cleaned_task_instruction = tf.strings.strip(cleaned_task_instruction)
    return cleaned_task_instruction


def quaternion_to_euler(quaternion: tf.Tensor) -> tf.Tensor:
    """
    Convert a quaternion (x, y, z, w) to Euler angles.
    """
    # Normalize the quaternion
    quaternion = tf.nn.l2_normalize(quaternion, axis=-1)
    return from_quaternion(quaternion)


def euler_to_quaternion(euler: tf.Tensor) -> tf.Tensor:
    """
    Convert Euler angles (roll, pitch, yaw) to a quaternion (x, y, z, w).
    """
    # Convert roll, pitch, yaw to yaw, pitch, roll
    euler = tf.reverse(euler, axis=[-1])
    return from_euler(euler)


def transform_to_euler(R: tf.Tensor) -> tf.Tensor:
    """
    Convert a 4x4 homogeneous transformation matrix to Euler angles.
    """
    sy = tf.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
    singular = sy < 1e-6

    def compute_angles():
        yaw = tf.atan2(R[1, 0], R[0, 0])
        pitch = tf.atan2(-R[2, 0], sy)
        roll = tf.atan2(R[2, 1], R[2, 2])
        return tf.stack([roll, pitch, yaw])

    def handle_singular_case():
        yaw = tf.atan2(-R[1, 2], R[1, 1])
        pitch = tf.atan2(-R[2, 0], sy)
        roll = tf.constant(0.0)
        return tf.stack([roll, pitch, yaw])

    return tf.cond(singular, true_fn=handle_singular_case, false_fn=compute_angles)

