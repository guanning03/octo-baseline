import jax
import jax.numpy as jnp
import numpy as np
import json
import tensorflow as tf

from PIL import Image
import io
import importlib

def pytree_display(example: dict):
    def print_shape_or_value(x):
        if isinstance(x, (jnp.ndarray, np.ndarray, tf.Tensor)):
            return f"Shape: {x.shape}"
        else:
            return x
    def apply_to_nested_dict(func, d):
        if isinstance(d, dict):
            return {k: apply_to_nested_dict(func, v) for k, v in d.items()}
        else:
            return func(d)
    converted_tree = jax.tree_util.tree_map(print_shape_or_value, example)
    formatted_output = json.dumps(converted_tree, indent=4)
    print(formatted_output)

def dataset_display(dataset: tf.data.Dataset):
    for step in dataset.take(1):
        pytree_display(step)
        
def standardize_pytree(params):
    def print_shape_or_value(x):
        if isinstance(x, (jnp.ndarray, np.ndarray, tf.Tensor)):
            return f'Shape: {x.shape}'
        else:
            return x
    def apply_to_nested_dict(func, d):
        if isinstance(d, dict):
            return {k: apply_to_nested_dict(func, v) for k, v in d.items()}
        else:
            return func(d)
    converted_tree = jax.tree_util.tree_map(print_shape_or_value, params)
    formatted_output = json.dumps(converted_tree, indent=4)
    return formatted_output


def channel_transform(image):
    torch = importlib.util.find_spec("torch")
    if torch is not None:
        torch = importlib.import_module("torch")
    tf = importlib.util.find_spec("tensorflow")
    if tf is not None:
        tf = importlib.import_module("tensorflow")
    original_type = type(image)
    original_format = 'JPEG'  
    if isinstance(image, bytes):
        try:
            stream = io.BytesIO(image)
            image = Image.open(stream)
            original_format = image.format
        except Exception as e:
            raise ValueError("Failed to open image: " + str(e))
    if isinstance(image, Image.Image):
        image_np = np.array(image)
        image_np = image_np[..., [2, 1, 0]]
        image = Image.fromarray(image_np)
    elif isinstance(image, np.ndarray) or (torch and torch.is_tensor(image)) or (tf and isinstance(image, tf.Tensor)):
        if image.ndim in [3, 4, 5] and image.shape[-1] == 3:
            image = image[..., [2, 1, 0]]
        elif image.ndim == 3 and image.shape[0] == 3:
            image = image[[2, 1, 0], ...]
        elif image.ndim == 4 and image.shape[1] == 3:
            image = image[:, [2, 1, 0], ...]
        elif image.ndim == 5 and image.shape[2] == 3:
            image = image[:, :, [2, 1, 0], ...]
        else:
            raise ValueError("Image dimensions or channels do not match expected format.")
    if str(original_type) == "<class 'numpy.bytes_'>":
        print('arrive here')
        buffer = io.BytesIO()
        image.save(buffer, format=original_format)
        return buffer.getvalue()
    return image