from PIL import Image
import requests
import matplotlib.pyplot as plt
import numpy as np
import jax
import jax.numpy as jnp
import tensorflow as tf
import h5py

### TODO: write a function to transform the hdf5 file to the form as a octo input

class OctoInput():
    
    def __init__(self, hdf5_path, statistics_path):
        self.hdf5_path = hdf5_path
        self.normalizer = Normalizer(statistics_path)
        
    def make_frame_list(self):
        pass
    
    def make_input_with_label(self, window_size, future_size):
        pass
    
class Normalizer():
    
    def __init__(self, statistics_path):
        self.statistics_path = statistics_path
        
    def preprocess_qpos(self, qpos):
        pass
    
    def postprocess_action(self, raw_action):
        pass