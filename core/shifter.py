import sys
import os
pypath = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(pypath)
sys.path.append(f'{pypath}/stylegan2-ada-pytorch')
import torch
from settings.config import Config,Generation
from utils.helpers import load_numpy  # assuming load_numpy is in helpers.py


class Shifter:
    def __init__(self, vectors_dir=Config.shifting.vectors_path, ext=Config.shifting.extension):
        # Get the list of vector files in the directory with the specified extension
        self.fnames = [file for file in os.listdir(vectors_dir) if file.endswith(ext)]
        self.vectors = {}
        
        # Load each vector and store in self.vectors
        for file in self.fnames:
            path = os.path.join(vectors_dir, file)
            name = file.replace(ext, '')
            
            # Load numpy vectors and pass them to the device specified in Config
            vec = load_numpy(path, device=Generation.device)
            
            # Unsqueeze to add a "batch" dimension
            vec = vec.unsqueeze(0)
            
            # Store the vector in the dictionary under the name (without extension)
            self.vectors[name] = vec

    def __call__(self, w, direction, amount):
        """
        Shifts latent vector w in the given direction
        @param w: input latent vector (shape [1, 18, 512])
        @param direction: name of a key in the vectors dictionary
        @param amount: scale factor for the direction (magnitude)
        @return: shifted latent vector
        """
        # Retrieve the latent direction vector by the direction key
        direction_vector = self.vectors[direction]
        
        # Perform the shift operation: w + latent_direction * amount
        w_shifted = w + direction_vector * amount
        
        return w_shifted
