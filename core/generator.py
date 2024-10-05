import sys
import os
pypath = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(pypath)
sys.path.append(f'{pypath}/stylegan2-ada-pytorch')

import torch
import pickle
import numpy as np
from settings.config import Config




class Generator:
    def __init__(self, pickle_path=Config.generation.model_path, device=Config.generation.device):
        self.device = device
        with open(pickle_path, 'rb') as f:
            self.G = pickle.load(f)['G_ema'].to(device)
            self.G.eval()

    def truncate_w(self, w, truncation_psi=1):
        """
        Performs linear interpolation between a given w and the average w vector.
        truncation_psi=1 means no truncation.
        """
        w_avg = self.G.mapping.w_avg
        # Perform truncation by linearly interpolating between w and w_avg
        w = w_avg + (w - w_avg) * truncation_psi
        return w

    def get_z(self, seed):
        """
        Generates latent vector z from a random seed.
        """
        z = np.random.RandomState(seed).randn(1, self.G.z_dim)
        return z

    def get_w(self, z, truncation_psi=1):
        """
        Generates w vector using latent vector z.
        """
        z = torch.tensor(z, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            # Get w using G.mapping(z, None)
            w = self.G.mapping(z, None)
            # Perform truncation
            w = self.truncate_w(w, truncation_psi)
        return w

    def get_img(self, w):
        """
        Generates image using latent vector w.
        """
        w = torch.tensor(w).to(self.device)  # Ensure w is a torch tensor and on the correct device
        with torch.no_grad():
            img = self.G.synthesis(w, noise_mode='const', force_fp32=True)[0]
        return img
