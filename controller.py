from core.generator import Generator
from core.shifter import Shifter
from core.projector import run_projection
from core.align_faces import align_face

from utils.helpers import convert_image, get_img_bits, load_numpy
from PIL import Image
from settings.config import Config

import numpy as np
import torch

class Controller:
    def __init__(self):
        self.generator = Generator()
        self.shifter = Shifter()
        self.w = None

    def generate_img_from_seed(self, seed, psi):
        """
        Generates bits of an image using seeded z vector and truncation psi.
        """
        # get z from seed
        z = torch.from_numpy(np.random.RandomState(seed).randn(1, self.generator.G.z_dim)).to(Config.generation.device)
        # get w from z
        self.w = self.generator.G.mapping(z, None)  # Map z to w
        # get img from w
        img = self.generator.G.synthesis(self.w, truncation_psi=psi, noise_mode='const')  # Generate image
        # convert image (use size=Config.gui.display_size)
        img = convert_image(img, size=Config.gui.display_size)  # Convert image to required size
        # convert image to bits
        img_bits = get_img_bits(img)  # Convert image to bits
        return img_bits

    def generate_img_from_z_vec(self, path, psi):
        """
        Generates bits of an image using a path to a z vector and truncation psi
        (also saves w vector).
        """
        # load z from path 
        z = load_numpy(path).to(Config.generation.device)
        # get w from z
        self.w = self.generator.G.mapping(z, None)  # Map z to w
        # get img from w
        img = self.generator.G.synthesis(self.w, truncation_psi=psi, noise_mode='const')  # Generate image
        # convert image (use size=Config.gui.display_size)
        img = convert_image(img, size=Config.gui.display_size)
        # convert image to bits
        img_bits = get_img_bits(img)
        return img_bits

    def generate_img_from_w_vec(self, path):
        """
        Generates bits of an image using a path to a w vector.
        """
        # load w from path
        self.w = load_numpy(path).to(Config.generation.device)
        # get img from w
        img = self.generator.G.synthesis(self.w, noise_mode='const')  # Generate image
        # convert image (use size=Config.gui.display_size)
        img = convert_image(img, size=Config.gui.display_size)
        # convert image to bits
        img_bits = get_img_bits(img)
        return img_bits

    def transform_img(self, directions, psi):
        """
        Transforms saved w vector and generates new transformed image bits.
        @param directions: dictionary where key is direction and value is shifting amount (i.e. 'age':5.0)
        @param psi: truncation psi value.
        @return: transformed image bits.
        """
        # perform truncation
        w_prime = self.w.clone()  # Start with the original w vector
        for direction, amount in directions.items():
            # shift
            w_prime = self.shifter.shift(w_prime, direction, amount)  # Apply shift for each direction
        # generate the image
        img = self.generator.G.synthesis(w_prime, truncation_psi=psi, noise_mode='const')
        # convert image (use size=Config.gui.display_size)
        img = convert_image(img, size=Config.gui.display_size)
        # convert image to bits
        img_bits = get_img_bits(img)
        return img_bits

    def read_path(self, path):
        """
        Read an image from a path, converts it to display size, and converts it to bits.
        """
        img = Image.open(path)
        img = img.resize(Config.gui.display_size)
        img_bits = get_img_bits(img)
        return img_bits

    def align(self, path):
        """
        Aligns the image using provided path and returns bits of aligned image and its path.
        """
        # align the image
        aligned_path = align_face(path)  # Align the face
        # read image bit from the path
        aligned_bits = self.read_path(aligned_path)  # Read the aligned image
        return aligned_bits, aligned_path

    def project(self, path):
        """
        Runs projection and returns bits of the generated image (also saves w vector).
        """
        # run projection
        img, self.w = run_projection(path, self.generator.G)
        # convert image (size = Config.gui.display_size)
        img = convert_image(img, size=Config.gui.display_size)
        # convert image to bits
        projected_bits = get_img_bits(img)
        return projected_bits