import torch


class MatFac:
    def __init__(self, latent_factors):
        self.latent_factors = latent_factors

    def fit(self, ui_mat):
        self.X = torch.random.randn(ui_mat.shape[0], self.latent_factors)
        self.Y = torch.random.randn(ui_mat.shape[1], self.latent_factors)