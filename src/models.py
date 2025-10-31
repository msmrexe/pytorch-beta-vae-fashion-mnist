import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):
    """
    Encoder module for the β-VAE.
    Maps input images to a latent space distribution (mean and log-variance).
    """
    def __init__(self, latent_dim: int = 10):
        """
        Initializes the Encoder.

        Args:
            latent_dim (int): Dimension of the latent space.
        """
        super(Encoder, self).__init__()

        self.conv1 = nn.Conv2d(1, 8, kernel_size=3, stride=2, padding=1)  # 28x28 -> 14x14
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=1) # 14x14 -> 7x7
        self.bn2 = nn.BatchNorm2d(16)
        self.conv3 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=0) # 7x7 -> 3x3 (output: 32x3x3)
        
        # Calculate flattened size after convolutions
        # Input size (1, 28, 28)
        # Conv1: (8, 14, 14)
        # Conv2: (16, 7, 7)
        # Conv3: (32, 3, 3)
        self.flattened_size = 32 * 3 * 3 
        self.fc = nn.Linear(self.flattened_size, 128)

        self.fc_mu = nn.Linear(128, latent_dim)
        self.fc_logvar = nn.Linear(128, latent_dim)

    def forward(self, x: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        """
        Forward pass for the Encoder.

        Args:
            x (torch.Tensor): Input image tensor.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Mean (mu) and log-variance (logvar) of the latent distribution.
        """
        x = F.relu(self.conv1(x))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.conv3(x))
        x = torch.flatten(x, start_dim=1) # Flatten for FC layer
        x = F.relu(self.fc(x))

        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)

        return mu, logvar


class Decoder(nn.Module):
    """
    Decoder module for the β-VAE.
    Maps latent space vectors back to image space.
    """
    def __init__(self, latent_dim: int = 10):
        """
        Initializes the Decoder.

        Args:
            latent_dim (int): Dimension of the latent space.
        """
        super(Decoder, self).__init__()

        self.fc1 = nn.Linear(latent_dim, 128)
        # This matches the flattened size from the encoder before mu/logvar layers
        self.fc2 = nn.Linear(128, 32 * 3 * 3) 
        
        # ConvTranspose2d layers for upsampling
        # Input: (batchsize, 32, 3, 3)
        self.deconv1 = nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=0) # Output: 16x7x7
        self.bn1 = nn.BatchNorm2d(16)
        self.deconv2 = nn.ConvTranspose2d(16, 8, kernel_size=3, stride=2, padding=1, output_padding=1) # Output: 8x14x14
        self.deconv3 = nn.ConvTranspose2d(8, 1, kernel_size=3, stride=2, padding=1, output_padding=1) # Output: 1x28x28

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the Decoder.

        Args:
            z (torch.Tensor): Latent space vector.

        Returns:
            torch.Tensor: Reconstructed image tensor.
        """
        x = F.relu(self.fc1(z))
        x = F.relu(self.fc2(x))
        x = x.view(-1, 32, 3, 3) # Reshape to (batch_size, 32, 3, 3) for deconv layers
        
        x = F.relu(self.bn1(self.deconv1(x)))
        x = F.relu(self.deconv2(x))
        x = torch.sigmoid(self.deconv3(x)) # Sigmoid for pixel values between 0 and 1

        return x


class BetaVAE(nn.Module):
    """
    β-Variational Autoencoder (β-VAE) model.
    Combines an Encoder and a Decoder with a reparameterization trick
    to learn disentangled latent representations.
    """
    def __init__(self, latent_dim: int = 10, beta: float = 1.0):
        """
        Initializes the BetaVAE model.

        Args:
            latent_dim (int): Dimension of the latent space.
            beta (float): Weight for the KL divergence term in the ELBO loss.
                          Higher beta encourages disentanglement.
        """
        super(BetaVAE, self).__init__()

        self.encoder = Encoder(latent_dim)
        self.decoder = Decoder(latent_dim)
        self.beta = beta
        self.latent_dim = latent_dim # Store for convenient access during generation

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        Performs the reparameterization trick to sample from the latent space.

        Args:
            mu (torch.Tensor): Mean of the latent Gaussian distribution.
            logvar (torch.Tensor): Log-variance of the latent Gaussian distribution.

        Returns:
            torch.Tensor: Sampled latent vector.
        """
        if self.training: # Only apply noise during training
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        else: # During evaluation, use the mean
            return mu

    def forward(self, x: torch.Tensor) -> (torch.Tensor, torch.Tensor, torch.Tensor):
        """
        Forward pass for the BetaVAE.

        Args:
            x (torch.Tensor): Input image tensor.

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor]: 
                Reconstructed image (recon_x), latent mean (mu), and latent log-variance (logvar).
        """
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decoder(z)

        return recon_x, mu, logvar
