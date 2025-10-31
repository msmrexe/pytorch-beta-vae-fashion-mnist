import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import sqrtm
import logging
import sys
import os

def setup_logging(log_file: str = "beta_vae_run.log"):
    """
    Sets up logging to both console and a file.

    Args:
        log_file (str): Path to the log file.
    """
    log_dir = os.path.dirname(log_file)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir)

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    logging.info(f"Logging configured. Output will be saved to '{log_file}'.")


def vae_loss(recon_x: torch.Tensor, x: torch.Tensor, mu: torch.Tensor, logvar: torch.Tensor, beta: float) -> torch.Tensor:
    """
    Calculates the β-VAE loss (ELBO).

    The loss is composed of:
    1. Reconstruction Loss: Binary Cross-Entropy (BCE) between reconstructed and original images.
       Note: Input images are normalized to [-1, 1] by `transforms.Normalize((0.5,), (0.5,))`.
       For BCE, pixel values should typically be in [0, 1].
       We transform `x` from [-1, 1] to [0, 1] by `(x + 1) / 2` before computing BCE.
       The decoder's final sigmoid activation ensures `recon_x` is in [0, 1].
    2. KL Divergence: Measures the difference between the learned latent distribution
       and a standard normal prior. Weighted by `beta`.

    Args:
        recon_x (torch.Tensor): Reconstructed images from the decoder.
        x (torch.Tensor): Original input images.
        mu (torch.Tensor): Mean of the latent distribution.
        logvar (torch.Tensor): Log-variance of the latent distribution.
        beta (float): Weight for the KL divergence term.

    Returns:
        torch.Tensor: The total β-VAE loss.
    """
    # Transform input x from [-1, 1] to [0, 1] for BCE loss calculation
    x_transformed = (x + 1) / 2
    
    # Reconstruction loss (Binary Cross-Entropy)
    # Using reduction='sum' to sum over all pixels and all images in the batch
    # This matches the original notebook's implicit summing behavior
    recon_loss = F.binary_cross_entropy(recon_x, x_transformed, reduction='sum')

    # KL Divergence
    # -0.5 * sum(1 + logvar - mu^2 - exp(logvar))
    kl_divergence = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return recon_loss + beta * kl_divergence


def plot_reconstructions(model, test_loader, device, num_samples: int = 10):
    """
    Visualizes original and reconstructed images from the test set.

    Args:
        model (BetaVAE): Trained β-VAE model.
        test_loader (DataLoader): DataLoader for the test dataset.
        device (torch.device): Device to run inference on.
        num_samples (int): Number of samples to visualize.
    """
    model.eval()
    
    # Fetch enough samples to meet num_samples
    x_batch = []
    for x, _ in test_loader:
        x_batch.append(x)
        if sum(b.shape[0] for b in x_batch) >= num_samples:
            break
    x = torch.cat(x_batch)[:num_samples].to(device)

    with torch.no_grad():
        recon_x, _, _ = model(x)

    # Convert back to numpy and adjust for plotting
    # Original images were normalized to [-1, 1], so convert back to [0, 1]
    x_display = ((x.cpu().numpy() + 1) / 2).transpose(0, 2, 3, 1).squeeze()
    recon_x_display = recon_x.cpu().numpy().transpose(0, 2, 3, 1).squeeze()

    fig, axes = plt.subplots(num_samples, 2, figsize=(6, num_samples * 2))
    for i in range(num_samples):
        # Handle cases where image might be grayscale (2D) or RGB (3D)
        axes[i, 0].imshow(x_display[i], cmap='gray' if x_display[i].ndim == 2 else None)
        axes[i, 0].set_title('Original')
        axes[i, 0].axis('off')

        axes[i, 1].imshow(recon_x_display[i], cmap='gray' if recon_x_display[i].ndim == 2 else None)
        axes[i, 1].set_title('Reconstructed')
        axes[i, 1].axis('off')
    
    fig.suptitle(f"Reconstructions from Test Set ({num_samples} Samples)", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to prevent title overlap
    plt.show()
    logging.info(f"Displayed {num_samples} reconstructed images.")


def plot_generations(model, device, latent_dim: int, num_samples: int = 25):
    """
    Generates and visualizes images from random latent space samples.

    Args:
        model (BetaVAE): Trained β-VAE model.
        device (torch.device): Device to run inference on.
        latent_dim (int): Dimension of the latent space.
        num_samples (int): Number of samples to generate and visualize.
    """
    model.eval()

    with torch.no_grad():
        z = torch.randn(num_samples, latent_dim).to(device)
        samples = model.decoder(z)

    # Convert back to numpy and adjust for plotting
    samples_display = samples.cpu().numpy().transpose(0, 2, 3, 1).squeeze()

    # Determine grid size, aiming for roughly square
    grid_rows = int(np.ceil(np.sqrt(num_samples)))
    grid_cols = int(np.ceil(num_samples / grid_rows))
    
    fig, axes = plt.subplots(grid_rows, grid_cols, figsize=(grid_cols * 2, grid_rows * 2))
    axes = axes.flatten() # Flatten in case of non-square grid

    for i in range(num_samples):
        ax = axes[i]
        ax.imshow(samples_display[i], cmap='gray' if samples_display[i].ndim == 2 else None)
        ax.axis('off')
    
    # Hide any unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    fig.suptitle(f"Generated Samples from Latent Space ({num_samples} Samples)", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to prevent title overlap
    plt.show()
    logging.info(f"Displayed {num_samples} generated images.")


def calculate_fid(model, data_loader, device, mode: str = 'reconstruction', num_generated_samples: int = None):
    """
    Calculates the Fréchet Inception Distance (FID) for either reconstruction or generation.

    Args:
        model (BetaVAE): The trained β-VAE model.
        data_loader (DataLoader): DataLoader for the dataset (e.g., test set).
        device (torch.device): Device to perform computations on.
        mode (str): 'reconstruction' to compare originals with reconstructions,
                    'generation' to compare originals with new generations.
        num_generated_samples (int, optional): Required for 'generation' mode
                                               to specify how many samples to generate.

    Returns:
        float: The calculated FID score.
    
    Raises:
        ValueError: If mode is invalid or num_generated_samples is not provided for generation mode.
    """
    model.eval()
    original_features = []
    comparison_features = [] # This will hold reconstructed or generated features

    if mode not in ['reconstruction', 'generation']:
        raise ValueError("Mode must be 'reconstruction' or 'generation'.")
    if mode == 'generation' and num_generated_samples is None:
        raise ValueError("num_generated_samples must be provided for 'generation' mode.")

    logging.info(f"Calculating FID in '{mode}' mode...")

    with torch.no_grad():
        for i, (x, _) in enumerate(data_loader):
            x = x.to(device)
            # Flatten original images for feature comparison (treating pixels as features)
            # Convert from [-1, 1] range to [0, 1] for more standard feature representation
            original_features.append(((x + 1) / 2).view(x.size(0), -1).cpu().numpy())

            if mode == 'reconstruction':
                recon_x, _, _ = model(x)
                comparison_features.append(recon_x.view(recon_x.size(0), -1).cpu().numpy())
            elif mode == 'generation':
                # Generate a batch of samples. Ensure total generated samples match desired num_generated_samples
                current_batch_size = x.size(0)
                if num_generated_samples is not None and len(comparison_features) * data_loader.batch_size + current_batch_size > num_generated_samples:
                    current_batch_size = num_generated_samples - len(comparison_features) * data_loader.batch_size
                    if current_batch_size <= 0:
                        break # Stop if we have generated enough samples
                
                z = torch.randn(current_batch_size, model.latent_dim).to(device)
                generated_x = model.decoder(z)
                comparison_features.append(generated_x.view(generated_x.size(0), -1).cpu().numpy())
            
    original_features = np.concatenate(original_features, axis=0)
    comparison_features = np.concatenate(comparison_features, axis=0)

    # Trim generated features if num_generated_samples was used and we generated more than needed
    if mode == 'generation' and comparison_features.shape[0] > num_generated_samples:
        comparison_features = comparison_features[:num_generated_samples]
    
    # Ensure both feature sets have the same number of samples for FID calculation if not generation
    if mode == 'reconstruction' and original_features.shape[0] != comparison_features.shape[0]:
        min_samples = min(original_features.shape[0], comparison_features.shape[0])
        original_features = original_features[:min_samples]
        comparison_features = comparison_features[:min_samples]


    if original_features.shape[0] == 0 or comparison_features.shape[0] == 0:
        logging.error("No features collected for FID calculation. Check data loader or generation logic.")
        return float('nan')

    # Calculate means and covariances
    mu_orig = original_features.mean(axis=0)
    sigma_orig = np.cov(original_features, rowvar=False)
    mu_comp = comparison_features.mean(axis=0)
    sigma_comp = np.cov(comparison_features, rowvar=False)

    # Calculate FID
    try:
        diff = mu_orig - mu_comp
        # Ensure that sigma matrices are positive semi-definite and handle sqrtm issues
        cov_sqrt = sqrtm(sigma_orig @ sigma_comp)
        if np.iscomplexobj(cov_sqrt):
            logging.warning("Complex numbers encountered in sqrtm, taking real part. This might indicate numerical instability.")
            cov_sqrt = cov_sqrt.real
        
        fid = np.sum(diff**2) + np.trace(sigma_orig + sigma_comp - 2 * cov_sqrt)
    except Exception as e:
        logging.error(f"Error during FID calculation: {e}", exc_info=True)
        return float('nan') # Return Not a Number if calculation fails

    return fid
