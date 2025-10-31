import torch
import logging

from src.utils import calculate_fid

def evaluate_model(model, test_loader, device):
    """
    Evaluates the β-VAE model using FID for reconstruction and generation.

    Args:
        model (BetaVAE): The trained β-VAE model.
        test_loader (DataLoader): DataLoader for the test dataset.
        device (torch.device): Device to run the evaluation on (cpu or cuda).
    """
    model.eval()
    
    logging.info("Calculating FID for reconstruction...")
    fid_reconstruction = calculate_fid(model, test_loader, device, mode='reconstruction')
    logging.info(f"FID for Reconstruction: {fid_reconstruction:.4f}")
    print(f"FID for Reconstruction: {fid_reconstruction:.4f}") # Command line output

    logging.info("Calculating FID for generation...")
    # For generation FID, we need to generate samples of the same size as the test dataset
    total_test_samples = len(test_loader.dataset)
    fid_generation = calculate_fid(model, test_loader, device, mode='generation', 
                                   num_generated_samples=total_test_samples)
    logging.info(f"FID for Generation: {fid_generation:.4f}")
    print(f"FID for Generation: {fid_generation:.4f}") # Command line output
