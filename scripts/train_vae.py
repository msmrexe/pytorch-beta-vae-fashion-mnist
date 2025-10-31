import torch
import torch.optim as optim
from tqdm.auto import tqdm
import logging
import os

from src.utils import vae_loss

def train_model(model, train_loader, epochs, lr, device, save_path=None):
    """
    Trains the β-VAE model.

    Args:
        model (BetaVAE): The β-VAE model to train.
        train_loader (DataLoader): DataLoader for the training dataset.
        epochs (int): Number of training epochs.
        lr (float): Learning rate for the optimizer.
        device (torch.device): Device to run the training on (cpu or cuda).
        save_path (str, optional): Path to save the trained model. Defaults to None.
    """
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Ensure save directory exists
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for batch_idx, (x, _) in enumerate(pbar):
            x = x.to(device)
            optimizer.zero_grad()

            try:
                recon_x, mu, logvar = model(x)
                loss = vae_loss(recon_x, x, mu, logvar, model.beta)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            except Exception as e:
                logging.error(f"Error during training batch {batch_idx} in Epoch {epoch+1}: {e}", exc_info=True)
                raise # Re-raise to stop training if a critical error occurs

            pbar.set_postfix(loss=loss.item() / x.size(0)) # Normalize loss per image in batch

        avg_train_loss = train_loss / len(train_loader.dataset)
        logging.info(f"Epoch {epoch+1}, Average Loss: {avg_train_loss:.4f}")
        print(f"Epoch {epoch+1}, Average Loss: {avg_train_loss:.4f}") # Command line output

    if save_path:
        torch.save(model.state_dict(), save_path)
        logging.info(f"Model saved to {save_path}")
