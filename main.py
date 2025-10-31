import argparse
import logging
import os
import sys

from src.data_loader import get_fashion_mnist_loaders
from src.models import BetaVAE
from src.utils import vae_loss, setup_logging, plot_reconstructions, plot_generations, calculate_fid
from scripts.train_vae import train_model
from scripts.evaluate_vae import evaluate_model

def main():
    """
    Main function to run the β-VAE training and evaluation pipeline.
    Handles argument parsing, logging setup, and orchestrates the
    data loading, model training, and evaluation steps.
    """
    parser = argparse.ArgumentParser(description="β-VAE for Fashion-MNIST")
    parser.add_argument("--latent_dim", type=int, default=10,
                        help="Dimension of the latent space (default: 10)")
    parser.add_argument("--beta", type=float, default=4.0,
                        help="Beta parameter for β-VAE loss (default: 4.0)")
    parser.add_argument("--epochs", type=int, default=20,
                        help="Number of training epochs (default: 20)")
    parser.add_argument("--batch_size", type=int, default=64,
                        help="Batch size for training (default: 64)")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Learning rate for the optimizer (default: 1e-3)")
    parser.add_argument("--num_reconstruction_samples", type=int, default=10,
                        help="Number of samples to visualize for reconstruction (default: 10)")
    parser.add_argument("--num_generation_samples", type=int, default=25,
                        help="Number of samples to visualize for generation (default: 25)")
    parser.add_argument("--log_file", type=str, default="beta_vae_run.log",
                        help="File to save logging output")
    parser.add_argument("--save_path", type=str, default="./checkpoints/beta_vae_model.pth",
                        help="Path to save the trained model checkpoint")
    parser.add_argument("--load_path", type=str, default=None,
                        help="Path to load a pre-trained model checkpoint")
    parser.add_argument("--data_dir", type=str, default="./data",
                        help="Directory to store Fashion-MNIST data")
    
    args = parser.parse_args()

    # Setup logging
    setup_logging(args.log_file)
    logging.info(f"Starting β-VAE project with arguments: {args}")

    try:
        # Set device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logging.info(f"Using device: {device}")

        # Load data
        logging.info("Loading Fashion-MNIST dataset...")
        train_loader, test_loader = get_fashion_mnist_loaders(
            batch_size=args.batch_size, data_dir=args.data_dir
        )
        logging.info(f"Data loaders created. Train batches: {len(train_loader)}, Test batches: {len(test_loader)}")

        # Initialize model
        model = BetaVAE(latent_dim=args.latent_dim, beta=args.beta).to(device)
        logging.info(f"Model initialized:\n{model}")

        if args.load_path:
            if os.path.exists(args.load_path):
                model.load_state_dict(torch.load(args.load_path, map_location=device))
                logging.info(f"Loaded pre-trained model from {args.load_path}")
            else:
                logging.warning(f"Pre-trained model not found at {args.load_path}. Training from scratch.")

        # Train model
        logging.info("Starting model training...")
        train_model(model, train_loader, args.epochs, args.lr, device, args.save_path)
        logging.info("Model training finished.")

        # Evaluate model
        logging.info("Starting model evaluation...")
        evaluate_model(model, test_loader, device)
        logging.info("Model evaluation finished.")

        # Visualize results
        logging.info("Generating visualizations...")
        plot_reconstructions(model, test_loader, device, num_samples=args.num_reconstruction_samples)
        plot_generations(model, device, args.latent_dim, num_samples=args.num_generation_samples)
        logging.info("Visualizations generated and displayed.")

    except Exception as e:
        logging.error(f"An error occurred during execution: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    import torch # Import torch here to avoid circular dependency with setup_logging
    main()
