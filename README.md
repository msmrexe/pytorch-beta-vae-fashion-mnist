# β-VAE for Fashion-MNIST Disentanglement

A PyTorch implementation of a β-Variational Autoencoder (β-VAE) for disentangled representation learning and image generation on the Fashion-MNIST dataset. Developed for the Generative Models (M.S.) course, this project showcases the ability of β-VAE to learn a disentangled latent space, which is a crucial concept for interpretability in generative models.

## Features

* **Disentangled Latent Space Learning:** Utilizes the β-VAE objective to encourage a more disentangled latent representation of Fashion-MNIST images.
* **Image Reconstruction:** Reconstructs input images from their learned latent representations.
* **Novel Image Generation:** Generates new, diverse images by sampling from the latent space.
* **Modular Codebase:** Organized into `src/`, `scripts/`, and `notebooks/` for clarity and maintainability.
* **Comprehensive Evaluation:** Includes Fréchet Inception Distance (FID) for quantitative assessment of reconstruction and generation quality.
* **Detailed Logging:** Provides command-line and file-based logging for easy tracking of training progress and potential errors.
* **Argument Parsing:** Allows flexible configuration of hyperparameters via command-line arguments.

## Concepts Showcased

* **Variational Autoencoders (VAEs):** Core understanding of VAE architecture, probabilistic encoding/decoding, and the Evidence Lower Bound (ELBO) objective.
* **β-VAE:** Application and theoretical understanding of the beta parameter's role in promoting latent space disentanglement.
* **Reparameterization Trick:** Implementation of the reparameterization trick for stable training of VAEs.
* **Convolutional and Transposed Convolutional Networks:** Design and implementation of CNNs for image encoding and decoding.
* **Latent Space Learning:** Demonstrates the ability to learn a meaningful, continuous latent representation of complex data.
* **Generative Models Evaluation:** Practical application and interpretation of metrics like Fréchet Inception Distance (FID) for evaluating generative model performance.
* **Concept-Based Interpretability:** Foundations in using disentangled representations as a basis for more interpretable models.

---

## How It Works

This project implements a β-Variational Autoencoder to learn a lower-dimensional, disentangled latent representation of Fashion-MNIST images. The model consists of an encoder that maps input images to a probabilistic latent space and a decoder that reconstructs images from samples in this latent space.

### 1. Overview of the β-VAE Architecture and Flow

The β-VAE processes images through the following steps:
1.  **Encoding:** An input image `x` is passed through the `Encoder` network, which outputs the mean ($\mu$) and log-variance ($\log\sigma^2$) of a Gaussian distribution in the latent space.
2.  **Reparameterization:** A latent vector `z` is sampled from this Gaussian distribution using the reparameterization trick, which allows gradients to flow back through the sampling process.
3.  **Decoding:** The sampled latent vector `z` is fed into the `Decoder` network, which reconstructs the original image, producing `recon_x`.
4.  **Loss Calculation:** The model's objective function (ELBO) combines a reconstruction loss (Binary Cross-Entropy between `recon_x` and `x`) and a Kullback-Leibler (KL) divergence term. The KL divergence penalizes the learned latent distribution for straying too far from a standard normal prior, encouraging a well-behaved latent space. The `beta` parameter specifically weights this KL divergence term, pushing the model towards more disentangled latent factors.

The `main.py` script orchestrates the entire pipeline, from data loading and model initialization to training, evaluation, and visualization. It uses `scripts/train_vae.py` and `scripts/evaluate_vae.py` for their respective tasks, leveraging helper functions and model definitions from the `src/` directory.

### 2. Algorithms, Functions, and Models

The core of the project lies in the `src/` directory:

* **`src/models.py`**:
    * **`Encoder` Class:**
        * **Purpose:** Maps an input image (28x28 grayscale) to the parameters ($\mu$, $\log\sigma^2$) of a latent Gaussian distribution.
        * **Architecture:**
            1.  `Conv2D(1, 8, k=3, s=2, p=1)`: Output Size (batchsize, 8, 14, 14)
            2.  `ReLU`
            3.  `Conv2D(8, 16, k=3, s=2, p=1)`: Output Size (batchsize, 16, 7, 7)
            4.  `ReLU`
            5.  `BatchNorm2D(16)`
            6.  `Conv2D(16, 32, k=3, s=2, p=0)`: Output Size (batchsize, 32, 3, 3)
            7.  `ReLU`
            8.  `Flatten`: Output Size (batchsize, 32 * 3 * 3 = 288)
            9.  `Linear(288, 128)`: Fully Connected layer
            10. `Linear(128, latent_dim)` for $\mu$
            11. `Linear(128, latent_dim)` for $\log\sigma^2$
    * **`Decoder` Class:**
        * **Purpose:** Reconstructs an image from a latent vector `z`.
        * **Architecture:**
            1.  `Linear(latent_dim, 128)`: Fully Connected layer
            2.  `ReLU`
            3.  `Linear(128, 288)`: Fully Connected layer, then reshaped to (batchsize, 32, 3, 3)
            4.  `ReLU`
            5.  `ConvTranspose2D(32, 16, k=3, s=2, p=0)`: Output Size (batchsize, 16, 7, 7)
            6.  `ReLU`
            7.  `BatchNorm2D(16)`
            8.  `ConvTranspose2D(16, 8, k=3, s=2, p=1, op=1)`: Output Size (batchsize, 8, 14, 14)
            9.  `ReLU`
            10. `ConvTranspose2D(8, 1, k=3, s=2, p=1, op=1)`: Output Size (batchsize, 1, 28, 28)
            11. `Sigmoid`: Ensures output pixel values are in [0, 1].
    * **`BetaVAE` Class:**
        * **Purpose:** Orchestrates the Encoder and Decoder, implementing the reparameterization trick and managing the `beta` parameter.
        * **Reparameterization Trick:** Samples `z` using $z = \mu + \epsilon \cdot \exp(0.5 \cdot \log\sigma^2)$, where $\epsilon \sim \mathcal{N}(0, I)$. This allows gradient flow through the sampling.
* **`src/utils.py`**:
    * **`vae_loss(recon_x, x, mu, logvar, beta)`:** Computes the β-VAE loss.
        * **Reconstruction Loss:** Binary Cross-Entropy (BCE) on the pixel values. Input images are transformed from `[-1, 1]` to `[0, 1]` to be compatible with BCE, as the decoder output is `[0, 1]` due to Sigmoid activation.
        * **KL Divergence:** $-\frac{1}{2} \sum (1 + \log\sigma^2 - \mu^2 - \exp(\log\sigma^2))$.
        * The total loss is `recon_loss + beta * kl_divergence`.
    * **`setup_logging()`:** Configures logging to both console and a file for consistent output tracking.
    * **`plot_reconstructions()`:** Visualizes original vs. reconstructed images from the test set.
    * **`plot_generations()`:** Visualizes new images generated by sampling from the latent space.
    * **`calculate_fid()`:** Computes the Fréchet Inception Distance (FID).
        * **Formula:** $FID=∣∣\mu_r−\mu_g∣∣^2+Tr(\Sigma_r+\Sigma_g−2(\Sigma_r\Sigma_g)^{1/2})$
        * Compares the mean ($\mu$) and covariance ($\Sigma$) of feature distributions (here, flattened pixel values serve as features) between two sets of images (e.g., real vs. reconstructed, or real vs. generated). Lower FID indicates higher similarity and better perceptual quality.
* **`src/data_loader.py`**:
    * **`get_fashion_mnist_loaders(batch_size, data_dir)`:** Downloads and loads the Fashion-MNIST dataset, applies `transforms.ToTensor()` and `transforms.Normalize((0.5,), (0.5,))` to transform pixel values to `[-1, 1]`, and creates `DataLoader` instances for training and testing.

The project reveals that the β-VAE is capable of learning a compressed yet meaningful representation of Fashion-MNIST. The reconstructed images maintain the overall structure of the originals with some expected blurring due to the information bottleneck. More importantly, the generated images show diversity and plausibility, forming recognizable apparel items, indicating a well-learned and disentangled latent space. The FID scores provide a quantitative measure of this quality, with values reflecting a reasonable similarity between real and generated/reconstructed distributions.

---

## Project Structure

```
pytorch-beta-vae-fashion-mnist/
├── .gitignore              # Specifies intentionally untracked files to ignore (e.g., data, logs, checkpoints, Python environment files)
├── LICENSE                 # MIT License details for the project
├── README.md               # This detailed project description
├── requirements.txt        # Lists all Python dependencies required to run the project
├── main.py                 # The primary script to run the entire β-VAE pipeline (training, evaluation, visualization)
├── scripts/                # Contains executable scripts for specific tasks
│   ├── train_vae.py        # Script encapsulating the β-VAE training loop
│   └── evaluate_vae.py     # Script for evaluating the trained β-VAE model using FID
├── src/                    # Source code directory for modular components
│   ├── models.py           # Defines the Encoder, Decoder, and BetaVAE model architectures
│   ├── data_loader.py      # Handles loading and preprocessing of the Fashion-MNIST dataset
│   └── utils.py            # Contains utility functions for logging, loss calculation, visualization, and FID computation
└── notebooks/
    └── run_project.ipynb   # Jupyter Notebook for easily running the `main.py` script with various parameters and visualizing outputs
```

## How to Use

1.  **Clone the Repository:**
    ```bash
    git clone https://github.com/msmrexe/pytorch-beta-vae-fashion-mnist.git
    cd pytorch-beta-vae-fashion-mnist
    ```

2.  **Setup the Environment:**
    It is highly recommended to use a virtual environment.
    ```bash
    python -m venv venv
    source venv/bin/activate # On Windows: `venv\Scripts\activate`
    pip install -r requirements.txt
    ```

3.  **Run the Project:**
    Execute the `main.py` script. This will train the model, save a checkpoint, perform evaluation, and display visualization plots. You can customize parameters using command-line arguments.

    ```bash
    # Run with default parameters (20 epochs, beta=4.0, latent_dim=10)
    python main.py

    # Example: Run with custom parameters
    python main.py --latent_dim 20 --beta 5.0 --epochs 30 --batch_size 128 --lr 5e-4 --num_reconstruction_samples 15 --num_generation_samples 50

    # Example: Load a pre-trained model and only evaluate/visualize
    # (assuming you have a 'beta_vae_model.pth' in ./checkpoints/)
    # python main.py --load_path ./checkpoints/beta_vae_model.pth --epochs 0 # epochs=0 to skip training
    ```
    The `main.py` script will print logs to the console and save them to `beta_vae_run.log`. Image visualization windows will pop up during execution.

4.  **Explore with Jupyter Notebook:**
    For an interactive experience and easier parameter tuning, you can also use the provided Jupyter Notebook.
    ```bash
    jupyter notebook notebooks/run_project.ipynb
    ```
    Follow the instructions within the notebook to run the project.

---

## Author

Feel free to connect or reach out if you have any questions!

* **Maryam Rezaee**
* **GitHub:** [@msmrexe](https://github.com/msmrexe)
* **Email:** [ms.maryamrezaee@gmail.com](mailto:ms.maryamrezaee@gmail.com)

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for full details.
