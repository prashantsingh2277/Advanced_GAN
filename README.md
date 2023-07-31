# Advanced_GAN

Importing Libraries: The code begins by importing various libraries, including PyTorch modules for data handling, neural networks, optimization, and visualization. It also imports additional libraries like wandb for experiment tracking and matplotlib for displaying images.

Visualization Function: The show function is a utility function to visualize a grid of images. It takes a tensor of image data and displays a grid of images using matplotlib. If wandbactive is set to 1, it also logs the images to Weights & Biases (wandb) for experiment tracking.

Hyperparameters and General Parameters: Several hyperparameters and general parameters are defined, such as the number of epochs, batch size, learning rate, size of the noise vector (z_dim), and the device to run the code on (GPU in this case).

Setting Up Weights & Biases (wandb): The code initializes Weights & Biases (wandb) for experiment tracking and logs some configuration settings.

Generator Model: The Generator class represents the generator neural network. It takes random noise vectors and generates fake images. The generator uses transposed convolutions (convolutional transpose layers) to upsample the noise into images. The output images have a size of 128x128 pixels and 3 channels (RGB).

Noise Generation Function: The gen_noise function generates random noise vectors for the generator. It creates a tensor of shape (num, z_dim) containing random numbers drawn from a normal distribution and moves it to the specified device.

Critic Model: The Critic class represents the critic neural network. The critic takes real and fake images as input and tries to distinguish between them. It uses standard convolutional layers to process the images. The output of the critic is a scalar value representing the critic's confidence in the input being real (positive value) or fake (negative value).

DataLoader and Data Preparation: The CelebA dataset is loaded using a custom Dataset class, and a DataLoader is set up to handle the data in batches during training.

Models and Optimizers: The generator and critic models are instantiated, and separate optimizers are set up for both models.

Wasserstein Loss and Gradient Penalty: The Wasserstein loss is calculated as the difference between the mean predictions of the critic on real and fake images. Additionally, a gradient penalty term is computed to enforce Lipschitz continuity, which helps stabilize the training. Both generator and critic losses are tracked during training.

Training Loop: The main training loop runs for a specified number of epochs. Within each epoch, the critic is trained for several cycles using a specified number of critic updates (crit_cycles). The generator is then trained once. After each training iteration, the losses are logged to wandb for experiment tracking.

Visualization and Monitoring: During training, the generated images and real images are visualized every show_step steps, and the generator and critic losses are plotted.

Checkpoints: The generator and critic models, as well as their optimizer states, are saved periodically as checkpoints.

Interpolation (Morphing) in Latent Space: The code demonstrates how to interpolate between two random noise vectors in the latent space to create a smooth transition of generated images.

