# Databricks notebook source
# MAGIC %md
# MAGIC 
# MAGIC ### Introduction
# MAGIC Some very cool applications here: https://github.com/nashory/gans-awesome-applications
# MAGIC 
# MAGIC Following sample taken from Mosquera, D.G., 2018 (https://medium.com/ai-society/gans-from-scratch-1-a-deep-introduction-with-code-in-pytorch-and-tensorflow-cb03cdcdba0f) Accessed May 2020
# MAGIC 
# MAGIC Some key take away's (Mosquera, D.G., 2018):
# MAGIC - belong to the familiy of generative models in the field of unsupervised learning
# MAGIC   - other Generative Models in this family include Variational Autoencoders, (VAE), pixelCNN, PixelRNN and realNVP
# MAGIC - Generative Models learn the intrinsic distribution function of the input data (p(x), p(x,y...)), rather than learning to map a function to the data (as in supervised learning)
# MAGIC 
# MAGIC Some call-out advantages/disadvantages (Goodfellow, 2018) (https://www.youtube.com/watch?v=HGYYEUSm-0Q):
# MAGIC - can generate some very sharp images
# MAGIC - no statistical inference is required so easy to train using back-propogation to obtain gradients
# MAGIC 
# MAGIC Disadvantages:
# MAGIC - can be difficult to stabalise the training (for example, consider modal instability)
# MAGIC - GAN's belong to the direct implicit density models so they can model p(x) without explicitly defining the Probablity Distribution Function
# MAGIC 
# MAGIC ### GAN Structure
# MAGIC Original diagram here
# MAGIC - training with a neural network means a gradient based approach can be taken
# MAGIC 
# MAGIC Steps to train:
# MAGIC The fundamental steps to train a GAN can be described as following:
# MAGIC 1. Sample a noise set and a real-data set, each with size m.
# MAGIC 1. Train the Discriminator on this data.
# MAGIC 1. Sample a different noise subset with size m.
# MAGIC 1. Train the Generator on this data.
# MAGIC 1. Repeat from Step 1.

# COMMAND ----------

# MAGIC %run ./utils/Helpers

# COMMAND ----------

# MAGIC %run ./utils/Logger

# COMMAND ----------

# working with MLRuntime 6.5 so have tensorFlow and Pytorch already available

# seems the client environment must match the target environment
# so make sure these are installed locally:
import torch
from torch import nn, optim
from torch.autograd.variable import Variable
from torchvision import transforms,datasets

# environment setup for databricks connect:
# https://docs.databricks.com/dev-tools/databricks-connect.html

# install requirements using 
# pip install -r pytorch-research/tutorial/db-connect/requirements.txt

# ensure you've run source ~/db-connect.sh before running pyspark

# session set-up:
# from pyspark.dbutils import DBUtils
# dbutils = DBUtils(spark.sparkContext)

dbfs_out_dir = '/tmp/ryansimpson/dataset'
out_dir = '/dbfs' + dbfs_out_dir

dbutils.tensorboard.start(out_dir + '/runs')

if dbutils.fs.mkdirs(dbfs_out_dir):
    print(f"{dbfs_out_dir} already exists")

# would be nice to be able to add these without a concrete path:
# sc.addPyFile("./pytorch-research/tutorial/db-connect/utils/Helpers.py")
# sc.addPyFile("./pytorch-research/tutorial/db-connect/utils/Logger.py")


# import Helpers
# import Logger

data = mnist_data(out_dir)
# Create loader with data, so that we can iterate over it
data_loader = torch.utils.data.DataLoader(data, batch_size=100, shuffle=True)
# Num batches
num_batches = len(data_loader)

# COMMAND ----------

# Discriminator
discriminator = DiscriminatorNet()

# Discriminator
generator = GeneratorNet()


# ADAM as the optimization algorithm, learning rate of 0.0002
# loss function for this task will be Binary Cross Entropy Loss (BCE)
# - resembles the log-loss for both the Generator and Discriminator

d_optimizer = optim.Adam(discriminator.parameters(), lr=0.0002)
g_optimizer = optim.Adam(generator.parameters(), lr=0.0002)

loss = nn.BCELoss()

num_test_samples = 16
test_noise = noise(num_test_samples)

# COMMAND ----------

# Create logger instance
logger = Logger(model_name='VGAN', data_name='MNIST', data_path_root=out_dir)
# Total number of epochs to train
num_epochs = 200
for epoch in range(num_epochs):
    for n_batch, (real_batch,_) in enumerate(data_loader):
        N = real_batch.size(0)
        # 1. Train Discriminator
        real_data = Variable(images_to_vectors(real_batch))
        # Generate fake data and detach 
        # (so gradients are not calculated for generator)
        fake_data = generator(noise(N)).detach()
        # Train D
        d_error, d_pred_real, d_pred_fake = \
              train_discriminator(d_optimizer, discriminator, real_data, fake_data)

        # 2. Train Generator
        # Generate fake data
        fake_data = generator(noise(N))
        # Train G
        g_error = train_generator(g_optimizer, discriminator, fake_data)
        # Log batch error
        logger.log(d_error, g_error, epoch, n_batch, num_batches)
        # Display Progress every few batches
        if (n_batch) % 100 == 0: 
            test_images = vectors_to_images(generator(test_noise))
            test_images = test_images.data
            logger.log_images(
                test_images, num_test_samples, 
                epoch, n_batch, num_batches
            );
            # Display status Logs
            logger.display_status(
                epoch, num_epochs, n_batch, num_batches,
                d_error, g_error, d_pred_real, d_pred_fake
            )