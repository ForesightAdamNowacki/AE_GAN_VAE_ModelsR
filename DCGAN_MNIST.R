# ------------------------------------------------------------------------------
# GENERATIVE ADVERSARIAL MODEL
# ------------------------------------------------------------------------------
# Data:
# https://www.kaggle.com/c/digit-recognizer
# utils::browseURL(url = "https://www.kaggle.com/c/digit-recognizer")

# ------------------------------------------------------------------------------
# Intro:
base::setwd("D:/GitHub/GANModelsR")

# ------------------------------------------------------------------------------
# Environment:
reticulate::use_condaenv("GPU_ML_2", required = TRUE)
base::library(tensorflow)
base::library(keras)
# keras::install_keras(tensorflow = "gpu")
base::library(tidyverse)
base::library(deepviz)
base::library(gridExtra)
base::library(ggplot2)

# ------------------------------------------------------------------------------
# Clear session:
keras::k_clear_session()

# ------------------------------------------------------------------------------
# Generator:
build_generator <- function(latent_size, image_size){
  
  # Build a Generator Model
  # Stack of Batch Normalization - ReLU activations - Conv2DTranspose layers to 
  # generate fake images.
  
  # Arguments:
  # * inputs (Layer): Input layer of the generator (z-vector)
  # * image_size: Target size of one side (assuming square image)
  
  # Returns:
  # * Model: Generator Model
  
  image_resize <- image_size %/% 4
  
  generator_input <-  keras::layer_input(shape = base::c(latent_size))
  generator_output <- generator_input %>%
    keras::layer_dense(units = image_resize * image_resize * 128, activation = "linear") %>%
    keras::layer_reshape(target_shape = base::c(image_resize, image_resize, 128)) %>%
    keras::layer_batch_normalization() %>%
    keras::layer_activation(activation = "relu") %>%
    keras::layer_conv_2d_transpose(filters = 128, kernel_size = 5, strides = 2, padding = "same", activation = "linear") %>%
    keras::layer_batch_normalization() %>%
    keras::layer_activation(activation = "relu") %>%
    keras::layer_conv_2d_transpose(filters = 64, kernel_size = 5, strides = 2, padding = "same", activation = "linear") %>%
    keras::layer_batch_normalization() %>%
    keras::layer_activation(activation = "relu") %>%
    keras::layer_conv_2d_transpose(filters = 32, kernel_size = 5, strides = 1, padding = "same", activation = "linear") %>%
    keras::layer_batch_normalization() %>%
    keras::layer_activation(activation = "relu") %>%
    keras::layer_conv_2d_transpose(filters = 1, kernel_size = 5, strides = 1, padding = "same", activation = "linear") %>%
    keras::layer_activation(activation = "sigmoid")
  
  generator <- keras::keras_model(inputs = generator_input, outputs = generator_output)
  base::return(generator)}

# build_generator(latent_size = 100, image_size = 28)

# ------------------------------------------------------------------------------
# Discriminator:
build_discriminator <- function(inputs){
  
  # Build a Discriminator Model
  # Stack of Leaky ReLU activations - Conv2D layers to discriminate real from 
  # fake. 
  
  # Arguments:
  # * inputs (Layer): Input layer of the discriminator (the image)
  
  # Returns:
  # * Model: Discriminator Model
  
  discriminator_input <- keras::layer_input(shape = inputs)
  discriminator_output <- discriminator_input %>%
    keras::layer_activation_leaky_relu(alpha = 0.2) %>%
    keras::layer_conv_2d(filters = 32, kernel_size = 5, strides = 2, padding = "same") %>%
    keras::layer_activation_leaky_relu(alpha = 0.2) %>%
    keras::layer_conv_2d(filters = 64, kernel_size = 5, strides = 2, padding = "same") %>%
    keras::layer_activation_leaky_relu(alpha = 0.2) %>%
    keras::layer_conv_2d(filters = 128, kernel_size = 5, strides = 2, padding = "same") %>%
    keras::layer_activation_leaky_relu(alpha = 0.2) %>%
    keras::layer_conv_2d(filters = 256, kernel_size = 5, strides = 1, padding = "same") %>%
    keras::layer_flatten() %>%
    keras::layer_dense(units = 1, activation = "linear") %>%
    keras::layer_activation(activation = "sigmoid")
    
  discriminator <- keras::keras_model(inputs = discriminator_input, outputs = discriminator_output)
  
  base::return(discriminator)}

# build_discriminator(inputs = base::c(28, 28, 1))

# ------------------------------------------------------------------------------
# Build and train models:
build_and_train_models <- function(){
  
  # Load MNIST dataset:
  c(c(x_train, y_train), c(x_test, y_test)) %<-% keras::dataset_mnist()
  
  # Reshape data for CNN as (28, 28, 1) and normalize:
  image_size <- base::dim(x_train)[2]
  x_train <- keras::array_reshape(x = x_train, dim = base::c(-1, image_size, image_size, 1))
  x_train <- x_train/255

  # Network parameters:
  model_name <- "dcgan"
  latent_size <- 100
  batch_size <- 256
  train_steps <- 40000 
  lr <- 2e-4
  decay <- 6e-8
  input_shape <- base::c(image_size, image_size, 1)
  
  # Build discriminator model:
  optimizer <- keras::optimizer_rmsprop(lr = lr, decay = decay)
  discriminator <- build_discriminator(inputs = input_shape)
  discriminator %>% keras::compile(loss = "binary_crossentropy",
                                   optimizer = optimizer,
                                   metrics = base::c("accuracy"))
  discriminator %>% 
    base::summary()
  
  # Build generator model:
  generator <- build_generator(latent_size = latent_size, image_size = image_size)
  generator %>% 
    base::summary()
  
  # Build adversarial model:
  optimizer <- keras::optimizer_rmsprop(lr = lr * 0.5, decay = decay * 0.5)
  discriminator$trainable <- FALSE
  
  # Adversarial = generator + discriminator
  adversarial <- keras::keras_model(inputs = generator$inputs,
                                    outputs = discriminator(generator(generator$inputs)))
  adversarial %>% 
    keras::compile(loss = "binary_crossentropy",
                   optimizer = optimizer,
                   metrics = base::c("accuracy"))
  adversarial %>%
    base::summary()
  
  # Train the Discriminator and Adversarial Networks:
  models <- base::list(generator, discriminator, adversarial)
  params <- base::list(batch_size, latent_size, train_steps, image_size, model_name)
  train(models, x_train, params)}

# ------------------------------------------------------------------------------
# Plot images:
plot_images <- function(generator,
                        noise_input,
                        step,
                        model_name){
  
  # Display and save results to show the generator output evolves during training.
  
  # Arguments:
  # * generator (Model): trained generator
  # * noise input (Tensor): 16 noise vectors with shape equal latent_size
  # * step: training step
  # * model_name: folder with saved plots
  
  # Returns:
  # * png files: Generator outputs
  
  # Create folder to store saved plots:
  base::dir.create(model_name, showWarnings = FALSE)
  filename <- base::paste(model_name, base::paste0(step, ".png"), sep = "/")
  
  # Predict with generator noise input:
  images <- generator %>%
    predict(noise_input)
  
  # Extract helpful features:
  num_images <- base::dim(images)[1]
  image_size <- base::dim(images)[2]
  rows <- base::sqrt(base::dim(noise_input)[1])
  
  # Display plots:
  plots <- list()
  
  for (j in 1:num_images){
    images[j,,,] %>%
      tibble::as_tibble() %>%
      dplyr::mutate(row = dplyr::row_number()) %>%
      tidyr::pivot_longer(cols = dplyr::starts_with("V"),
                          names_to = "col",
                          values_to = "value") %>%
      dplyr::mutate(col = stringr::str_sub(col, 2, -1),
                    col = base::factor(col, levels = 1:image_size, ordered = TRUE),
                    row = base::factor(row, levels = 1:image_size, ordered = TRUE)) %>%
      ggplot2::ggplot(data = ., mapping = ggplot2::aes(x = col, y = row, fill = value)) +
      ggplot2::geom_tile() +
      ggplot2::scale_fill_gradient2(low = "white", high = "black", limits = base::c(0, 1)) +
      ggplot2::theme(plot.title = element_blank(),
                     axis.text.y = element_blank(),
                     axis.text.x = element_blank(),
                     axis.title.y = element_blank(),
                     axis.title.x = element_blank(),
                     axis.ticks = element_line(size = 1, color = "black", linetype = "solid"),
                     axis.ticks.length = unit(0, "cm"),
                     panel.grid.major.x = element_blank(),
                     panel.grid.major.y = element_blank(),
                     panel.grid.minor.x = element_blank(),
                     panel.grid.minor.y = element_blank(),
                     plot.caption = element_blank(),
                     legend.position = "none") -> plots[[j]]}
  plots <- base::do.call(grid.arrange, base::c(plots, ncol = rows))
  
  # Save plots:
  ggplot2::ggsave(filename, plots)
  log_image <- base::paste("Plot saved:", filename)  
  base::print(log_image)}

# ------------------------------------------------------------------------------
# Train:
train <- function(models, x_train, params){
  
  # Train the Discriminator and Adversarial Networks
  # Alternately train Discriminator and Adversarial networks by batch. Discriminator
  # is trained first with properly real and fake images. Adversarial is trained next 
  # with fake images pretending to be real - generated sample images per save_interval.
  
  # Arguments:
  # * models (list): Generator, Discriminator, Arversarial models
  # * x_train (tensor): Train images
  # * params (list): Networks parameters

  # The GAN Models:
  c(generator, discriminator, adversarial) %<-% models
  
  # Network parameters:
  c(batch_size, latent_size, train_steps, image_size) %<-% params[1:4]
  c(model_name) %<-% params[5]
  
  # The generator image is saved every 500 steps:
  save_interval <- 1000
  
  # The noise vector to see how the generator output evolves during training:
  noise_input <- base::matrix(data = stats::runif(n = 64 * latent_size, min = -1, max = 1), nrow = 64, ncol = latent_size)

  # Number of elements in train dataset:
  train_size <- base::dim(x_train)[1]

  for (i in 1:train_steps){
    
    # Train the discriminator for 1 batch: 
      # 1 batch of real (label = 1)
      # 1 batch of fake (label = 0)
    
    # Randomly pick real images from dataset:
    rand_indexes <- base::sample(x = 1:train_size, size = batch_size, replace = FALSE)
    real_images <- x_train[rand_indexes,,,]
    real_images <- keras::array_reshape(x = real_images, dim = base::c(-1, image_size, image_size, 1))
    
    # Generate fake images from noise using generator - generate noise using uniform
    # distribution:
    noise <- base::matrix(data = stats::runif(n = batch_size * latent_size, min = -1, max = 1), nrow = batch_size, ncol = latent_size); dim(noise)
    fake_images <- generator %>% stats::predict(noise)
    
    # real + fake images = 1 batch of train data:
    x <- keras::k_concatenate(tensors = base::list(real_images, fake_images), axis = 1)

    # Label real and fake images: real images -> 1, fake images -> 0
    y <- base::rep(x = base::c(1, 0), each = batch_size)

    # Train discriminator network - log the loss and accuracy:    
    discriminator_result <- discriminator %>% 
      keras::train_on_batch(x = x, y = y)
    c(discriminator_loss, discriminator_acc) %<-% discriminator_result
    discriminator_log <- base::paste0(i, ": Discriminator loss: ", base::round(discriminator_loss, 6), ", Discriminator accuracy: ", base::round(discriminator_acc, 6))
    
    # Train the adversarial network for 1 batch (1 batch of fake images with 
    # label = 1). The discriminator weights are frozen in adversarial and only
    # generator is trained. Generate noise using uniform distribution:
    noise <- base::matrix(data = stats::runif(n = batch_size * latent_size, min = -1, max = 1), nrow = batch_size, ncol = latent_size)
    
    # Label fake images as real = 1:
    y <- base::rep(x = 1, times = batch_size)
    
    # Train the adversarial network. Note that unlike in discriminator training we
    # do not save the fake images in a variable. The fake images go to the 
    # discriminator input of the adversarial for classification. Log the loss 
    # and accuracy:
    adversarial_result <- adversarial %>%
      keras::train_on_batch(x = noise, y = y)
    c(adversarial_loss, adversarial_acc) %<-% adversarial_result
    adversarial_log <- base::paste0("Adversarial loss: ", base::round(adversarial_loss, 6), ", Adversarial accuracy: ", base::round(adversarial_acc, 6))
    
    # Concatenate logs from Discriminator and Adversarial Network training: 
    log <- base::paste(discriminator_log, adversarial_log, sep = " | ")
    base::print(log)
    
    if (i %% save_interval == 0){
      plot_images(generator = generator,
                  noise_input = noise_input,
                  step = i,
                  model_name = model_name)}}
  
  # Save the model after training the generator (.h5 and .hdf5). The trained 
  # generator can be reloaded for future MNIST digit generation.
  generator %>%
    keras::save_model_hdf5(base::paste0(model_name, ".h5"))
  generator %>%
    keras::save_model_hdf5(base::paste0(model_name, ".hdf5"))}

# ------------------------------------------------------------------------------
# Train DCGAN model:
keras::k_clear_session()
build_and_train_models()

# ------------------------------------------------------------------------------






  
