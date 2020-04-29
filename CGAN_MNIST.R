# ------------------------------------------------------------------------------
# CONDITIONAL GENERATIVE ADVERSARIAL MODEL - CGAN
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
build_generator <- function(inputs, labels, image_size){
  
  # Build a Generator Model
  # Stack of Batch Normalization - ReLU activations - Conv2DTranspose layers to 
  # generate fake images.
  
  # Arguments:
  # * inputs (Layer): Input layer of the generator (z-vector)
  # * y_labels (Layer): Input layer for one-hot vector to condition the inputs
  # * image_size: Target size of one side (assuming square image)
  
  # Returns:
  # * Model: Generator Model
  
  image_resize <- image_size %/% 4

  generator_concatenated <- keras::layer_concatenate(inputs = base::list(inputs, labels))
  
  generator_output <- generator_concatenated %>%
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
  
  generator <- keras::keras_model(inputs = base::list(inputs, labels), outputs = generator_output)
  base::return(generator)}

keras::k_clear_session()
test_build_generator <- build_generator(inputs = keras::layer_input(shape = base::c(100)),
                                        labels = keras::layer_input(shape = base::c(10)),
                                        image_size = 28); test_build_generator
deepviz::plot_model(test_build_generator)

# ------------------------------------------------------------------------------
# Discriminator:
build_discriminator <- function(inputs, labels, image_size){
  
  # Build a Discriminator Model
  # Stack of Leaky ReLU activations - Conv2D layers to discriminate real from 
  # fake. 
  
  # Arguments:
  # * inputs (Layer): Input layer of the discriminator (the image)
  
  # Returns:
  # * Model: Discriminator Model
  
  discriminator_reshaped <- labels %>%
    keras::layer_dense(units = image_size * image_size) %>%
    keras::layer_reshape(target_shape = base::c(image_size, image_size, 1))
  
  discriminator_concatenated <- keras::layer_concatenate(inputs = base::list(inputs, discriminator_reshaped))
  
  discriminator_output <- discriminator_concatenated %>%
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
  
  discriminator <- keras::keras_model(inputs = base::list(inputs, labels), outputs = discriminator_output)
  
  base::return(discriminator)}

keras::k_clear_session()
test_build_discriminator <- build_discriminator(inputs = keras::layer_input(shape = base::c(28, 28, 1)),
                                                labels = keras::layer_input(shape = base::c(10)),
                                                image_size = 28); test_build_discriminator
deepviz::plot_model(test_build_discriminator)

# ------------------------------------------------------------------------------
# Build and train models:
build_and_train_models <- function(){
  
  # Load MNIST dataset:
  c(c(x_train, y_train), c(x_test, y_test)) %<-% keras::dataset_mnist()
  
  # Reshape data for CNN as (28, 28, 1) and normalize:
  image_size <- base::dim(x_train)[2]
  x_train <- keras::array_reshape(x = x_train, dim = base::c(-1, image_size, image_size, 1))
  x_train <- x_train/255
  
  num_labels <- base::max(y_train) + 1
  y_train <- keras::to_categorical(y_train)
  
  # Network parameters:
  model_name <- "cgan"
  latent_size <- 1000
  batch_size <- 512
  train_steps <- 25000
  lr <- 2e-4
  decay <- 6e-8
  input_shape <- base::c(image_size, image_size, 1)
  label_shape <- base::c(num_labels)
  
  # Build discriminator model:
  optimizer <- keras::optimizer_rmsprop(lr = lr, decay = decay)
  inputs <- keras::layer_input(shape = base::c(input_shape), name = "Discriminator_Input")
  labels <- keras::layer_input(shape = base::c(label_shape), name = "Class_Labels")
  discriminator <- build_discriminator(inputs = inputs,
                                       labels = labels,
                                       image_size = image_size)
  discriminator %>% keras::compile(loss = "binary_crossentropy",
                                   optimizer = optimizer,
                                   metrics = base::c("accuracy"))
  discriminator %>% 
    base::summary()
  
  # Build generator model:
  inputs <- keras::layer_input(shape = base::c(latent_size), name = "Z_Input")
  generator <- build_generator(inputs = inputs,
                               labels = labels,
                               image_size = image_size)
  generator %>% 
    base::summary()
  
  # Build adversarial model:
  optimizer <- keras::optimizer_rmsprop(lr = lr * 0.5, decay = decay * 0.5)
  discriminator$trainable <- FALSE
  
  # Adversarial = generator + discriminator
  outputs <- discriminator(base::list(generator(base::list(inputs, labels)), labels))
  adversarial <- keras::keras_model(inputs = base::list(inputs, labels),
                                    outputs = outputs)
  adversarial %>% 
    keras::compile(loss = "binary_crossentropy",
                   optimizer = optimizer,
                   metrics = base::c("accuracy"))
  adversarial %>%
    base::summary()
  
  # Train the Discriminator and Adversarial Networks:
  models <- base::list(generator, discriminator, adversarial)
  data <- base::list(x_train, y_train)
  params <- base::list(batch_size, latent_size, train_steps, image_size, num_labels, model_name)
  train(models, data, params)}

# ------------------------------------------------------------------------------
# Plot images:
plot_images <- function(generator,
                        noise_input,
                        noise_class,
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
  images <-  generator %>% 
    stats::predict(base::list(noise_input, noise_class))
  
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
                    col = base::as.numeric(col),
                    row = base::as.numeric(row) * (-1)) %>%
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
train <- function(models, data, params){
  
  # Train the Discriminator and Adversarial Networks
  # Alternately train Discriminator and Adversarial networks by batch. Discriminator
  # is trained first with properly real and fake images. Adversarial is trained next 
  # with fake images pretending to be real - generated sample images per save_interval.
  
  # Arguments:
  # * models (list): Generator, Discriminator, Arversarial models
  # * data (list): x_train, y_train data
  # * params (list): Networks parameters
  
  # The GAN Models:
  c(generator, discriminator, adversarial) %<-% models
  
  # Images and labels:
  c(x_train, y_train) %<-% data
  
  # Network parameters:
  c(batch_size, latent_size, train_steps, image_size, num_labels) %<-% params[1:5]
  c(model_name) %<-% params[6]
  
  # The generator image is saved every 500 steps:
  save_interval <- 100
  
  # The noise vector to see how the generator output evolves during training:
  numbers <- 100
  noise_input <- base::matrix(data = stats::runif(n = numbers * latent_size, min = -1, max = 1), nrow = numbers, ncol = latent_size)

  # The class:
  noise_class <- base::diag(num_labels)[(1 + (base::seq(from = 0, to = numbers, by = 1) %% (num_labels)))[1:numbers],]
   
  # Number of elements in train dataset:
  train_size <- base::dim(x_train)[1]

  # Save results:
  discriminator_loss_results <- base::numeric(train_steps)
  discriminator_accuracy_results <- base::numeric(train_steps)
  adversarial_loss_results <- base::numeric(train_steps)
  adversarial_accuracy_results <- base::numeric(train_steps)
  
  for (i in 1:train_steps){
    
    # Train the discriminator for 1 batch: 
    # 1 batch of real (label = 1)
    # 1 batch of fake (label = 0)
    
    # Randomly pick real images from dataset:
    rand_indexes <- base::sample(x = 1:train_size, size = batch_size, replace = FALSE)
    real_images <- x_train[rand_indexes,,,]
    real_images <- keras::array_reshape(x = real_images, dim = base::c(-1, image_size, image_size, 1))

    # Corresponding one_hot labels for real images:
    real_labels <- y_train[rand_indexes,]
    
    # Generate fake images from noise using generator - generate noise using uniform
    # distribution:
    noise <- base::matrix(data = stats::runif(n = batch_size * latent_size, min = -1, max = 1), nrow = batch_size, ncol = latent_size)
    
    # Assign random one-hot labels:
    fake_labels <- base::diag(num_labels)[base::sample(x = 1:num_labels, size = batch_size, replace = TRUE),]
    
    # Generate fake images conditioned on fake labels: 
    fake_images <- generator %>% 
      stats::predict(base::list(noise, fake_labels))

    # real + fake images = 1 batch of train data:
    x <- keras::k_concatenate(tensors = base::list(real_images, fake_images), axis = 1)

    # real + fake one-hot labels = 1 batch of train one-hot labels:
    y_labels <- keras::k_concatenate(tensors = base::list(real_labels, fake_labels), axis = 1)
    
    # Label real and fake images: real images -> 1, fake images -> 0
    y <- base::rep(x = base::c(1, 0), each = batch_size)
    
    # Train discriminator network - log the loss and accuracy:    
    discriminator_result <- discriminator %>% 
      keras::train_on_batch(x = base::list(x, y_labels), y = y)
    c(discriminator_loss, discriminator_acc) %<-% discriminator_result
    discriminator_log <- base::paste0(i, ": Discriminator loss: ", base::round(discriminator_loss, 6), ", Discriminator accuracy: ", base::round(discriminator_acc, 6))
    
    # Train the adversarial network for 1 batch (1 batch of fake images conditioned
    # on fake one-hot labels / label = 1. The discriminator weights are frozen in 
    # adversarial and only generator is trained. Generate noise using uniform 
    # distribution:
    noise <- base::matrix(data = stats::runif(n = batch_size * latent_size, min = -1, max = 1), nrow = batch_size, ncol = latent_size)
    
    # Assign random one-hot labels:
    fake_labels <- base::diag(num_labels)[base::sample(x = 1:num_labels, size = batch_size, replace = TRUE),]
    
    # Label fake images as real = 1:
    y <- base::rep(x = 1, times = batch_size)
    
    # Train the adversarial network. Note that unlike in discriminator training we
    # do not save the fake images in a variable. The fake images go to the 
    # discriminator input of the adversarial for classification. Log the loss 
    # and accuracy:
    adversarial_result <- adversarial %>%
      keras::train_on_batch(x = base::list(noise, fake_labels), y = y)
    c(adversarial_loss, adversarial_acc) %<-% adversarial_result
    adversarial_log <- base::paste0("Adversarial loss: ", base::round(adversarial_loss, 6), ", Adversarial accuracy: ", base::round(adversarial_acc, 6))
    
    # Concatenate logs from Discriminator and Adversarial Network training: 
    log <- base::paste(discriminator_log, adversarial_log, sep = " | ")
    base::print(log)
    
    # Save model results:
    discriminator_loss_results[i] <- discriminator_loss
    discriminator_accuracy_results[i] <- discriminator_acc
    adversarial_loss_results[i] <- adversarial_loss
    adversarial_accuracy_results[i] <- adversarial_acc
    
    if (i %% save_interval == 0){
      plot_images(generator = generator,
                  noise_input = noise_input,
                  noise_class = noise_class,
                  step = i,
                  model_name = model_name)}}
  
  # Save the model after training the generator (.h5 and .hdf5). The trained 
  # generator can be reloaded for future MNIST digit generation.
  generator %>%
    keras::save_model_hdf5(base::paste(model_name, base::paste0(model_name, "_generator.h5"), sep = "/"))
  generator %>%
    keras::save_model_hdf5(base::paste(model_name, base::paste0(model_name, "_generator.hdf5"), sep = "/"))
  
  # Save the discriminator model:
  discriminator %>%
    keras::save_model_hdf5(base::paste(model_name, base::paste0(model_name, "_discriminator.h5"), sep = "/"))
  discriminator %>%
    keras::save_model_hdf5(base::paste(model_name, base::paste0(model_name, "_discriminator.hdf5"), sep = "/"))
  
  # Save results - loss and accuracy for discriminator and adversarial model:
  results <- base::list(discriminator_loss_results = discriminator_loss_results,
                        discriminator_accuracy_results = discriminator_accuracy_results,
                        adversarial_loss_results = adversarial_loss_results,
                        adversarial_accuracy_results = adversarial_accuracy_results)
  
  results <- results %>%
    tibble::as_tibble() %>%
    readr::write_csv(base::paste(model_name, base::paste0(model_name, ".csv"), sep = "/"))}

# ------------------------------------------------------------------------------
# Train DCGAN model:
keras::k_clear_session()
build_and_train_models()

# ------------------------------------------------------------------------------
# Display result:
test_generator <- function(generator, 
                           numbers = 100, 
                           class_label = NULL,
                           save = TRUE,
                           model_name = "cgan"){
  latent_size <- 100
  num_labels <- 10
  noise_input <- base::matrix(data = stats::runif(n = numbers * latent_size, min = -1, max = 1), nrow = numbers, ncol = latent_size)
  
  if (base::is.null(class_label)){
    noise_class <- base::diag(num_labels)[(1 + (base::seq(from = 0, to = numbers, by = 1) %% (num_labels)))[1:numbers],]
  } else {
    noise_class <- base::rep(x = 0, times = numbers)
    noise_class <- base::matrix(data = 0, nrow = numbers, ncol = num_labels)
    noise_class[,class_label + 1] <- 1}
  
  plot_final_images <- function(generator,
                                noise_input,
                                noise_class){
    
    # Predict with generator noise input:
    images <- generator %>%
      predict(base::list(noise_input, noise_class))
    
    # Extract helpful features:
    num_images <- base::dim(images)[1]
    image_size <- base::dim(images)[2]
    rows <- base::ceiling(base::sqrt(base::dim(noise_input)[1]))
    
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
                      col = base::as.numeric(col),
                      row = base::as.numeric(row) * (-1)) %>%
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
    plots <- base::do.call(grid.arrange, base::c(plots, ncol = rows, top = base::paste(model_name, class_label, sep = ": ")))
    base::return(plots)}
  
  plots <- plot_final_images(generator,
                    noise_input = noise_input,
                    noise_class = noise_class)
  
  if (save == TRUE){
    base::setwd(base::paste("D:/GitHub/GANModelsR", model_name, sep = "/"))
    ggplot2::ggsave(base::paste0(base::paste(class_label, model_name, sep = " "), ".png"), plot = plots)
    base::setwd("..")}}

for (i in 1:10){
  k <- (i - 1)
  base::print(base::paste("Generated number:", k))
  
  test_generator(generator = keras::load_model_hdf5("cgan/cgan_generator.hdf5", compile = FALSE),
                 numbers = 100,
                 class_label = k,
                 save = TRUE)}

# ------------------------------------------------------------------------------
# Training visualization:
base::list.files(path = base::paste(base::getwd(), "cgan", sep = "/"), pattern = "*00.png", full.names = TRUE) %>%
  stringr::str_sort(numeric = TRUE) %>%
  magick::image_read() %>%
  magick::image_join() %>%
  magick::image_animate(delay = 0.5, loop = 1) %>%
  # magick::image_write_video(base::paste("cgan", "cgan_training.gif", sep = "/"))
  magick::image_write(base::paste("cgan", "cgan_training.gif", sep = "/"))
# ------------------------------------------------------------------------------
  