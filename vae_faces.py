# From deep learning with keras 2nd edition

import keras
import keras.utils
import keras.layers as layers
import keras.metrics
import keras.callbacks
import keras.optimizers
import keras.losses
import tensorflow as tf
import numpy as np
from os.path import exists

latent_dim = 16
version = 1

encoder_model_path = f"vae_faces_encoder_{latent_dim}_v{version}.keras"
decoder_model_path = f"vae_faces_decoder_{latent_dim}_v{version}.keras"

dataset = keras.utils.image_dataset_from_directory(
  "img_align_celeba",
  label_mode=None,
  image_size=(64, 64),
  batch_size=64,
  smart_resize=True,
  seed=1,
  subset="training",
  validation_split=0.1)

dataset = dataset.map(lambda x: x / 255.)


def get_encoder():
  if exists(encoder_model_path):
    return keras.models.load_model(encoder_model_path)

  encoder_inputs = keras.Input(shape=(64, 64, 3))
  x = layers.Conv2D(64, 3, activation="relu", strides=2, padding="same")(encoder_inputs)
  x = layers.Conv2D(128, 3, activation="relu", strides=2, padding="same")(x)
  x = layers.Conv2D(256, 3, activation="relu", strides=2, padding="same")(x)
  x = layers.Flatten()(x)
  x = layers.Dense(256, activation="relu")(x)
  z_mean = layers.Dense(latent_dim, name="z_mean")(x)
  z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)
  return keras.Model(encoder_inputs, [z_mean, z_log_var], name="encoder")

encoder = get_encoder()
encoder.summary()

class Sampler(layers.Layer):
  def call(self, z_mean, z_log_var):
    batch_size = tf.shape(z_mean)[0]
    z_size = tf.shape(z_mean)[1]
    epsilon = tf.random.normal(shape=(batch_size, z_size))
    return z_mean + tf.exp(0.5 * z_log_var) * epsilon

def get_decoder():
  if exists(decoder_model_path):
    return keras.models.load_model(decoder_model_path)

  latent_inputs = keras.Input(shape=(latent_dim,))
  x = layers.Dense(8 * 8 * 128, activation="relu")(latent_inputs)
  x = layers.Reshape((8, 8, 128))(x)
  x = layers.Conv2DTranspose(128, 3, activation="relu", strides=2, padding="same")(x)
  x = layers.Conv2DTranspose(256, 3, activation="relu", strides=2, padding="same")(x)
  x = layers.Conv2DTranspose(512, 3, activation="relu", strides=2, padding="same")(x)
  decoder_outputs = layers.Conv2D(3, kernel_size=5, activation="sigmoid", padding="same")(x)
  return keras.Model(latent_inputs, decoder_outputs, name="decoder")


decoder = get_decoder()
decoder.summary()


class VAE(keras.Model):
  def __init__(self, encoder, decoder, **kwargs):
    super().__init__(**kwargs)
    self.encoder = encoder
    self.decoder = decoder
    self.sampler = Sampler()
    self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
    self.reconstruction_loss_tracker = keras.metrics.Mean(name="reconstruction_loss")
    self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")

  @property
  def metrics(self):
    return [
      self.total_loss_tracker,
      self.reconstruction_loss_tracker,
      self.kl_loss_tracker
    ]

  def train_step(self, data):
    with tf.GradientTape() as tape:
      z_mean, z_log_var = self.encoder(data)
      z = self.sampler(z_mean, z_log_var)
      reconstruction = decoder(z)
      reconstruction_mse = keras.losses.mse(data, reconstruction)
      reconstruction_sum_loss = tf.reduce_sum(reconstruction_mse, axis=(1, 2))
      reconstruction_loss = tf.reduce_mean(reconstruction_sum_loss)
      kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
      total_loss = reconstruction_loss + tf.reduce_mean(kl_loss)
    grads = tape.gradient(total_loss, self.trainable_weights)
    self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
    self.total_loss_tracker.update_state(total_loss)
    self.reconstruction_loss_tracker.update_state(reconstruction_loss)
    self.kl_loss_tracker.update_state(kl_loss)

    return {
      "total_loss": self.total_loss_tracker.result(),
      "reconstruction_loss": self.reconstruction_loss_tracker.result(),
      "kl_loss": self.kl_loss_tracker.result()
    }


class Monitor(keras.callbacks.Callback):
  def __init__(self, num_img, latent_dim):
    self.num_img = num_img
    self.latent_dim = latent_dim

  def on_epoch_end(self, epoch, logs=None):
    random_latent_vectors = tf.random.normal(shape=(self.num_img, self.latent_dim))
    generated_images = self.model.decoder(random_latent_vectors)
    generated_images *= 255
    generated_images.numpy()
    for i in range(self.num_img):
      img = keras.utils.array_to_img(generated_images[i])
      img.save(f"generated-vae-{latent_dim}-v{version}/generated_img_{epoch:03d}_{i}.png")

class Save(keras.callbacks.Callback):
  def __init__(self, decoder_path, encoder_path):
    self.decoder_path = decoder_path
    self.encoder_path = encoder_path

  def on_epoch_end(self, epoch, logs=None):
    self.model.decoder.save(self.decoder_path)
    self.model.encoder.save(self.encoder_path)

callbacks = [Monitor(3, latent_dim), Save(decoder_model_path, encoder_model_path)]

vae = VAE(encoder, decoder)
vae.compile(optimizer=keras.optimizers.Adam(), run_eagerly=True)
vae.fit(dataset, epochs=100, batch_size=64, callbacks=callbacks)

