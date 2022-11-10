import keras
import keras.utils
import keras.layers as layers
import keras.metrics
import keras.callbacks
import keras.optimizers
import keras.losses
import tensorflow as tf
import numpy as np

dataset = keras.utils.image_dataset_from_directory(
  "img_align_celeba",
  label_mode=None,
  image_size=(64, 64),
  batch_size=64,
  smart_resize=True,
  seed=1,
  subset="training",
  validation_split=0.9)

dataset = dataset.map(lambda x: x / 255.)


latent_dim = 128

discriminator = keras.Sequential([
  keras.Input(shape=(64, 64, 3)),
  layers.Conv2D(64, kernel_size=4, strides=2, padding="same"),
  layers.LeakyReLU(alpha=0.2),
  layers.Conv2D(128, kernel_size=4, strides=2, padding="same"),
  layers.LeakyReLU(alpha=0.2),
  layers.Conv2D(128, kernel_size=4, strides=2, padding="same"),
  layers.LeakyReLU(alpha=0.2),
  layers.Flatten(),
  layers.Dropout(0.2),
  layers.Dense(1, activation="sigmoid")
], name="discriminator")

discriminator.summary()

encoder_inputs = keras.Input(shape=(64, 64, 3))
x = layers.Conv2D(64, 3, activation="relu", strides=2, padding="same")(encoder_inputs)
x = layers.Conv2D(128, 3, activation="relu", strides=2, padding="same")(x)
x = layers.Conv2D(256, 3, activation="relu", strides=2, padding="same")(x)
x = layers.Flatten()(x)
x = layers.Dense(256, activation="relu")(x)
z_mean = layers.Dense(latent_dim, name="z_mean")(x)
z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)
encoder = keras.Model(encoder_inputs, [z_mean, z_log_var], name="encoder")

encoder.summary()

class Sampler(layers.Layer):
  def call(self, z_mean, z_log_var):
    batch_size = tf.shape(z_mean)[0]
    z_size = tf.shape(z_mean)[1]
    epsilon = tf.random.normal(shape=(batch_size, z_size))
    return z_mean + tf.exp(0.5 * z_log_var) * epsilon

generator = keras.Sequential([
  keras.Input(shape=(latent_dim,)),
  layers.Dense(8 * 8 * 128),
  layers.Reshape((8, 8, 128)),
  layers.Conv2DTranspose(128, kernel_size=4, strides=2, padding="same"),
  layers.LeakyReLU(alpha=0.2),
  layers.Conv2DTranspose(256, kernel_size=4, strides=2, padding="same"),
  layers.LeakyReLU(alpha=0.2),
  layers.Conv2DTranspose(512, kernel_size=4, strides=2, padding="same"),
  layers.LeakyReLU(alpha=0.2),
  layers.Conv2D(3, kernel_size=5, padding="same", activation="sigmoid"),
], name="generator")

generator.summary()


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
      reconstruction = self.decoder(z)
      reconstruction_loss = tf.reduce_mean(
        tf.reduce_sum(keras.losses.binary_crossentropy(data, reconstruction), axis=(1, 2))
      )
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
      img.save(f"generated-vae-gan/generated_img_vae_{epoch:03d}_{i}.png")


class GAN(keras.Model):
  def __init__(self, discriminator, generator, latent_dim):
    super().__init__()
    self.discriminator = discriminator
    self.generator = generator
    self.latent_dim = latent_dim
    self.d_loss_metric = keras.metrics.Mean(name="d_loss")
    self.g_loss_metric = keras.metrics.Mean(name="g_loss")

  def compile(self, d_optimizer, g_optimizer, loss_fn):
    super(GAN, self).compile()
    self.d_optimizer = d_optimizer
    self.g_optimizer = g_optimizer
    self.loss_fn = loss_fn

  @property
  def metrics(self):
    return [self.d_loss_metric, self.g_loss_metric]

  def train_step(self, real_images):
    batch_size = tf.shape(real_images)[0]
    random_latent_vectors = tf.random.normal(shape=(batch_size, self.latent_dim))
    generated_images = self.generator(random_latent_vectors)
    combined_images = tf.concat([generated_images, real_images], axis=0)
    labels = tf.concat([tf.ones((batch_size, 1)), tf.zeros((batch_size, 1))], axis=0)
    labels += 0.05 * tf.random.uniform(tf.shape(labels))

    with tf.GradientTape() as tape:
      predictions = self.discriminator(combined_images)
      d_loss = self.loss_fn(labels, predictions)
    grads = tape.gradient(d_loss, self.discriminator.trainable_weights)
    self.d_optimizer.apply_gradients(zip(grads, self.discriminator.trainable_weights))

    random_latent_vectors = tf.random.normal(shape=(batch_size, self.latent_dim))

    misleading_labels = tf.zeros((batch_size, 1))

    with tf.GradientTape() as tape:
      predictions = self.discriminator(self.generator(random_latent_vectors))
      g_loss = self.loss_fn(misleading_labels, predictions)
    grads = tape.gradient(g_loss, self.generator.trainable_weights)
    self.g_optimizer.apply_gradients(zip(grads, self.generator.trainable_weights))

    self.d_loss_metric.update_state(d_loss)
    self.g_loss_metric.update_state(g_loss)
    return {
      "d_loss": self.d_loss_metric.result(),
      "g_loss": self.g_loss_metric.result()
    }

class GANMonitor(keras.callbacks.Callback):
  def __init__(self, num_img=3, latent_dim=128):
    self.num_img = num_img
    self.latent_dim = latent_dim

  def on_epoch_end(self, epoch, logs=None):
    random_latent_vectors = tf.random.normal(shape=(self.num_img, self.latent_dim))
    generated_images = self.model.generator(random_latent_vectors)
    generated_images *= 255
    generated_images.numpy()
    for i in range(self.num_img):
      img = keras.utils.array_to_img(generated_images[i])
      img.save(f"generated-vae-gan/generated_img_gan_{epoch:03d}_{i}.png")

callbacks = [Monitor(3, latent_dim)]

vae = VAE(encoder, generator)
gan = GAN(discriminator, generator, latent_dim)

vae.compile(optimizer=keras.optimizers.Adam(), run_eagerly=True)
gan.compile(
  d_optimizer=keras.optimizers.Adam(learning_rate=0.0001),
  g_optimizer=keras.optimizers.Adam(learning_rate=0.0001),
  loss_fn=keras.losses.BinaryCrossentropy())

num_img = 3
vae_callbacks = [Monitor(3, latent_dim)]
gan_callbacks = [GANMonitor(3, latent_dim)]

vae.fit(dataset, epochs=2, batch_size=64, callbacks=vae_callbacks)
gan.fit(dataset, epochs=100, batch_size=64, callbacks=gan_callbacks)


