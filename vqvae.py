import keras
import keras.losses
import keras.optimizers
import keras.layers as layers
import tensorflow as tf

image_size = (64, 64)
embedding_length = 512
embedding_dim = 1
runeager = False

class VQVAE(keras.models.Model):
  def __init__(self, encoder, decoder, embedding_dim, embedding_length, beta=0.25, embedding_learning_rate=0.0001, **kwargs):
    super().__init__(**kwargs)
    self.total_loss_tracker = keras.metrics.Mean("total_loss")
    self.decoder_loss_tracker = keras.metrics.Mean("decoder_loss")
    self.encoder_loss_tracker = keras.metrics.Mean("encoder_loss")
    self.embedding_loss_tracker = keras.metrics.Mean("embedding_loss")
    self.encoder = encoder
    self.decoder = decoder
    self.latent_embedding = tf.Variable(tf.random.uniform(shape=(embedding_length, embedding_dim)))
    self.embedding_length = embedding_length
    self.embedding_dim = embedding_dim
    self.decode_loss_fn = keras.losses.MSE
    self.beta = beta
    self.decoder_optimizer = keras.optimizers.RMSprop()
    self.encoder_optimizer = keras.optimizers.RMSprop()
    self.embedding_optimizer = keras.optimizers.RMSprop()
    self.embedding_learning_rate = embedding_learning_rate

    assert encoder.output.shape == decoder.input.shape

  @property
  def metrics(self):
    return [
      self.total_loss_tracker,
      self.decoder_loss_tracker,
      self.encoder_loss_tracker,
      self.embedding_loss_tracker
    ]

  def train_step(self, real_images):
    # Encode
    with tf.GradientTape(persistent=True) as tape:
      encoded = self.encoder(real_images)
      (_, w, h, c) = encoded.shape
      B = tf.shape(real_images)[0]
      assert c == self.embedding_dim

      # Map to latent embedding
      encoded_flat = tf.reshape(encoded, shape=(B * w * h, c))
      encoded_flat = tf.tile(encoded_flat, [1, self.embedding_length])
      encoded_flat = tf.reshape(encoded_flat, shape=(B * w * h, self.embedding_length, c))
      diff = tf.pow(encoded_flat - self.latent_embedding, 2)
      diff = tf.reduce_sum(diff, axis=-1)
      embedding_indexes = tf.argmin(diff, axis=-1)
      embedding_indexes = tf.reshape(embedding_indexes, shape=(B, w, h))
      encoding_vectors = tf.gather(self.latent_embedding, embedding_indexes)

      # Straight through estimator
      decoded = self.decoder(encoded + tf.stop_gradient(encoding_vectors - encoded))

      decoder_loss = self.decode_loss_fn(real_images, decoded)
      embedding_loss = tf.reduce_mean(tf.pow(tf.stop_gradient(encoded) - encoding_vectors, 2))
      encoder_loss = decoder_loss + self.beta * tf.reduce_mean(tf.pow(encoded - tf.stop_gradient(encoding_vectors), 2))

    # Optimize decoder
    grads = tape.gradient(decoder_loss, self.decoder.trainable_weights)
    self.decoder_optimizer.apply_gradients(zip(grads, self.decoder.trainable_weights))
    self.decoder_loss_tracker.update_state(decoder_loss)

    # Optimize encoder
    encoder_grads = tape.gradient(encoder_loss, self.encoder.trainable_weights)
    self.encoder_optimizer.apply_gradients(zip(encoder_grads, self.encoder.trainable_weights))
    self.encoder_loss_tracker.update_state(encoder_loss)

    # Optimize embedding space
    embedding_grads = tape.gradient(embedding_loss, self.latent_embedding)
    self.embedding_optimizer.apply_gradients([(embedding_grads, self.latent_embedding)])
    # self.latent_embedding += self.embedding_learning_rate * (encoded - encoding_vectors)
    self.embedding_loss_tracker.update_state(embedding_loss)

    total_loss = encoder_loss + embedding_loss
    self.total_loss_tracker.update_state(total_loss)

    return {
      "total_loss": self.total_loss_tracker.result(),
      "decoder_loss": self.decoder_loss_tracker.result(),
      "encoder_loss": self.encoder_loss_tracker.result(),
      "embedding_loss": self.embedding_loss_tracker.result(),
    }

  def encode(self, images):
    encoded = self.encoder(images)
    (B, w, h, c) = encoded.shape
    encoded_flat = tf.reshape(encoded, shape=(B * w * h, c))
    encoded_flat = tf.tile(encoded_flat, [1, self.embedding_length])
    encoded_flat = tf.reshape(encoded_flat, shape=(B * w * h, self.embedding_length, c))
    diff = tf.pow(encoded_flat - self.latent_embedding, 2)
    diff = tf.reduce_sum(diff, axis=-1)
    embedding_indexes = tf.argmin(diff, axis=-1)
    embedding_indexes = tf.reshape(embedding_indexes, shape=(B, w, h))
    return embedding_indexes

  def decode(self, embedding_indexes):
    encoding_vectors = tf.gather(self.latent_embedding, embedding_indexes)
    return decoder(encoding_vectors)

def conv_block(filters, x):
  x = layers.Conv2D(filters, kernel_size=3, padding="same", activation="relu", strides=2)(x)
  x = layers.Dropout(0.2)(x)
  x = layers.Conv2D(filters, kernel_size=3, padding="same", activation="relu")(x)
  return x


encoder_inputs = keras.Input(shape=(*image_size, 3))
x = layers.Conv2D(8, kernel_size=3, padding="same", activation="relu")(encoder_inputs)
#x = conv_block(16, x)
x = conv_block(32, x)
x = conv_block(64, x)
encoder_output = layers.Conv2D(embedding_dim, kernel_size=3, padding="same")(x)

encoder = keras.models.Model(encoder_inputs, encoder_output)

decoder_input = keras.Input(shape=(16, 16, embedding_dim))
x = layers.Conv2D(64, 3, activation="relu", padding="same")(decoder_input)
x = layers.Conv2D(64, 3, activation="relu", padding="same")(x)
x = layers.Conv2DTranspose(32, 4, strides=2, padding="same")(x)
x = layers.Conv2DTranspose(16, 4, strides=2, padding="same")(x)
# x = layers.Conv2DTranspose(8, 4, strides=2, padding="same")(x)
decoder_output = layers.Conv2D(3, 3, padding="same")(x)

decoder = keras.models.Model(decoder_input, decoder_output)

assert encoder.output.shape == decoder.input.shape

vqvae = VQVAE(encoder, decoder, embedding_dim, embedding_length)
vqvae.compile(run_eagerly=runeager)

dataset = keras.utils.image_dataset_from_directory(
  "img_align_celeba_small",
  label_mode=None,
  image_size=image_size,
  batch_size=16,
  smart_resize=True,
  )

class VQVaeMonitor(keras.callbacks.Callback):
  def __init__(self, test_ds):
    self.test_ds = test_ds

  def on_epoch_end(self, epoch, logs=None):
    encoded = self.model.encode(self.test_ds)
    decoded = self.model.decode(encoded)
    decoded.numpy()
    for i in range(len(decoded)):
      img = keras.utils.array_to_img(decoded[i])
      img.save(f"generated-vqvae/autoencoded_{epoch:03d}_{i}.png")

dataset = dataset.take(30000)
test_size = 5
dataset = dataset.map(lambda x: x / (255. / 2) - 1.)
test_ds = dataset.take(test_size).as_numpy_iterator().next()[:5]
dataset = dataset.skip(test_size)
for i in range(len(test_ds)):
  img = keras.utils.array_to_img(test_ds[i])
  img.save(f"generated-vqvae/aaref_{i}.png")

callbacks = [
  VQVaeMonitor(test_ds),
  
]
vqvae.fit(dataset, epochs=500, callbacks=callbacks)