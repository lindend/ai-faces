import datetime
from genericpath import exists
import keras
import keras.optimizers
import keras.backend as K
import keras.layers as layers
import keras.losses as losses
import keras.callbacks as callbacks
import keras.utils
from keras.applications.convnext import ConvNeXtTiny
import tensorflow as tf
import numpy as np

version = 1
model_path = f"ploss_v{version}.keras"
runeager = False
convnext_input_size = (224, 224)
batch_size = 8
epochs = 200
log_dir = "logs/ploss"
data_root_dir = "data-ploss"

def get_perception_model(model_name):
  convnext = ConvNeXtTiny(weights='imagenet')
  input_layer = convnext.layers[0]
  perception_layer_names = [
    "convnext_tiny_stage_0_block_0_depthwise_conv",
    #"convnext_tiny_stage_0_block_0_identity",
    "convnext_tiny_stage_0_block_1_depthwise_conv",
    #"convnext_tiny_stage_0_block_1_identity",
    "convnext_tiny_stage_0_block_2_depthwise_conv",
    #"convnext_tiny_stage_0_block_2_identity",

    "convnext_tiny_stage_1_block_0_depthwise_conv",
    #"convnext_tiny_stage_1_block_0_identity",
    "convnext_tiny_stage_1_block_1_depthwise_conv",
    #"convnext_tiny_stage_1_block_1_identity",
    "convnext_tiny_stage_1_block_2_depthwise_conv",
    #"convnext_tiny_stage_1_block_2_identity",
    
    "convnext_tiny_stage_2_block_0_depthwise_conv",
    #"convnext_tiny_stage_2_block_0_identity",
    "convnext_tiny_stage_2_block_1_depthwise_conv",
    #"convnext_tiny_stage_2_block_1_identity",
    "convnext_tiny_stage_2_block_2_depthwise_conv",
    #"convnext_tiny_stage_2_block_2_identity",
  ]
  perception_layers = [layers.UnitNormalization(axis=-1)(convnext.get_layer(name).input) for name in perception_layer_names]
  # perception_layer_outputs = [layer.output for layer in perception_layers]
  perception_model = keras.models.Model(input_layer.input, perception_layers)
  for layer in perception_model.layers:
    layer._name = f"{model_name}_{layer.name}"
  perception_model.trainable = False
  return perception_model

original_model = get_perception_model("convnext_original")
generated_model = get_perception_model("convnext_generated")

input_shape = original_model.input_shape
inputs = [original_model.input, generated_model.input]
outputs = zip(original_model.output, generated_model.output)
xs = [layers.Subtract()([x, y]) for (x, y) in outputs]
xs = [layers.Reshape(target_shape=(x.shape[1], x.shape[2], x.shape[3], 1))(x) for x in xs]
xs = [layers.GlobalAveragePooling3D()(x) for x in xs]
xs = [layers.Reshape(target_shape=(1,))(x) for x in xs]
x = layers.Concatenate()(xs)
loss = layers.Dense(1, use_bias=False)(x)

ploss_model = keras.Model(inputs=inputs, outputs=loss)

ploss_model.summary()

ploss_model.compile(loss=losses.mse, run_eagerly=runeager)

def load_judgement_file(path):
  path = path.numpy().decode("utf-8")
  return np.load(path)[0]


@tf.autograph.experimental.do_not_convert
def load_img_entry(judge_path, type):
  img_path = tf.strings.regex_replace(judge_path, ".npy", ".png")
  img_path = tf.strings.regex_replace(img_path, "judge", type)
  img = tf.io.read_file(img_path)
  img = tf.image.decode_image(img, dtype = tf.float32, channels=3) / (256.0 / 2) - 1
  img = tf.image.resize_with_crop_or_pad(img, convnext_input_size[0], convnext_input_size[1])
  return img

def load_dataset(name):
  files_ds = tf.data.Dataset.list_files(f"{data_root_dir}/{name}/*/judge/*")
  judgements = files_ds.map(lambda path: tf.py_function(load_judgement_file, [path], tf.float32))
  p0_labels = judgements
  p1_labels = judgements.map(lambda x: 1-x)

  refs = files_ds.map(lambda path: load_img_entry(path, "ref"))
  p0 = files_ds.map(lambda path: load_img_entry(path, "p0"))
  p1 = files_ds.map(lambda path: load_img_entry(path, "p1"))
  
  p0_ds = tf.data.Dataset.zip(((refs, p0), p0_labels))
  p1_ds = tf.data.Dataset.zip(((refs, p1), p1_labels))

  files_ds = tf.data.Dataset.concatenate(p0_ds, p1_ds)
  files_ds = files_ds.batch(batch_size)
  return files_ds
  
ds = load_dataset("train")
val_ds = load_dataset("val")

callbacks=[
  callbacks.ModelCheckpoint(model_path, save_best_only=True),
  callbacks.TensorBoard(log_dir)
]
ploss_model.fit(ds, validation_data=val_ds, batch_size=batch_size, epochs=epochs, callbacks=callbacks)


