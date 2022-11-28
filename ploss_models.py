import keras.models
import keras.layers as layers
from keras.applications.convnext import LayerScale
from keras.applications.convnext import ConvNeXtTiny
import tensorflow as tf

perceptual_loss_model_path = "ploss_v1.keras"

def convnext():
  convnext = ConvNeXtTiny(weights='imagenet')
  input_layer = convnext.get_layer("input_1")
  perception_layer_names = [
    "convnext_tiny_stage_0_block_0_depthwise_conv",
    "convnext_tiny_stage_0_block_1_depthwise_conv",
    "convnext_tiny_stage_0_block_2_depthwise_conv",

    "convnext_tiny_stage_1_block_0_depthwise_conv",
    "convnext_tiny_stage_1_block_1_depthwise_conv",
    "convnext_tiny_stage_1_block_2_depthwise_conv",

    "convnext_tiny_stage_2_block_0_depthwise_conv",
    "convnext_tiny_stage_2_block_1_depthwise_conv",
    "convnext_tiny_stage_2_block_2_depthwise_conv",

    # "convnext_tiny_stage_0_block_0_identity",
    # "convnext_tiny_stage_0_block_1_identity",
    # "convnext_tiny_stage_0_block_2_identity",

    # "convnext_tiny_stage_1_block_0_identity",
    # "convnext_tiny_stage_1_block_1_identity",
    # "convnext_tiny_stage_1_block_2_identity",
    
    # "convnext_tiny_stage_2_block_0_identity",
    # "convnext_tiny_stage_2_block_1_identity",
    # "convnext_tiny_stage_2_block_2_identity",
  ]

  perception_layers = [convnext.get_layer(name) for name in perception_layer_names]
  perception_layer_outputs = [layer.output for layer in perception_layers]
  perception_activation = keras.models.Model(input_layer.input, perception_layer_outputs)
  perception_activation.summary()
  return perception_activation

def trained_convnext():
  perceptual_loss_model = keras.models.load_model(perceptual_loss_model_path, custom_objects={
      "LayerScale": LayerScale
  })
  return perceptual_loss_model