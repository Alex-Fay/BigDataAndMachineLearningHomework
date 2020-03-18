# Style Transfer Tensorflow

Tutorial can be found here: https://www.tensorflow.org/tutorials/generative/style_transfer
I have updated the readme for the new style transfer version (simpler) with th link above. The attached
code is for the old version (still compiles, but has a far longer run time then the link above.)

## Simplified Overview
Skipping a few basic steps (please see code) including inputs and full model setup, the general structure is explained below. 

  ### SetUp:
Using Keras' VGG19, a pretrained classification model, import image weights without a classification head.
This will give you acess to the following:
```
input_2
block1_conv1
block1_conv2
block1_pool
block2_conv1
block2_conv2
block2_pool
block3_conv1
block3_conv2
block3_conv3
block3_conv4
block3_pool
block4_conv1
block4_conv2
block4_conv3
block4_conv4
block4_pool
block5_conv1
block5_conv2
block5_conv3
block5_conv4
block5_pool
```

Using any of the blocks above, asgin one block to content layers and multiple for style. 

  ### Build Layers:

```python
def vgg_layers(layer_names):
  """ Creates a vgg model that returns a list of intermediate output values."""
  vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet')
  vgg.trainable = False
  
  outputs = [vgg.get_layer(name).output for name in layer_names]

  model = tf.keras.Model([vgg.input], outputs)
  return model
  style_extractor = vgg_layers(style_layers)
  style_outputs = style_extractor(style_image*255)
 ```
 
  ### Calculate Style
  
The style of an image is equal to the mean and correlations across the different feature maps of the image. We can use the Gram Matrix to get this information. This is found by taking the summation of the square feature weights and averaging the values by the feature's outer product.

![alt text]("GramMatrix.png")

```python
def gram_matrix(input_tensor):
  result = tf.linalg.einsum('bijc,bijd->bcd', input_tensor, input_tensor)
  input_shape = tf.shape(input_tensor)
  num_locations = tf.cast(input_shape[1]*input_shape[2], tf.float32)
  return result/(num_locations)
  ```
  
  ### Run Gradient Descent
Calculate the mean square error compared to the style image and getting the weighted sum.

```python
def style_content_loss(outputs):
    style_outputs = outputs['style']
    content_outputs = outputs['content']
    style_loss = tf.add_n([tf.reduce_mean((style_outputs[name]-style_targets[name])**2) 
                           for name in style_outputs.keys()])
    style_loss *= style_weight / num_style_layers

    content_loss = tf.add_n([tf.reduce_mean((content_outputs[name]-content_targets[name])**2) 
                             for name in content_outputs.keys()])
    content_loss *= content_weight / num_content_layers
    loss = style_loss + content_loss
    return loss
  ```
  
  Then update te new image.
  ```python
  @tf.function()
def train_step(image):
  with tf.GradientTape() as tape:
    outputs = extractor(image)
    loss = style_content_loss(outputs)

  grad = tape.gradient(loss, image)
  opt.apply_gradients([(grad, image)])
  image.assign(clip_0_1(image))
  ```

Run time takes about 4 hours for completion of 1000 iterations. You're done!
