# -*- coding:utf-8 -*-
import tensorflow as tf

# Predefine the Foreground and background Segmentation

# Encoder
def conv2d_block(input_tensor, n_filters, kernel_size=3):
    '''
    Add 2 convolutional layers with the parameters
    '''
    x = input_tensor
    for i in range(2):
        x = tf.keras.layers.Conv2D(filters=n_filters, kernel_size=kernel_size,
        activation='relu', padding='same')(x)
    return x

def encoder_block(inputs, n_filters=64, pool_size=(2,2), dropout=0.3):
    '''
    Add 2 convolutional blocks and then perform down sampling on output of convolutions
    '''
    f = conv2d_block(inputs, n_filters)
    p = tf.keras.layers.MaxPooling2D(pool_size=pool_size)(f)
    p = tf.keras.layers.Dropout(dropout)(p)
    return f, p

def encoder(inputs):
    '''
    defines the encoder or downsampling path.
    '''
    f1, p1 = encoder_block(inputs, n_filters=64)
    f2, p2 = encoder_block(p1, n_filters=128)
    f3, p3 = encoder_block(p2, n_filters=256)
    f4, p4 = encoder_block(p3, n_filters=512)
    return p4, (f1, f2, f3, f4)

# Bottlenect
def bottleneck(inputs):
    bottle_neck = conv2d_block(inputs, n_filters=1024)
    return bottle_neck

# Decoder
def decoder_block(inputs, conv_output, n_filters=64, kernel_size=3, strides=3, dropout=0.3):
    '''
    defines the one decoder block of the UNet
    '''
    u = tf.keras.layers.Conv2DTranspose(n_filters, kernel_size, strides, padding='same')(inputs)
    c = tf.keras.layers.concatenate([u, conv_output])
    c = tf.keras.layers.Dropout(dropout)(c)
    c = conv2d_block(c, n_filters)
    return c

def decoder(inputs, convs, output_channels):
    '''
    Defines the decoder of the UNet chaining together 4 decoder blocks.
    '''
    f1, f2, f3, f4 = convs
    c6 = decoder_block(inputs, f4, n_filters=512, kernel_size=3, strides=2)
    c7 = decoder_block(c6, f3, n_filters=256, kernel_size=3, strides=2)
    c8 = decoder_block(c7, f2, n_filters=128, kernel_size=3, strides=2)
    c9 = decoder_block(c8, f1, n_filters=64, kernel_size=3, strides=2)
    outputs = tf.keras.layers.Conv2D(output_channels, 1, activation='softmax')(c9)
    return outputs



def PFB_model(input_shape=(256, 256, 3), OUTPUT_CHANNELS=2):

    '''
    Defines the UNet by connecting the encoder, bottleneck and decoder
    '''
    inputs = tf.keras.layers.Input(input_shape)
    encoder_output, convs = encoder(inputs)
    bottle_neck = bottleneck(encoder_output)
    outputs = decoder(bottle_neck, convs, OUTPUT_CHANNELS)
    model = tf.keras.Model(inputs, outputs)
    return model

model = PFB_model()
model.summary()

def get_weights(img_size, model):
    pre_trained_model = tf.keras.applications.VGG16(input_shape=(img_size, img_size, 3), include_top=False)
    model.get_layer("local_1_conv1").set_weights(pre_trained_model.get_layer("block1_conv1").get_weights())
    model.get_layer("local_2_conv1").set_weights(pre_trained_model.get_layer("block1_conv1").get_weights())
    model.get_layer("local_3_conv1").set_weights(pre_trained_model.get_layer("block1_conv1").get_weights())
    model.get_layer("local_4_conv1").set_weights(pre_trained_model.get_layer("block1_conv1").get_weights())
    model.get_layer("global_1_conv1").set_weights(pre_trained_model.get_layer("block1_conv1").get_weights())

    model.get_layer("local_1_conv2").set_weights(pre_trained_model.get_layer("block1_conv2").get_weights())
    model.get_layer("local_2_conv2").set_weights(pre_trained_model.get_layer("block1_conv2").get_weights())
    model.get_layer("local_3_conv2").set_weights(pre_trained_model.get_layer("block1_conv2").get_weights())
    model.get_layer("local_4_conv2").set_weights(pre_trained_model.get_layer("block1_conv2").get_weights())
    model.get_layer("global_1_conv2").set_weights(pre_trained_model.get_layer("block1_conv2").get_weights())

    model.get_layer("local_1_conv3").set_weights(pre_trained_model.get_layer("block2_conv1").get_weights())
    model.get_layer("local_2_conv3").set_weights(pre_trained_model.get_layer("block2_conv1").get_weights())
    model.get_layer("local_3_conv3").set_weights(pre_trained_model.get_layer("block2_conv1").get_weights())
    model.get_layer("local_4_conv3").set_weights(pre_trained_model.get_layer("block2_conv1").get_weights())
    model.get_layer("global_1_conv3").set_weights(pre_trained_model.get_layer("block2_conv1").get_weights())

    model.get_layer("local_1_conv4").set_weights(pre_trained_model.get_layer("block2_conv2").get_weights())
    model.get_layer("local_2_conv4").set_weights(pre_trained_model.get_layer("block2_conv2").get_weights())
    model.get_layer("local_3_conv4").set_weights(pre_trained_model.get_layer("block2_conv2").get_weights())
    model.get_layer("local_4_conv4").set_weights(pre_trained_model.get_layer("block2_conv2").get_weights())
    model.get_layer("global_1_conv4").set_weights(pre_trained_model.get_layer("block2_conv2").get_weights())

    model.get_layer("local_1_conv5").set_weights(pre_trained_model.get_layer("block3_conv1").get_weights())
    model.get_layer("local_2_conv5").set_weights(pre_trained_model.get_layer("block3_conv1").get_weights())
    model.get_layer("local_3_conv5").set_weights(pre_trained_model.get_layer("block3_conv1").get_weights())
    model.get_layer("local_4_conv5").set_weights(pre_trained_model.get_layer("block3_conv1").get_weights())
    model.get_layer("global_1_conv5").set_weights(pre_trained_model.get_layer("block3_conv1").get_weights())

    model.get_layer("local_1_conv6").set_weights(pre_trained_model.get_layer("block3_conv2").get_weights())
    model.get_layer("local_2_conv6").set_weights(pre_trained_model.get_layer("block3_conv2").get_weights())
    model.get_layer("local_3_conv6").set_weights(pre_trained_model.get_layer("block3_conv2").get_weights())
    model.get_layer("local_4_conv6").set_weights(pre_trained_model.get_layer("block3_conv2").get_weights())
    model.get_layer("global_1_conv6").set_weights(pre_trained_model.get_layer("block3_conv2").get_weights())

    model.get_layer("local_1_conv7").set_weights(pre_trained_model.get_layer("block3_conv3").get_weights())
    model.get_layer("local_2_conv7").set_weights(pre_trained_model.get_layer("block3_conv3").get_weights())
    model.get_layer("local_3_conv7").set_weights(pre_trained_model.get_layer("block3_conv3").get_weights())
    model.get_layer("local_4_conv7").set_weights(pre_trained_model.get_layer("block3_conv3").get_weights())
    model.get_layer("global_1_conv7").set_weights(pre_trained_model.get_layer("block3_conv3").get_weights())

    return model

#import matplotlib.pyplot as plt

#img = tf.io.read_file("D:/[1]DB/[5]4th_paper_DB/CAMO/CAMO-COCO-V.1.0/CAMO-COCO-V.1.0-CVIU2019/Camouflage/Images/Train/camourflage_00001.jpg")
#img = tf.image.decode_jpeg(img, 3)
#img = tf.image.resize(img, [256, 256]) / 255.
#img1 = img[0:128, 0:128, :]
#img2 = img[0:128:, 128:, :]
#img3 = img[128:, 0:128, :]
#img4 = img[128:, 128:, :]

#h_col_1 = tf.concat([img1, img3], axis=0) # [256, 128, 3]
#h_col_2 = tf.concat([img2, img4], axis=0) # [256, 128, 3]
#h = tf.concat([h_col_1, h_col_2], axis=1)

#plt.imshow(img1)
#plt.show()
#plt.imshow(img2)
#plt.show()
#plt.imshow(img3)
#plt.show()
#plt.imshow(img4)
#plt.show()
#plt.imshow(h)
#plt.show()