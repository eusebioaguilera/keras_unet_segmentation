
"""
## Prepare paths of input images and target segmentation masks
"""

import os
import argparse
import random

"""
## Prepare `Sequence` class to load & vectorize batches of data
"""

from tensorflow import keras
import numpy as np
from tensorflow.keras.preprocessing.image import load_img

"""
## Prepare U-Net Xception-style model
"""

from tensorflow.keras import layers

import logging

from config import img_size, num_classes

class SegClass(keras.utils.Sequence):
    """
    This class contains the definition for image loading during the training process 
    """

    def __init__(self, batch_size, img_size, input_img_paths, target_img_paths):
        self.batch_size = batch_size
        self.img_size = img_size
        self.input_img_paths = input_img_paths
        self.target_img_paths = target_img_paths

    def __len__(self):
        return len(self.target_img_paths) // self.batch_size

    def __getitem__(self, idx):
        """
        This method returns the tuple images (input, target) based on the idx of the image batch
        """
        i = idx * self.batch_size
        batch_input_img_paths = self.input_img_paths[i : i + self.batch_size]
        batch_target_img_paths = self.target_img_paths[i : i + self.batch_size]
        x = np.zeros((self.batch_size,) + self.img_size + (3,), dtype="float32")
        for j, path in enumerate(batch_input_img_paths):
            # Loads the image and pass it
            img = load_img(path, target_size=self.img_size)
            x[j] = img
        y = np.zeros((self.batch_size,) + self.img_size + (1,), dtype="uint8")
        for j, path in enumerate(batch_target_img_paths):
            # Load the image and process before pass it
            img = load_img(path, target_size=self.img_size, color_mode="grayscale")
            y[j] = np.expand_dims(img, 2)
            # Labels starts from 1 so subtract one to make start from 0
            y[j] -= 1
        return x, y


def get_model(img_size, num_classes):
    inputs = keras.Input(shape=img_size + (3,))

    # This method creates the UNET based model

    # Input block
    x = layers.Conv2D(32, 3, strides=2, padding="same")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    pre_block_activation = x  # Set aside residual

    # Blocks 1, 2, 3 with different feature depth.
    for filters in [64, 128, 256]:
        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(filters, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(filters, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.MaxPooling2D(3, strides=2, padding="same")(x)

        res_layer = layers.Conv2D(filters, 1, strides=2, padding="same")(
            pre_block_activation
        )
        x = layers.add([x, res_layer]) 
        pre_block_activation = x  

    ### [Second half of the network: upsampling inputs] ###

    for filters in [256, 128, 64, 32]:
        x = layers.Activation("relu")(x)
        x = layers.Conv2DTranspose(filters, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.Activation("relu")(x)
        x = layers.Conv2DTranspose(filters, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.UpSampling2D(2)(x)

        # Project residual
        res_layer = layers.UpSampling2D(2)(pre_block_activation)
        res_layer = layers.Conv2D(filters, 1, padding="same")(res_layer)
        x = layers.add([x, res_layer])  # Add back residual
        pre_block_activation = x  # Set aside next residual

    # Pixel classification layer
    outputs = layers.Conv2D(num_classes, 3, activation="softmax", padding="same")(x)

    # Creates the keras model
    model = keras.Model(inputs, outputs)

    # Returns the model
    return model

def train(args):
    logging.basicConfig()
    if args.debug:
        print(args)
    input_dir = args.images_dir
    target_dir = args.labels_dir

    input_img_paths = sorted(
        [
            os.path.join(input_dir, fname)
            for fname in os.listdir(input_dir)
            if fname.endswith(".jpg")
        ]
    )
    target_img_paths = sorted(
        [
            os.path.join(target_dir, fname)
            for fname in os.listdir(target_dir)
            if fname.endswith(".png") and not fname.startswith(".")
        ]
    )

    # Free up RAM in case the model definition cells were run multiple times
    keras.backend.clear_session()

    # Build model
    if args.debug:
        logging.debug("Creating model ...")
    model = get_model(img_size, num_classes)
    if args.debug:
        model.summary()

    """
    ## Set aside a validation split
    """

    randomValue = random.randint(0, 10000)

    # Split our img paths into a training and a validation set
    val_samples = int(len(input_img_paths) * args.val_size)
    random.Random(randomValue).shuffle(input_img_paths)
    random.Random(randomValue).shuffle(target_img_paths)
    train_input_img_paths = input_img_paths[:-val_samples]
    train_target_img_paths = target_img_paths[:-val_samples]
    val_input_img_paths = input_img_paths[-val_samples:]
    val_target_img_paths = target_img_paths[-val_samples:]

    if args.debug:
        logging.debug("Creating train/val dataloaders ...")
    # Instantiate data Sequences for each split
    train_gen = SegClass(
        args.batch_size, img_size, train_input_img_paths, train_target_img_paths
    )
    val_gen = SegClass(args.batch_size, img_size, val_input_img_paths, val_target_img_paths)

    """
    ## Train the model
    """

    # Configure the model for training.
    # We use the "sparse" version of categorical_crossentropy
    # because our target data is integers.
    if args.debug:
        logging.debug("Compiling model ...")

    if args.solver == 'Adagrad':
        opt = keras.optimizers.Adagrad(learning_rate=args.learning_rate)
    elif args.solver == 'Adadelta':
        opt = keras.optimizers.Adadelta(learning_rate=args.learning_rate)
    elif args.solver == 'SGD':
        opt = keras.optimizers.SGD(learning_rate=args.learning_rate)
    elif args.solver == 'RMSprop':
        opt = keras.optimizers.RMSprop(learning_rate=args.learning_rate)
    else:
        opt = keras.optimizers.Adam(learning_rate=args.learning_rate)

    model.compile(optimizer=opt, loss="sparse_categorical_crossentropy")

    callbacks = [
        keras.callbacks.ModelCheckpoint( args.model_file + ".h5", save_best_only=True)
    ]

    if args.debug:
        logging.debug("Training model ...")
    # Train the model, doing validation at the end of each epoch.
    model.fit(train_gen, epochs=args.epochs, validation_data=val_gen, callbacks=callbacks)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--images_dir', required=True, help='Train directory.')
    parser.add_argument('--labels_dir', required=True,
                        help='Validation directory.')
    parser.add_argument('--solver', required=False, choices=['Adam', 'Adadelta', 'SGD', 'RMSprop', 'Adagrad'], default='Adam',
                        help='Number of epochs to train (Default: 50)')
    parser.add_argument('--epochs', required=False, type=int, default=50,
                        help='Number of epochs to train (Default: 50)')
    parser.add_argument('--val_size', required=False, type=float, default=0.2,
                        help='Number of epochs to train (Default: 0.2)')
    parser.add_argument('--learning_rate', required=False, type=float, default=0.0001,
                        help='Learning rate parameter for Adam optimizer (Default: 0.0001)')
    parser.add_argument('--batch_size', required=False, type=int, default=32,
                        help='Batch size (Default: 32)')
    parser.add_argument('--convert_to_onnx', default=False, action='store_true',
                        help='Convert the trained model to ONNX format instead of KERAS format')
    parser.add_argument('--model_file', required=False, type=str, default="classification_model",
                        help='File name to save the trained model')
    parser.add_argument('--debug', required=False, action='store_true',
                        help='Active logging for debug mode')
    parser.add_argument('--load_weights', required=False, type=str, default="",
                        help='File containing previous weights to load for training again from this point')
    args = parser.parse_args()
    train(args)
