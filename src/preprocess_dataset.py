import argparse
import glob
import hashlib
import os
import random
import time
from os.path import join

import Augmentor
from PIL import Image, ImageOps
from tqdm.std import tqdm
import numpy as np


BACKGROUND = 1
SEGMENTED_CLASS = 2

def get_name(file):
    hash = hashlib.sha256(file.encode("utf-8")).hexdigest()
    ext = os.path.splitext(file)[1][1:]

    return hash + "." + ext

def convert_labelled_images(path):
    """
    This code converts binary labelled (Black is the blakground/White is the class you want to segment) images in images ready for training:
        * Grayscales images
        * Each pixel will be the value of the label starting from 1 (Background), 2 (Class 1), ...
    @param path Path of the labelled images
    """
    files = glob.glob(path + "/*")
    with tqdm(total=len(files), desc="Executing Pipeline", unit=" Samples") as progress_bar:
        for file in files:
            if file.endswith(".png") or file.endswith(".jpg"):
                im = Image.open(file)
                grayscale = ImageOps.grayscale(im)
                mat = np.asarray(grayscale)
                values = np.unique(mat)
                changes = {}
                # Hacemos que se pueda escribir en la matriz
                mat.setflags(write=1)
                mat[np.where(mat < 200)] = 0
                mat[np.where(mat >= 200)] = 1
                # Pasamos a imagen
                grayscale2 = Image.fromarray(mat)
                grayscale2 = ImageOps.grayscale(grayscale2)
                new_file = file.replace(".jpg", ".png")
                grayscale2.save(new_file)
                progress_bar.set_description("Converting %s" % os.path.basename(file))
                progress_bar.update(1)


def get_relabelled_image(img):
    mat = np.array(img)
    # Make mat writable
    # This line of code may fail if an older version numpy is used
    mat.setflags(write=1)
    # Two labels
    mat[np.where(mat < 200)] = BACKGROUND
    mat[np.where(mat >= 200)] = SEGMENTED_CLASS

    # Return pillow type image
    return Image.fromarray(mat)

def execute_augment_process(augmentor, augmentor_image):
    """
    This method is used to generated augmented synthetic data for your dataset
    """

    images = []

    if augmentor_image.image_path is not None:
        images.append(Image.open(augmentor_image.image_path))

    if augmentor_image.pil_images is not None:
        images.append(augmentor_image.pil_images)

    if augmentor_image.ground_truth is not None:
        if isinstance(augmentor_image.ground_truth, list):
            for image in augmentor_image.ground_truth:
                images.append(Image.open(image))
        else:
            images.append(Image.open(augmentor_image.ground_truth))

    for operation in augmentor.operations:
        r = round(random.uniform(0, 1), 1)
        if r <= operation.probability:
            images = operation.perform_operation(images)

    return images[0], images

def preprocess(args):
    # First of all convert the labelled images
    random.seed(time.time())
    output_original = join(args.output_dir, "imgs")
    output_label = join(args.output_dir, "label")
    os.makedirs(output_original, exist_ok=True)
    os.makedirs(output_label, exist_ok=True)

    # Create the augmentor pipeline 
    p = Augmentor.Pipeline(args.source_dir)
    p.ground_truth(args.label_dir) #Comprobar los nombres de los ficheros (nombre + extensión)
    p.random_distortion(1.0, 5, 5, 2)
    augmentor_images = [random.choice(p.augmentor_images) for _ in range(args.samples)]
    with tqdm(total=len(augmentor_images), desc="Executing Pipeline", unit=" Samples") as progress_bar:
        for augmentor_image in augmentor_images:
            # Obtain the augmented image
            img, images = execute_augment_process(p, augmentor_image)
            new_file_name = get_name(augmentor_image.image_file_name)
            new_original = join(args.output_dir, "imgs", new_file_name)
            new_label = join(args.output_dir, "label", new_file_name.replace(".jpg", ".png"))
            images[0].save(new_original)
            # Convert to grayscale image
            bw_img = ImageOps.grayscale(images[1])
            bw_img = get_relabelled_image(bw_img)
            bw_img.save(new_label)
            progress_bar.set_description("Processing %s" % os.path.basename(augmentor_image.image_path))
            progress_bar.update(1)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--source_dir', required=True,
                        help='Directory containing the image files')
    parser.add_argument('--label_dir', required=True,
                        help='Directory containing the labelled image files')
    parser.add_argument('--output_dir', required=True, default="output",
                        help='Directory containing the output image files')
    parser.add_argument('--samples', required=False, type=int, default=1000,
                        help='Number of samples to generate. Default 1000.')
    args = parser.parse_args()

    preprocess(args)