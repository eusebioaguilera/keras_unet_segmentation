import argparse
from train_keras_unet import get_model
from PIL import Image, ImageOps
from tensorflow.keras.preprocessing.image import load_img
import numpy as np
from config import img_size, num_classes

def predict(args):
    # Get the UNET model
    model = get_model(img_size=img_size, num_classes=num_classes)
    # Loads the weights for the trained model
    model.load_weights(args.model)
    # Load the image for the segmentation
    img = load_img(args.image, target_size=img_size)

    # Convert the image for the deep learning model input format
    x = np.zeros((1,) + img_size + (3,), dtype="float32")
    x[0] = img

    # Get the prediction for that image
    val_preds = model.predict(x)[0]

    # Get the max probability class label for each pixel
    mat_pred = np.argmax(val_preds, axis=2)

    # Convert to int8 data type 
    mat_pred = np.int8(mat_pred)
    # Create the segmentation image and save it
    pred_img = Image.fromarray(mat_pred).convert("L")
    pred_img = ImageOps.autocontrast(pred_img)
    pred_img.save("./pred.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--model', required=True,
                        help='model file')
    parser.add_argument('--image', required=True,
                        help='image file')
    args = parser.parse_args()

    predict(args)