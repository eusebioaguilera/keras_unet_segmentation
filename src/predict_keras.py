import argparse
from train_keras_unet import get_model, Roots
from PIL import Image, ImageOps
from tensorflow.keras.preprocessing.image import load_img
import numpy as np
from config import img_size, num_classes

def predict(args):
    model = get_model(img_size=img_size, num_classes=num_classes)
    #opt = keras.optimizers.Adam(learning_rate=args.learning_rate)
    #model.compile(optimizer=opt, loss="sparse_categorical_crossentropy")
    model.load_weights(args.model)
    img = load_img(args.image, target_size=img_size)
    #output = model.predict([img])

    x = np.zeros((1,) + img_size + (3,), dtype="float32")
    x[0] = img

    # Obtenemos la prediccion para la imagen pasada como argumento
    val_preds = model.predict(x)[0]
    # Obtenemos la clase con mayor probabilidad para cada pixel (0 --> fondo, 1 --> raiz)
    mat_pred = np.argmax(val_preds, axis=2)
    print(np.unique(mat_pred))
    # Se binariza en formato de imagen
    #mat_pred = ImageOps.autocontrast(mat_pred)
    # Convertimos a int8 (0-255)
    mat_pred = np.int8(mat_pred)
    # Creamos un objeto imagen y guardamos
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