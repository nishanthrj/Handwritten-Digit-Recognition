import os
import string
import random
from pathlib import Path
import numpy as np
from PIL import Image, ImageOps

os.environ["KERAS_BACKEND"] = "torch"
import keras


def random_name_generator(n):
    return "".join(random.choices(string.ascii_uppercase + string.digits, k=n))


def save_image(image, img_name):
    if not os.path.exists("./static/data/"):
        os.mkdir(os.path.join("./static/", "data"))
    image.save(Path(f"./static/data/{img_name}"))


def prep_input(img):
    img = ImageOps.grayscale(img)
    img = ImageOps.invert(img)
    img = img.resize((28, 28))
    img_arr = np.array(img)
    img_arr = img_arr / 255.0
    img_arr = img_arr.reshape(1, 28, 28, 1)
    return img_arr

def prep_output(results):
    results = list(map(lambda x: round(x * 100, 2), results[0]))
    best = max(results)
    others = results
    values = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    pred = list(zip(values, others))
    best = pred.pop(results.index(best))
    return best, pred

def recognize(image):
    img = Image.open(image).convert("L")
    img_name = random_name_generator(10) + ".jpg"
    save_image(img, img_name)
    input_image = prep_input(img)
    model = keras.models.load_model("./model/model.h5")
    results = model.predict(input_image)
    best, pred = prep_output(results)
    return best, pred, img_name
