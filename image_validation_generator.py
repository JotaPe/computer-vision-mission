from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
import os, glob
from tqdm import tqdm

DATAGEN = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')

pbar = tqdm(total=len(os.listdir('data/validation')))

for image in glob.glob(os.path.join('data/validation/', '*.jpeg')):
    img = load_img(image)
    x = img_to_array(img)
    x = x.reshape((1,) + x.shape)
    i = 0
    pbar.update(1)
    for batch in DATAGEN.flow(x, batch_size=1, save_to_dir="data/train/",
                              save_format="jpeg", save_prefix=f"plane"):
        i += 1
        if i > 20:
            break
pbar.close()
