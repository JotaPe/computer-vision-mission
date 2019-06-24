from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
import os
import glob
from tqdm import tqdm
import fire

DATAGEN = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest',
        rescale=1. / 255)

pbar = tqdm(total=len(os.listdir('data/train')))


def datagen(folder_path, size_of_gen):
    """Data generator to mitigate overfitting!"""
    for image in glob.glob(os.path.join(f'{folder_path}', '*.jpeg')):
        img = load_img(image)
        x = img_to_array(img)
        x = x.reshape((1,) + x.shape)
        i = 0
        pbar.update(1)
        for batch in DATAGEN.flow(x,
                                  batch_size=1,
                                  save_to_dir=f"{folder_path}",
                                  save_format="jpeg", save_prefix=f"plane"):
            i += 1
            if i > size_of_gen:
                break
        pbar.close()


if __name__ == "__main__":
    fire.Fire(datagen)
