# Computer Vision Mission

### Using Python and Keras

Requirements: Anaconda

``` sh
$ conda install --yes --file requirements.txt
# or to install all dependencies
$ while read requirements; do conda install --yes $requirement; done < requirements.txt
```

### Using convolutional networks to do image classification

The use of convolutional is better for images, convolutional works too well with
multiple dimensions data, like images.

### How to use:

To start simply create your own environment on Anaconda
after that install keras and load in order the scripts:

``` python-console
$ python -i scrapper.py
>>> airliner_scrapper(URL_AIRLINER, COMPANY_NAME))
```

This script will start downloading multiple files in data/train,
so do you need to copy some files to validation and test.

This script gets data from https://airliners.net/ and you need a search
url, not posts.

Example: https://www.airliners.net/search

On source file you already have some example links. With variable names:

    URL_TAM | URL_UNITED_AIRLINERS | URL_DELTA | URL_UNTITLED

DON'T PUT SAME COMPANY IN VALIDATION AND TEST, THIS ARE MEANT TO DIFFERENT COMPANIES OR RANDOM IMAGES

Now we need to start the convolutional network script.

``` sh
$ python -i convnet.py
>>> train()
>>> # Wait to finish training
>>> model = load_trained_model("model_saved.h5")
```

And to test new images

``` python-console
>>> test_image(image_path)
```

You will receive the class number that it was predicted by the model trained.
