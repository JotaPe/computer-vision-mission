# Computer Vision Mission

### Using Python and Keras

Requirements: Anaconda, Keras, urllib3, BeautifulSoup, Fire and Pillow

### Using convolutional networks to do image classification

The use of convolutional is better for images, convolutional works too well with
multiple dimensions data, like images.

### How to use:

To start simply create your own environment on Anaconda
after that install keras and load in order the scripts:

    $ python -i scrapper.py
    >>> airliner_scrapper(URL_AIRLINER, COMPANY_NAME))

This script will start downloading multiple files in data/train,
so do you need to copy some files to validation and test.

This script gets data from https://airliners.net/ and you need a search
url, not posts.

Example: https://www.airliners.net/search

On source file you already have some example links.

DON'T PUT SAME COMPANY IN VALIDATION AND TEST, THIS ARE MEANT TO DIFFERENT COMPANIES OR RANDOM IMAGES

Because this test have a small dataset, to evade overfitting we can use Keras
ImageDataGenerator from his preprocessing libraries

This is going to generate random edited images, is good to have a GPU because
this is going to use TensorFlow to do it and generate a lot of files, you just
need to run this script after the scrapper.

    $ python image_train_generator.py
    # After this load the validation generator too
    $ python image_validation_generator.py

I have done in this way to not need user input to use in different folders.

Now we need to start the convolutional network script.

    $ python -i convnet.py

And to test new images

    >>> test_image(image_path)

You will receive the class number that it was predicted by the model trained.
