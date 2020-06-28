# Overview
This  program converts a rough sketch into an illustration of a landscape picture. I have implemented the Pix2Pix algorithm published in the “Image to Image translation with conditional Adversarial Nets” written by AI Research Laboratory, UC Berkeley, 

## pix2pix_landscape_train.ipynb
This program will train the model and preprocesses the images. To get the inputs for training, the edges will be extracted from the sample images to imitate the sketch that should be inputed. Then a copy of the original image will be blurred and random parts eliminated, this image will act as the color guide for the program. The network will then take in these two images as inputs, and try to make its output as close as possible to the original image. 

The sample images come from https://www.kaggle.com/arnaud58/landscape-pictures

*Please note: Since the trained models are too big, they have not been included

## Create.py
This application uses the trained model. It takes a sketch of a landscape and a rough color guide as an input, and returns the generated image.

The trained model was too big to be included so pix2pix_landscape_train.ipynb will have to be run first in order for a trained version to be obtained
