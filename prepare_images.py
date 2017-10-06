"""
Code to generate the SiSi dataset

"""
import os
import PIL
import PIL.Image
import PIL.ImageOps
import numpy as np

__author__ = "Ronny Restrepo"
__copyright__ = "Copyright 2017, Ronny Restrepo"
__credits__ = ["Ronny Restrepo"]
__license__ = "Apache License"
__version__ = "2.0"

# SETTINGS
id2label = ["null", "bird", "cat", "dog"]
label2id = {label:id for id,label in enumerate(id2label)}
n_classes = len(id2label)


# ==============================================================================
#                                                      CREATE_TEMPLATE_FROM_FILE
# ==============================================================================
def create_template_from_file(file, shape=(100,100)):
    """ Given a filepath to a silhouette image, it returns a numpy array with
        1 representing the region of the object of interesst, and 0 as the
        background.
    """
    img = PIL.Image.open(file)

    # CONVERT TO GRESYSCALE
    # NOTE: conversion to greyscale NEEDS to be done BEFORE resizing with RGBA
    # images, otherwise it does unexpected things.
    if img.mode in ["RGBA", "LA"]:
        # Handle images where background is transparent
        # - Paste the image into a white canvas
        img = PIL.Image.open(file)
        img = img.convert("RGBA")
        canvas = PIL.Image.new('RGBA', img.size, (255,255,255,255))
        canvas.paste(img, mask=img)
        img = canvas.convert("L")
    else:
        img = img.convert('L')  # Convert to greyscale

    img = img.resize(shape, resample=PIL.Image.BICUBIC)
    img = PIL.ImageOps.invert(img) # make white the inside of the object
    return np.array(np.asarray(img) > 125, dtype=np.uint8)


