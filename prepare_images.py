"""
Code to generate the SiSi dataset

"""
import os
import glob
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


# ==============================================================================
#                                                      TEMPLATES_FROM_RAW_IMAGES
# ==============================================================================
def templates_from_raw_images(data_dir, id2label, shape=(100,100)):
    """ Given a directory containing a separate subdirectories for each
        class of object, each containing image files of silhouettes
        of those objects, it creates a list of template images, such that:

        Each image is resized to `shape`, and the silhouette regions
        are converted to a value of 1, and background converted to value
        of 0.

        The templates list takes the following structure:
            [None, class1_templates, class2_templates, .... classN_templates]
        where each classI_templates is a numpy array of shape
            [n_images, shape[1], shape[0]]
        and of np.uint8 dtype.
    """
    n_classes = len(id2label)
    templates_list = [None]
    for label in id2label[1:]:
        # TEMPLATES FOR ONE CLASS
        dirpath = os.path.join(data_dir, label)

        # Get list of image files for this class
        img_files = []
        for ext in ['*.jpg', '*.png']:
            img_files.extend(glob.glob(os.path.join(dirpath, ext)))
        n_images = len(img_files)

        # For each image file convert to a template array
        templates = np.zeros([n_images, shape[1], shape[0]], dtype=np.uint8)
        for i, img_file in enumerate(img_files):
            templates[i] = create_template_from_file(img_file, shape=shape)

        # Add to templates dictionary
        templates_list.append(templates)

    return templates_list


