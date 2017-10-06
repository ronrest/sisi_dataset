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


def random_scene(templates, img_shape=(100,100), min_scale=0.5, rotate=30, noise=10, rgb=False):
    height, width = img_shape
    label = np.zeros(img_shape, dtype=np.uint8)

    # Randonly chose the order in which the class objects will be added to scene
    # - to simulate depth occlusion.
    order = np.random.permutation(n_classes)

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #  CREATE SCENE LABEL
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # For each class of object, randomly add it (or exclude it) from the scene
    for id in order:
        # Skip the background class
        if id == 0:
            continue

        # Only add this class of object to scene 50% of time
        if np.random.choice([True, False]):
            choice_id = np.random.choice(len(templates[id]))
            template = templates[id][choice_id]

            # --------------------------------------
            # TRANSFORMATIONS ON TEMPLATE - position, flip, rotation
            # --------------------------------------
            template = PIL.Image.fromarray(template)

            # Random flip
            if np.random.choice([True,False]):
                template = template.transpose(method=PIL.Image.FLIP_LEFT_RIGHT)

            # Random rotation of template
            angle = np.random.randint(-rotate, rotate)
            template = template.rotate(angle, expand=True, resample=PIL.Image.NEAREST)

            # Randomly scale the template
            rescale = (np.array(template.size)*np.random.uniform(low=min_scale, high=1.0)).astype(dtype=np.uint8)
            template = template.resize(rescale, resample=PIL.Image.NEAREST)

            # Randomly position in the scene
            pos_x = np.random.randint(max(1,width-template.width+1))
            pos_y = np.random.randint(max(1,height-template.height+1))
            base = PIL.Image.fromarray(np.zeros(img_shape, dtype=np.uint8))
            base.paste(template, box=[pos_x, pos_y])
            template = np.asarray(base, dtype=np.uint8)
            # --------------------------------------

            # Assign class labels to pixels associated with this object
            label[template>0] = id

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # CREATE SCENE IMAGE FROM LABEL
    # Given the class labels for pixels, add
    # unique color/texture to those regions
    # for each class of object.
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # BASE COLORS FOR OBJECTS/BG - Randomly shuffled for variety
    if rgb:
        base_colors = np.array(
            [(94, 61, 8),
            (77, 63, 43),
            (116, 102, 82),
            (154, 136, 111),
            (209, 194, 167),
            (204, 164, 108),
            (85, 87, 83),
            (136, 138, 133),
            ])
    else:
        base_colors = np.array([0, 50, 87, 135, 185, 210, 255])
    np.random.shuffle(base_colors)

    # Scene blank canvas
    n_channels = 3 if rgb else 1
    scene = np.zeros((height,width,n_channels), dtype=np.uint8)

    uids = np.unique(label) # class ids
    for uid in uids:
        # ADD COLOR - to the pixels associated with this class of object
        pixels = label==uid
        base = np.ones([height, width, n_channels])*base_colors[uid]

        # ADD RANDOM NOISE - for texture
        if noise:
            base += np.random.normal(loc=0, scale=noise, size=[height, width, n_channels])
            base = np.clip(base, 0, 255)
            base = base.astype(np.uint8)

        # Update the scene
        scene[pixels] = base[pixels]

    return scene, label


