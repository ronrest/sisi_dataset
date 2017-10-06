import PIL
import PIL.Image

__author__ = "Ronny Restrepo"
__copyright__ = "Copyright 2017, Ronny Restrepo"
__credits__ = ["Ronny Restrepo"]
__license__ = "Apache License"
__version__ = "2.0"


# ==============================================================================
#                                                            SHOW_TEMPLATE_IMAGE
# ==============================================================================
def show_template_image(x):
    """ Given a numpy array of 0's and 1's, representing a template/mask image
        with
            0 = background
            1 = region of interest
        it returns a PIL black and white image, with:
            white = region of interest
            black = background
    """
    PIL.Image.fromarray(x*255, mode="L").show()


