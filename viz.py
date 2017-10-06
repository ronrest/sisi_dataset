import PIL
import PIL.Image

__author__ = "Ronny Restrepo"
__copyright__ = "Copyright 2017, Ronny Restrepo"
__credits__ = ["Ronny Restrepo"]
__license__ = "Apache License"
__version__ = "2.0"


def show_template_image(x):
    PIL.Image.fromarray(x*255, mode="L").show()


