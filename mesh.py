import PIL as pil
import numpy as np
import nanomesh as nm


def generate_bw(file_name):
    '''
    Generate a black and white image and save it into a file.
    '''
    image = pil.Image.open(file_name)
    image = image.convert('L') 
    image = image.point(lambda i: i < 150 and 255)
    image.save("bw-valve-figure.png")

def generate_array(file_name, save=False):
    '''
    Generate a numpy array from an image.
    '''
    np_image = np.asarray(pil.Image.open(file_name))[3]
    print(np_image.shape)
    if save:
        np.save("np_image", np_image)

def mesh(file_name):
    data = np.asarray(pil.Image.open(file_name))   # turn the image into a numpy array
    return data

if __name__ == "__main__":
    #data = mesh("bw-adjusted-valve-figure.png")
    #data = mesh("valve-figure.jpg") 
    #print(data.shape)
    generate_array("bw-adjusted-valve-figure.png", save=True)
