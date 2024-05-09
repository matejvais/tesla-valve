import PIL as pil
import numpy as np
import nanomesh as nm



def generate_bw(file_name):
    '''
    Generate a black and white image and save it into a file.
    '''
    image = pil.Image.open(file_name)
    image = image.convert('L') 
    image = image.point(lambda i: i < 175 and 255)
    image.save("bw-valve-figure.png")

def generate_array(file_name, save=False):
    '''
    Generate a numpy array from an image.
    '''
    np_image = np.asarray(pil.Image.open(file_name))[:,:,3]
    print(np_image.shape)
    if save:
        np.save("np_image", np_image)

def mesh(file_name):
    '''
    Generation of the mesh from a numpy array.
    See the following website for details:
        https://nanomesh.readthedocs.io/en/latest/examples/examples_generate_a_2d_triangular_mesh.html
    '''
    data = nm.Image.load(file_name)
    mesher = nm.Mesher2D(data)
    mesher.generate_contour(max_edge_dist=3)
    mesh = mesher.triangulate(opts='q30a100')
    #triangle_mesh = mesh.get('triangle')
    mesh.write('out.msh', file_format='gmsh22', binary=False)


if __name__ == "__main__":
    #generate_bw("valve-figure.jpg")
    generate_array("bw-adjusted-valve-figure.png",save=True)
