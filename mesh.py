import PIL as pil
import numpy as np
#import nanomesh as nm
from netgen.geom2d import SplineGeometry



def generate_bw(file_name):
    '''
    Generate a black and white image and save it.
    '''
    image = pil.Image.open(file_name)
    image = image.convert('L') 
    image = image.point(lambda i: i < 175 and 255)
    image.save("bw-valve-figure.png")

def generate_array(file_name, save=False, display_np=False):
    '''
    Generate a numpy array from an image.
    options:
        save - saves the numpy array as a binary file
        display_np - saves the visual representation of the numpy array as a image
    '''
    np_image = np.asarray(pil.Image.open(file_name))[:,:,0:3]
    np_image = np.max(np_image, axis=2)
    #print(np_image.shape)
    if save:
        np.save("np_image", np_image)
    if display_np:
        pil.Image.fromarray(np_image).save('visual_np_image.png')

def nanomesh(file_name):
    '''
    Generation of the mesh from a numpy array.
    See the following website for details:
        https://nanomesh.readthedocs.io/en/latest/examples/examples_generate_a_2d_triangular_mesh.html
    '''
    data = nm.Image.load(file_name)
    mesher = nm.Mesher2D(data)
    mesher.generate_contour(max_edge_dist=5)
    mesh = mesher.triangulate(opts='q20a10')
    #triangle_mesh = mesh.get('triangle')
    mesh.write('out.msh', file_format='gmsh22', binary=False)

def netgen_mesh():
    heights = np.array([1,8,10,20,21,33,37,41,53])
    geo = SplineGeometry()


if __name__ == "__main__":
    netgen_mesh()
