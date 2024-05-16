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
    #heights = np.array([1,8,10,20,21,33,37,41,53])
    geo = SplineGeometry()
    pnts = [
     (0   ,-38),
     (133 ,  0),
     (189 ,-16),
     (189 ,-24),
     (195 ,-26),
     (210 ,-53),
     (181 ,-69),
     ( 86 ,-37),
     (129 ,-26),
     (191 ,-37),
     (197 ,-49),
     (184 ,-57)]
    
def corner(pointA,pointB):
    '''
    Return the coordinates of the "corner" point used for creating a spline3 curve approximating a
    quarter-circle.
    '''
    a1,a2,b1,b2 = pointA[0],pointA[1],pointB[0],pointB[1]
    if b2 > a2:
        c1, c1 = 0.5 * (a1 + b1 + b2 - a2), 0.5 * (a2 + b2 + a1 - b1)
    else:
        c1, c2 = 0.5 * (a1 + b1 + a2 - b2), 0.5 * (a2 + b2 + b1 - a1)
    return tuple((c1, c2))


def shift_coords(data, x_shift, y_shift):
    '''
    Prints coordinates of points together with their shifted counterparts.
    '''
    data = np.concatenate([data+np.array([x_shift, y_shift])], axis=1)
    print(data)


if __name__ == "__main__":
    
    '''
    data = np.array([
        [123,-20],
        [179,1],
        [253,16],
        [309,-1],
        [309,-8],
        [318,-10],
        [333,-37],
        [304,-53],
        [209,-21],
        [252,-10],
        [314,-21],
        [320,-33],
        [307,-41]
        ])
    shift_coords(data, -123, -16)
    '''
    A, B = (195 ,-26), (210 ,-53)
    C,D = (210 ,-53),(181 ,-69)

    print(corner(C,D))
