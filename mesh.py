import PIL as pil
import numpy as np
#import nanomesh as nm
from netgen.geom2d import SplineGeometry
from firedrake import *



# BEGINNING OF NANOMESH FUNCTIONS
#################################################

# def generate_bw(file_name):
#     '''
#     Generate a black and white image and save it.
#     '''
#     image = pil.Image.open(file_name)
#     image = image.convert('L') 
#     image = image.point(lambda i: i < 175 and 255)
#     image.save("bw-valve-figure.png")

# def generate_array(file_name, save=False, display_np=False):
#     '''
#     Generate a numpy array from an image.
#     options:
#         save - saves the numpy array as a binary file
#         display_np - saves the visual representation of the numpy array as a image
#     '''
#     np_image = np.asarray(pil.Image.open(file_name))[:,:,0:3]
#     np_image = np.max(np_image, axis=2)
#     #print(np_image.shape)
#     if save:
#         np.save("np_image", np_image)
#     if display_np:
#         pil.Image.fromarray(np_image).save('visual_np_image.png')

# def nanomesh(file_name):
#     '''
#     Generation of the mesh from a numpy array.
#     See the following website for details:
#         https://nanomesh.readthedocs.io/en/latest/examples/examples_generate_a_2d_triangular_mesh.html
#     '''
#     data = nm.Image.load(file_name)
#     mesher = nm.Mesher2D(data)
#     mesher.generate_contour(max_edge_dist=5)
#     mesh = mesher.triangulate(opts='q20a10')
#     #triangle_mesh = mesh.get('triangle')
#     mesh.write('out.msh', file_format='gmsh22', binary=False)

# END OF NANOMESH FUNCTIONS
#################################################


def generate_points(lobes):
    '''
    Return an 2D array of points defining the Tesla valve. The first lobe is facing downward.
    Variables:
        lobes - number of lobes in the Tesla valve, an even number (eg. 2, 4, 6, ...)
    '''
    coords = np.zeros((lobes, 12, 2))
    x_shift, y_shift = 133, 38
    ps = np.array([
     [0   ,-38],
     [133 ,  0],
     [189 ,-16],
     [189 ,-24],
     [195 ,-26],
     [210 ,-53],
     [181 ,-69],
     [ 86 ,-37],
     [129 ,-26],
     [191 ,-37],
     [197 ,-49],
     [184 ,-57]])
    for l in range(lobes):
        qs = np.copy(ps)
        qs[:,1] *= (-1)**l  # mirror the y coordinates for odd lobes
        qs[:,0] += x_shift*l    # shift by the length of one lobe l times
        qs[:,1] += y_shift*(l%2)    # shift odd lobes up
        coords[l, :, :] = qs
    return coords
        
# ps = [
#     (0   ,-38),
#     (133 ,  0),
#     (189 ,-16),
#     (189 ,-24),
#     (195 ,-26),
#     (210 ,-53),
#     (181 ,-69)]
# qs = [
#     ( 86 ,-37),
#     (129 ,-26),
#     (191 ,-37),
#     (197 ,-49),
#     (184 ,-57)]

def netgen_mesh(lobes, elem_size):
    geo = SplineGeometry()
    coords = generate_points(lobes)

    # add the beginning of the valve (inlet)
    bs = [(-45, 0), (-45, -38)]
    b0, b1 = [geo.AppendPoint(*b) for b in bs]
    ps = [coords[0,0], coords[0,1]]
    p0, p1 = [geo.AppendPoint(*p) for p in ps]
    beginning_curves = [
        [["line", p1, b0], "line"],
        [["line", b0, b1], "line"],
        [["line", b1, p0], "line"]
    ]
    [geo.Append(c, bc=bc) for c, bc in beginning_curves]

    # add all of the lobes of the valve
    p0, p1, p2, p3, p4, p5, p6 = [geo.AppendPoint(*p) for p in ps]
    p45 = geo.AppendPoint(*corner(ps[4], ps[5])) 
    p56 = geo.AppendPoint(*corner(ps[5], ps[6]))
    q0, q1, q2, q3, q4 = [geo.AppendPoint(*q) for q in qs]
    q23 = geo.AppendPoint(*corner(qs[2], qs[3]))
    q34 = geo.AppendPoint(*corner(qs[3], qs[4]))
    outer_curves = [
        [["line", p0, p6], "line"],
        [["spline3", p6, p56, p5], "curve"],
        [["spline3", p5, p45, p4], "curve"],
        [["line", p4, p3], "line"],
        [["line", p3, p2], "line"],
        [["line", p2, p1], "line"],    
        [["line", p1, p0], "line"] 
    ]
    inner_curves = [
        [["line", q0, q1], "line"],
        [["line", q1, q2], "line"],
        [["spline3", q2, q23, q3], "curve"],
        [["spline3", q3, q34, q4], "curve"],
        [["line", q4, q0], "line"] 
    ]
    [geo.Append(c, bc=bc) for c, bc in outer_curves]
    [geo.Append(c, bc=bc) for c, bc in inner_curves]

    # add the end of the valve
    # ...

    ngmsh = geo.GenerateMesh(maxh=elem_size)
    msh = Mesh(ngmsh)
    VTKFile("output/MeshExample2.pvd").write(msh)
    
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


# def shift_coords(data, x_shift, y_shift):
#     '''
#     Auxiliary function.
#     Prints coordinates of points together with their shifted counterparts.
#     '''
#     data = np.concatenate([data+np.array([x_shift, y_shift])], axis=1)
#     print(data)


if __name__ == "__main__":
    # netgen_mesh()
    ps = generate_points(4)
    print(ps)