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
        qs[:,0] += x_shift*l    # shift by the length of one lobe l times in the x direction
        qs[:,1] += y_shift*(l%2)    # shift odd lobes up in the y direction
        coords[l, :, :] = qs
    return coords


def netgen_mesh(lobes, max_elem_size, debug=False):
    geo = SplineGeometry()
    coords = generate_points(lobes)
    p = np.empty((lobes, 16), dtype=int)

    # add the beginning of the valve (inlet)
    bs = [(-45, 0), (-45, -38)] # define points
    b0, b1 = [geo.AppendPoint(*b) for b in bs]  # append points to the geometry
    # ps = [coords[0,0], coords[0,1]]
    # p[0,0], p[0,1] = [geo.AppendPoint(*p) for p in ps]
    p[0,0] = geo.AppendPoint(*coords[0,0])
    beginning_curves = [    # define curves that constitute the initial part of the domain boundary
        # [["line", p[0,1], b0], "line"],
        [["line", b0, b1], "line"],
        [["line", b1, p[0,0]], "line"]
    ]
    [geo.Append(c, bc=bc) for c, bc in beginning_curves]    # append curves to the geometry

    # add all lobes except for the last one
    for l in range(lobes-1):
        ps = [coords[l][0], coords[l][1], coords[l][2], coords[l][3], coords[l][4], coords[l][5], coords[l][6]]
        if l != 0:
            p[l,0], p[l,1], p[l,2], p[l,3], p[l,4], p[l,5], p[l,6] = [geo.AppendPoint(*p) for p in ps]
        else:   # for the first lobe, the points p0, p1 have already been appended above
            p[l,1], p[l,2], p[l,3], p[l,4], p[l,5], p[l,6] = [geo.AppendPoint(*p) for p in ps[1:7]]
        p[l,12] = geo.AppendPoint(*corner(ps[4], ps[5]))    # p45
        p[l,13] = geo.AppendPoint(*corner(ps[5], ps[6]))    # p56
        qs = [coords[l][7], coords[l][8], coords[l][9], coords[l][10], coords[l][11]]
        p[l,7], p[l,8], p[l,9], p[l,10], p[l,11] = [geo.AppendPoint(*q) for q in qs]
        p[l,14] = geo.AppendPoint(*corner(qs[2], qs[3]))    # q23
        p[l,15] = geo.AppendPoint(*corner(qs[3], qs[4]))    # q34
        outer_curves = [    # curves forming the outer boundary of a lobe
            [["line", p[l, 0], p[l,6]], "line"], # p[l-1, 1] instead of p[l,0]
            [["spline3", p[l,6], p[l,13], p[l,5]], "curve"],
            [["spline3", p[l,5], p[l,12], p[l,4]], "curve"],
            [["line", p[l,4], p[l,3]], "line"],
            [["line", p[l,3], p[l,2]], "line"],
        ]
            # [["line", p2, p1], "line"],   # omitted
        if l != 0:
            outer_curves.append([["line", p[l-1,2], p[l,1]], "line"])   # connect p1 to p2 from the previous lobe
        else:
            outer_curves.append([["line", p[l,1], b0], "line"])
        inner_curves = [    # curves forming the inner boundary of a lobe
            [["line", p[l,7], p[l,8]], "line"],
            [["line", p[l,8], p[l,9]], "line"],
            [["spline3", p[l,9], p[l,14], p[l,10]], "curve"],
            [["spline3", p[l,10], p[l,15], p[l,11]], "curve"],
            [["line", p[l,11], p[l,7]], "line"] 
        ]

        if debug and l == lobes-2:
            outer_curves.append([["line", p[l,2], p[l,1]], "line"])
        if debug: print(l)
        [geo.Append(c, bc=bc) for c, bc in outer_curves]
        [geo.Append(c, bc=bc) for c, bc in inner_curves]
    
    # # add the last lobe
    # ps = [coords[lobes-1][0], coords[lobes-1][1], coords[lobes-1][5], coords[l][6]]
    # p0, p1, p5, p6 = [geo.AppendPoint(*p) for p in ps]
    # p56 = geo.AppendPoint(*corner(ps[2], ps[3]))
    # qs = [coords[lobes-1][7], coords[lobes-1][8], coords[lobes-1][9], coords[lobes-1][10], coords[lobes-1][11]]
    # q0, q1, q2, q3, q4 = [geo.AppendPoint(*q) for q in qs]
    # q23 = geo.AppendPoint(*corner(qs[2], qs[3]))
    # q34 = geo.AppendPoint(*corner(qs[3], qs[4]))
    # outer_curves = [    # curves forming the outer boundary of a lobe
    #     [["line", p0, p6], "line"],
    #     [["spline3", p6, p56, p5], "curve"],
    #     [["line", p1, p2_previous], "line"]
    # ]
    # inner_curves = [    # curves forming the inner boundary of a lobe
    #     [["line", q0, q1], "line"],
    #     [["line", q1, q2], "line"],
    #     [["spline3", q2, q23, q3], "curve"],
    #     [["spline3", q3, q34, q4], "curve"],
    #     [["line", q4, q0], "line"] 
    # ]
    # [geo.Append(c, bc=bc) for c, bc in outer_curves]
    # [geo.Append(c, bc=bc) for c, bc in inner_curves]
    
    # # add the end of the valve (outlet)
    # ref = coords[lobes-1][0]   # p1 point of the last lobe, used for reference coordinates
    # es = [ref+[96,-39], ref+[100,-34], ref+[100,-34], ref+[148,-34], ref+[148,4], ref+[103,4]]
    # e0, e1, e2, e3, e4, e5 = [geo.AppendPoint(*e) for e in es]
    # end_curves = [
    #         [["line", p5, e0], "line"],
    #         [["spline3", e0, e1, e2], "curve"],
    #         [["line", e2, e3], "line"],
    #         [["line", e3, e4], "line"],
    #         [["line", e4, e5], "line"],
    #         [["line", e5, p1], "line"]
    # ]
    # [geo.Append(c, bc=bc) for c, bc in end_curves]
    if debug: print("1\n")
    # construct the mesh
    ngmsh = geo.GenerateMesh(maxh=max_elem_size)
    if debug: print("2")
    msh = Mesh(ngmsh)
    VTKFile("output/MeshExample2.pvd").write(msh)
    
def corner(pointA,pointB):
    '''
    Return the coordinates of the "corner" point used for creating a spline3 curve approximating a
    quarter-circle.
    '''
    a1,a2,b1,b2 = pointA[0],pointA[1],pointB[0],pointB[1]
    if b2 > a2:
        c1, c2 = 0.5 * (a1 + b1 + b2 - a2), 0.5 * (a2 + b2 + a1 - b1)
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
    # ps = generate_points(3)
    # print(ps)
    netgen_mesh(lobes=2, max_elem_size=10, debug=True)