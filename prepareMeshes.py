import numpy as np
import pandas as pd
import gmsh # Creation of fault models
from scipy.interpolate import NearestNDInterpolator
import celeri
import meshio # Interaction between fault model files and Python

# Define some helpful functions for determining points along triangle legs

# There are 6 possible of tri verts [A,B,C] where at least one is above and at least one is below the plane (spanning depth contour).
# (where T=below and F=above the plane)
# [TTF], [TFT], [TFF], [FTT], [FTF], [FFT]
# After determining the combo (which legs are crossing the plane), the points along the depth contour are calculated on each leg respectively.
# For three triangle vertices: [A,B,C]
# Where each vertex consists of [*lon*, *lat*, *dep*]
# To find point P, along line segment **AB** at a depth of 60 km:
# Each element in point P can be found using an endpoint of the line segment, and adding a scalar quantity (t) of the line segment's direction vector.

# *dep*P = *dep***A** + (t * **AB**[2])
# But we know that the depth will always be 60 for the depth contour

# 60 = *dep***A** + (t * [*dep***B** - *dep***A**])
# t = (60 - *dep***A**) / (*dep***B** - *dep***A**)
# So because we can find a value for scalar t, we can determine the other values of elements in point P.

def direction_vector(p1, p2):
    dir_vec = [(p2[0] - p1[0]), (p2[1] - p1[1]), (p2[2] - p1[2])]
    return dir_vec


# a point P on a line segment will be P = t * direction_vector + end_point
# where t is a scalar amount, here, determined by the known plane_depth
# this is done for each element of point P, (x,y,z)
def point_grab(point1, point2, plane_depth): 
    dir_vector = direction_vector(point1, point2)
    t = scalar(point1, point2, plane_depth)

    point = [(t*dir_vector[0] + point1[0]), (t*dir_vector[1] + point1[1]), (t*dir_vector[2] + point1[2])]
    return point

# for each point, the desired depth value
# is the same as plane_depth
# so z is always known, and we can create the scalar quantity
# for the rest of the points based upon it
# plane_depth = z1 + t(z2 - z1)
# t = (plane_depth - z1) / (z2 - z1)
def scalar(p1, p2, depth):
    t = (depth - p1[2]) / (p2[2] - p1[2])
    return t

# cross_lat = y1 + t(y2 - y1)
# (cross_lat - y1)/(y2 - y1)
def scalar2(p1, p2, cross_lat):
    t = (cross_lat - p1[1]) / (p2[1] - p1[1])
    return t


# given a depth contour, or z value, 
# trace a contour of points along a given mesh and
# return that array of points as a numpy array
# parameter plane depth is the depth value 
# or the "clipping plane"
# the depth contour is returned,
# as well as the clipped fault mesh
def findContour(mesh, plane_depth) :


    ### SORT OUT TRIANGLES SPANNING THE DEPTH CONTOUR ###

    # triangle node values and empty list
    # each row is a triangle element [lon1, lat1, dep1, lon2, lat2, dep2, lon3, lat3, dep3]
    nodes = np.empty((mesh["lon1"].size, 9))
    nodes[:,0] = mesh["lon1"] ; nodes[:,1] = mesh["lat1"] ; nodes[:,2] = abs(mesh["dep1"]) # POSITIVE DEPTH CONVENTION HERE #
    nodes[:,3] = mesh["lon2"] ; nodes[:,4] = mesh["lat2"] ; nodes[:,5] = abs(mesh["dep2"])
    nodes[:,6] = mesh["lon3"] ; nodes[:,7] = mesh["lat3"] ; nodes[:,8] = abs(mesh["dep3"])

    # for each row (triangle element), check if the depths span the contour needed
    # if the depths span the contour, that means the triangle legs cross clipping plane depth
    # and a point on two legs can be included in the contour 

    bool_mask = np.zeros((np.size(nodes, 0)), dtype=bool)

    for i in range(np.size(nodes, 0)):
        node_depth = [nodes[i,2], nodes[i,5], nodes[i,8]] # access the depth values of a triangle element
        if (max(node_depth) > plane_depth > min(node_depth)):
            bool_mask[i] = True
        else:
            bool_mask[i] = False
    # so for each row (each triangle) if it crosses the clipping plane, that row is True, if it doesn't that row is False

    # construct an array where each row is a triangle (lon1, lat1, dep1, lon2, lat2, dep2, lon3, lat3, dep3)
    # which spans the depth contour, sorted by the boolean mask made earlier
    tri_elem = np.empty((bool_mask.sum(), 9)) #sum is adding up True=1 False=0
    tri_elem[::] = nodes[::][bool_mask] # grab triangle elements based on the depth spanning criteria


    ### DETERMINE WHICH NODES LIE ABOVE AND BELOW THE DEPTH CONTOUR ###

    # there are six combinations of point arrangements, but only 3 unique combinations of line segments ac, ab, bc

    # array to keep track of combinations of points above/below clipping plane
    node_combo = np.empty((len(tri_elem[:,0]), 3))
    num_extra = 0 # keep track of how many new triangles need to be created in the fault mesh

    # array for points along the depth contour (including duplicates), each element will have two points
    depth_all = np.empty(((2*np.size(tri_elem, 0)), 3)) 
    m = 0 # row number of the current point, initialized outside of loop

    # 1) determine which line segments are crossing plane, (ab, bc, or ac)
    # 2) determine point coordinates where the clipping plane crosses the line segment
    # 3) add the points to the array of all depth points
    for j in range(np.size(tri_elem, 0)):
        a = tri_elem[j,[0,1,2]] #node a
        b = tri_elem[j,[3,4,5]] #node b
        c = tri_elem[j,[6,7,8]] #node c

        # determine the combination of nodes above and below the CMI plane
        # T = below plane, F = above plane
        combo = [bool(a[2]>plane_depth), bool(b[2]>plane_depth), bool(c[2]>plane_depth)]
        node_combo[j,:] = combo # store result

        if sum(combo) == 1 : # if only one is below the plane, a new triangle needs to be added to the fault mesh
            num_extra += 1

        # ab, ac are the triangle legs when a is alone above or below the line
        if ((combo==[True, False, False]) or (combo==[False,True,True])):

            # each point along the depth contour can be found using an end point, plus a scalar quantity along a direction vector
            p1 = point_grab(a, b, plane_depth) 
            p2 = point_grab(a, c, plane_depth)

            depth_all[m,:] = p1
            m+=1 # increment the row number for each point added
            depth_all[m,:] = p2
            m+=1 # increment row

        # ab, bc when b is alone above or below the line
        elif ((combo==[True,False,True]) or (combo==[False,True,False])):

            p1 = point_grab(a, b, plane_depth)
            p2 = point_grab(b, c, plane_depth)

            depth_all[m,:] = p1
            m+=1 # increment the row number for each point added
            depth_all[m,:] = p2
            m+=1 # increment row

        # only remaining line segment combination is [ac, bc], where vertex C is isolated above/below
        else:

            #each of the three components of point1 are endpoint, plus a scalar quantity of the direction vector
            p1 = point_grab(a, c, plane_depth)
            p2 = point_grab(b, c, plane_depth) # the z component will always be 60 km or plane_depth, including the equation helps clarify other indicies

            depth_all[m,:] = p1
            m+=1 # increment the row number for each point added
            depth_all[m,:] = p2
            m+=1 # increment row

    # triangles share legs, so points are repeated unnecessarily, this takes only the unique rows in the 2D array
    # using pandas not numpy, because numpy will sort the columns, but that shuffles lon, lat pairs apart
    df = pd.DataFrame(depth_all, columns=["lon", "lat", "dep"])
    depth_contour = df.drop_duplicates(subset=["lon"]) 
    depth_contour = np.array(df.drop_duplicates(subset="lat")) # need to check both longitude and latitude for duplicates, otherwise mesh problems arise

    fault = clipFault(bool_mask, depth_all, tri_elem, node_combo, num_extra, plane_depth, mesh)

    return depth_contour, fault



def clipFault(bool_mask, depth_all, tri_elem, node_combo, num_extra, plane_depth, mesh) :

    # create clipped fault mesh based upon CMI meshing, MUST have run CMI depth contour cell without changing anything, the order of elements in arrays is important

    # there are 6 possibilities for mesh / plane intersection of the triangles
    # given 3 nodes [a,b,c], where T is below and F is above the clipping plane, 
    # [TFF], [TTF], [TFT], [FTF], [FFT], [FTT]
    # these possibilities are numbered 1 through 6
    # if 1: keep node a, and add points along ab and ac calculated from the depth contour
    # if 2: [a, bc, ac] [a, b, ac]
    # if 3: [a, bc, ab] [a, c, bc]
    # if 4: [b, ab, bc]
    # if 5: [c, ac, bc]
    # if 6: [b, ac, ab] [b, c, ac]

    df = pd.DataFrame(depth_all, columns=["lon", "lat", "dep"])
    depth_contour = df.drop_duplicates(subset=["lon"]) 
    depth_contour = np.array(df.drop_duplicates(subset="lat")) # need to check both longitude and latitude for duplicates, otherwise mesh problems arise

    num_old = len((mesh["points"][:,0]))
    new_points = np.empty(((num_old + len(depth_contour[:,0])), 3))

    # add the old points as well as the new points to a shared array, contains points deeper than needed for now
    new_points[0:num_old, [0,1,2]] = mesh["points"][:, [0,1,2]]
    new_points[num_old:, [0,1]] = depth_contour[:,[0,1]]
    new_points[num_old:, 2] = -1*depth_contour[:,2] # negative to match previous convention for points

    # create array of new triangle elements
    new_tri_elem = np.zeros(((len(tri_elem[:,0])+num_extra), 9)) # as many as previously plus the new ones
    point_count = 0
    tri_count = 0

    # node_combo is the combination of T below plane and F above plane for triangle nodes
    for i in range(len(node_combo[:,0])):
        temp_list = [node_combo[i,0], node_combo[i,1], node_combo[i,2]]

        if temp_list==[False,True,True]:
            new_tri_elem[tri_count, [0,1,2]] = tri_elem[i,[0,1,2]] # keep node a
            new_tri_elem[tri_count, [3,4,5]] = depth_all[point_count, [0,1,2]] ; point_count += 1 # add ab
            new_tri_elem[tri_count, [6,7,8]] = depth_all[point_count, [0,1,2]] ; point_count += 1 # add ac
            tri_count += 1
        elif temp_list==[False,False,True]:
            new_tri_elem[tri_count, [0,1,2]] = tri_elem[i,[0,1,2]] # keep node a
            new_tri_elem[tri_count, [3,4,5]] = depth_all[point_count, [0,1,2]] ; point_count += 1 # add ac
            new_tri_elem[tri_count, [6,7,8]] = depth_all[point_count, [0,1,2]] ; point_count += 1 # add bc
            tri_count += 1
            new_tri_elem[tri_count, [0,1,2]] = tri_elem[i, [0,1,2]] # keep node a
            new_tri_elem[tri_count, [3,4,5]] = tri_elem[i, [3,4,5]] # keep node b
            new_tri_elem[tri_count, [6,7,8]] = depth_all[point_count-1, [0,1,2]] # add node ac
            tri_count += 1
        elif temp_list==[False,True,False]:
            new_tri_elem[tri_count, [0,1,2]] = tri_elem[i, [0,1,2]] # keep node a
            new_tri_elem[tri_count, [3,4,5]] = depth_all[point_count, [0,1,2]] ; point_count += 1 # add node ab
            new_tri_elem[tri_count, [6,7,8]] = depth_all[point_count, [0,1,2]] ; point_count += 1 # add node bc
            tri_count += 1
            new_tri_elem[tri_count, [0,1,2]] = tri_elem[i, [0,1,2]] # keep node a
            new_tri_elem[tri_count, [3,4,5]] = tri_elem[i, [6,7,8]] # keep node c
            new_tri_elem[tri_count, [6,7,8]] = depth_all[point_count-1, [0,1,2]] # add node bc
            tri_count += 1
        elif temp_list==[True,False,True]:
            new_tri_elem[tri_count, [0,1,2]] = tri_elem[i, [3,4,5]] # keep node b
            new_tri_elem[tri_count, [3,4,5]] = depth_all[point_count, [0,1,2]] ; point_count += 1 # add node ab
            new_tri_elem[tri_count, [6,7,8]] = depth_all[point_count, [0,1,2]] ; point_count += 1 # add node bc
            tri_count += 1
        elif temp_list==[True,True,False]:
            new_tri_elem[tri_count, [0,1,2]] = tri_elem[i, [6,7,8]] # keep node c
            new_tri_elem[tri_count, [3,4,5]] = depth_all[point_count, [0,1,2]] ; point_count += 1 # add node ac
            new_tri_elem[tri_count, [6,7,8]] = depth_all[point_count, [0,1,2]] ; point_count += 1 # add node bc
            tri_count += 1
        elif temp_list==[True,False,False]:
            new_tri_elem[tri_count, [0,1,2]] = tri_elem[i, [3,4,5]] # keep node b
            new_tri_elem[tri_count, [3,4,5]] = depth_all[point_count, [0,1,2]] ; point_count += 1 # add node ab
            new_tri_elem[tri_count, [6,7,8]] = depth_all[point_count, [0,1,2]] ; point_count += 1 # add node ac
            tri_count += 1
            new_tri_elem[tri_count, [0,1,2]] = tri_elem[i, [3,4,5]] # keep node b
            new_tri_elem[tri_count, [3,4,5]] = tri_elem[i, [6,7,8]] # keep node c
            new_tri_elem[tri_count, [6,7,8]] = depth_all[point_count-1, [0,1,2]] # add node ac
            tri_count += 1

    #depth val of new_tri_elem needs to be negative to match the points
    new_tri_elem[:,[2,5,8]] = -1*new_tri_elem[:,[2,5,8]]
    new_verts = np.empty((len(new_tri_elem[:,0]), 3))

    # determine the indecies of points for triangle vertexes
    for m in range(len(new_tri_elem[:,0])):
        points = np.round(new_points, decimals=9)
        lon1 = round(new_tri_elem[m,0], 9) ; lat1 = round(new_tri_elem[m,1], 9) ; dep1 = round(new_tri_elem[m,2], 9)
        lon2 = round(new_tri_elem[m,3], 9) ; lat2 = round(new_tri_elem[m,4], 9) ; dep2 = round(new_tri_elem[m,5], 9)
        lon3 = round(new_tri_elem[m,6], 9) ; lat3 = round(new_tri_elem[m,7], 9) ; dep3 = round(new_tri_elem[m,8], 9)
        index1 = np.where((points[:,0]==lon1) & (points[:,1]==lat1) & (points[:,2]==dep1))[0]
        index2 = np.where((points[:,0]==lon2) & (points[:,1]==lat2) & (points[:,2]==dep2))[0]
        index3 = np.where((points[:,0]==lon3) & (points[:,1]==lat3) & (points[:,2]==dep3))[0]
        new_verts[m,0] = int(index1[0])
        new_verts[m,1] = int(index2[0])
        new_verts[m,2] = int(index3[0])


    # bool mask was defined for elem spanning depth contour, need opposite now
    flip_bool = ~bool_mask
    old_verts = mesh["verts"][flip_bool]

    # find points that are lower than the depth contour and drop them
    too_deep = np.where(new_points[:,2] < -plane_depth)[0] # rows of points that have values too deep
    big_mask = np.isin(old_verts, too_deep, invert=True)

    mask = []
    for n in range(len(big_mask[:,0])):
        if big_mask[n].all():
            mask.append(True)
        else:
            mask.append(False)
    mask = np.array(mask) # True = tri elem above the maximum depth

    new_verts = new_verts.astype(int) # switch from float so that the indicies are compatible
    total_verts = np.vstack((old_verts[::][mask], new_verts)).astype(int)

    fault = remeshFault(new_points, total_verts, mesh)

    return fault


def remeshFault(new_points, total_verts, mesh) :
    # remesh the clipped fault and interpolate depths based on original subduction zone mesh

    clipped = {}
    clipped["points"] = new_points
    clipped["verts"] = total_verts

    # re mesh the clipped fault mesh as a flat plane, then interpolate the depth values based on the old point cloud
    celeri.mesh._compute_mesh_edge_elements(clipped) 

    ordered_edges = clipped["ordered_edge_nodes"][:,0]
    # using ordered edge nodes, create a perimiter around the inside of the fault

    edge_pts = np.empty((len(ordered_edges),3))

    for i in range(len(ordered_edges)):
        edge_pts[i] = clipped["points"][ordered_edges[i]] # grab indicies mentioned by ordered edges

    # begin mesh

    clen = 10

    if gmsh.isInitialized() == 0:
        gmsh.initialize()
    gmsh.option.setNumber("General.Verbosity", 0)
    gmsh.clear()

    for i in range(len(edge_pts[:,0])):
        gmsh.model.geo.addPoint(edge_pts[i, 0], edge_pts[i, 1], 0, clen, tag=i) # all have depth zero

    for j in range(len(edge_pts[:,0])-1):
        gmsh.model.geo.addLine(j, j+1, j)
    gmsh.model.geo.addLine(j+1, 0, j+1)

    gmsh.model.geo.synchronize()

    gmsh.model.geo.addCurveLoop(list(range(j+2)), 1)

    gmsh.model.geo.addPlaneSurface([1], 1)

    # Finish writing geo attributes
    gmsh.model.geo.synchronize()

    gmsh.write('clipped_fault' + '.geo_unrolled')

    # Generate mesh
    gmsh.model.mesh.generate(2) #meshed in spherical because the depth being in km isn't as important when it's flat

    gmsh.write('clipped_fault' + '.msh')
    gmsh.finalize()  


    # Read and parse mesh
    fault = dict()
    faultobj = meshio.read("clipped_fault.msh") 
    fault["points"] = faultobj.points
    fault["verts"] = meshio.CellBlock("triangle", faultobj.get_cells_type("triangle")).data

    # interpolate depth values
    interpolated = NearestNDInterpolator((mesh["points"][:,0], mesh["points"][:,1]), mesh["points"][:,2])
    new_depth_vals = interpolated(fault["points"][:,0], fault["points"][:,1])

    fault["points"][:,2] = new_depth_vals

    return fault