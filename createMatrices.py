import numpy as np
import celeri
from dataclasses import dataclass


# create a displacement matrix and a smoothing matrix
def createDispSmoothMats(gps, numTri, beginIndices, endIndices, meshes) :

    # Quick config class
    @dataclass
    class Config:
        material_lambda = 30000000000
        material_mu = 30000000000
    config = Config()

    # Allocate space for slip-to-displacement array
    disp_mat = np.zeros((3*len(gps.lon), 3*numTri))
    # Allocate space for slip-to-displacement array
    smoothing_mat = np.zeros((3*numTri, 3*numTri))

    # For each mesh, fill in disp_mat with the values and not just zeros
    for mesh_idx in range(len(meshes)):
        # Calculate slip to displacement partials, using geographic coordinates
        disp_mat[:, 3*beginIndices[mesh_idx]:3*endIndices[mesh_idx]] = celeri.spatial.get_tde_to_velocities_single_mesh(meshes, gps, config, mesh_idx)
        # Get smoothing operator

        # Indices of shared sides in this mesh
        share = celeri.spatial.get_shared_sides(meshes[mesh_idx].verts)
        # Distances between centroids of shared elements
        tri_shared_sides_distances = celeri.spatial.get_tri_shared_sides_distances(
                share,
                meshes[mesh_idx].x_centroid,
                meshes[mesh_idx].y_centroid,
                meshes[mesh_idx].z_centroid,
            )
        # Distance-scaled smoothing matrix
        smat = celeri.spatial.get_tri_smoothing_matrix(
                share, tri_shared_sides_distances
            )
        # Insert sparse matrix into full array
        smoothing_mat[3*beginIndices[mesh_idx]:3*endIndices[mesh_idx], 3*beginIndices[mesh_idx]:3*endIndices[mesh_idx]] = smat.toarray()
        # Get smoothing operator

    # get rid of tensile slip columns in fault part of disp_mat and  rows/columns of smoothing matrix (needs to be square)
    from celeri import celeri_util
    keep = celeri_util.get_keep_index_12(3*endIndices[0])

    # using the indicies of the elements to be kept, find the elements that need to be deleted
    throw = []
    for i in range(3*beginIndices[1]):
        if np.isin(keep, i).any():
            pass
        else:
            throw.append(i)
    throw=np.array(throw)

    # take out the tensile columns of the subduction zone matrix, leave the CMI matrix as is
    # take out tensile rows/columns of smoothing mat, leaving it a square matrix

    disp_mat = np.delete(disp_mat, throw, axis=1)
    smoothing_mat = np.delete(smoothing_mat, throw, axis=0)
    smoothing_mat = np.delete(smoothing_mat, throw, axis=1)

    return disp_mat, smoothing_mat


# pass in the cmi mesh, and new 
# dictionary entries will be made
# for elements touching the far latitude
# and minimum longitude edges
def findEdgeElem(cmi) :
    # find elements on the minimum longitude and maximum latitude edge of the CMI

    min_lon = np.min(cmi["points"][:,0])
    max_lat = np.max(cmi["points"][:,1])
    min_lat = np.min(cmi["points"][:,1])

    # find the rows of points that touch the minimum longitude value
    lon_indicies = np.nonzero(cmi["points"][:,0]==min_lon)
    # find the vertices / elements that correspond to those minimum longitude points
    far_west = np.isin(cmi["verts"], lon_indicies)

    # fill in the dictionary section of "far_west" for the CMI, true/false an element is at the min lon
    col = np.full((len(far_west[0:,]),2), False)
    for n in range(len(far_west[0:,])):
        if far_west[n,:].any():
            col[n,0] = True
    cmi["far_west"] = col[:,0]

    # find the rows of points that touch the maximum latitude values
    lat_indicies = np.nonzero(cmi["points"][:,1] == max_lat)
    # find the verts/elem that correspond to those points
    far_north = np.isin(cmi["verts"], lat_indicies)

    # fill in dictionary section of "far_north" for the CMI
    for p in range(len(far_north[0:,])):
        if far_north[p,:].any():
            col[p,1] = True
    cmi["far_north"] = col[:,1]

    # find the rows of points that touch the minimum latitude values
    min_lat_ind = np.nonzero(cmi["points"][:,1] == min_lat)
    far_south = np.isin(cmi["verts"], min_lat_ind)

    # fill in dictionary section of "far_south" for CMI
    for q in range(len(far_south[0:,])):
        if far_south[q,:].any():
            col[q,1] = True
    cmi["far_south"] = col[:,1]

    return cmi



# create a constraint matrix for different elements in the fault and cmi meshes.
# putting the row/col value for a mesh element = one means when it's multiplied 
# by the estimated slip, the result is zero in the data vector, so the trivial answer is that slip must be zero 
def constraintMatrix(index, constraintMatrix, containsTensile, shift=0) :

    # if the constraint matrix contains tensile slip, that means it is the cmi
    # and the cmi elements need to be shifted down columns by the fault elements
    if containsTensile :
        for i in range(len(index)):
            constraintMatrix[3*i, shift + 3*index[i]] = 1
            constraintMatrix[3*i +1, shift + 3*index[i]+1] = 1
            constraintMatrix[3*i +2, shift + 3*index[i]+2] = 1

    else :    
        for i in range(len(index)):
            constraintMatrix[2*i, shift+ 2*index[i]] = 1
            constraintMatrix[2*i +1, shift+ 2*index[i] +1] = 1
        
    return constraintMatrix