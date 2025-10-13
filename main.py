# Import libraries
import pandas as pd
import numpy as np # Numerical analysis
import gmsh # Creation of fault models
import meshio # Interaction between fault model files and Python
from expandMesh import expandMesh
from prepareMeshes import findContour
from createMatrices import findEdgeElem, constraintMatrix, createDispSmoothMats
import celeri
from results import slipDist

### SET MODEL CHOICES ###

# set depth of CMI
cmi_depth = int(input("Enter the cmi depth: " ))

# set depth of clipping plane for subduction zone
# if coldNose = true, plane_depth!=cmi_depth
plane_depth = int(input("Enter the clipping plane depth (Likely same as cmi depth): "))

# minimum longitude and latitude for the cmi
min_lon = 125 # (originally 137) (wide = 130)
min_lat = 34

# set spatially variable smoothing values to true or false (false=uniform smoothing values)
spatiallyVar = input("Slip spatially variable (True/False): ")
if spatiallyVar == "True":
    spatiallyVar = True
else :
    spatiallyVar = False

# set smoothing weight for fault and cmi
fault_smooth = float(input("Enter fault smoothing weight: "))
cmi_smooth = float(input("Enter cmi smoothing weight: ")) # the smoothing value e.g., 1e14

# save figures automatically to a file, must update test/file name otherwise past testing overwritten
# file will be created and written to a folder named cmiModeling on desktop
saveFigures = False
saveData = False # saves numerical output (e.g., max fault slip mag) as a text file in the outputs directory for the test
testName = input("Enter test name: ")

# FIGURES #
# choose which figures to save

slip_dist = True # slip distributions plotted on cmi and on fault
disp_sep = True  # map view gps displacements separated into contributions from fault and cmi
all_disp = False # predicted and total displacements map view
cross_fig = False # cross section figure
ratio_fig = True # ratio of cmi contribution to total displacement
resid_fig = True # residual plotting




###  READ IN SUBDUCTION ZONE MESH AND PARSE BEFORE USING IT TO CREATE A DEPTH CONTOUR ###

# Read in source fault

filename = "japan.msh"
mesh = dict()
meshobj = meshio.read(filename)
mesh["file_name"] = filename
mesh["points"] = meshobj.points
mesh["verts"] = meshio.CellBlock("triangle", meshobj.get_cells_type("triangle")).data
ntri = len(mesh["verts"])

mesh = expandMesh(mesh) # expand coordinates and calculate properties of mesh


# Create depth contour points using triangle elements spanning the CMI depth
# fault mesh automatically clipped to match

depth_contour, fault = findContour(mesh, plane_depth)
fault = expandMesh(fault) 


### MESH THE CMI BASED ON THE LOWER DEPTH EXTENT OF AFTERSLIP AND THE DEPTH CONTOUR ###

#sort points by increasing latitude
indicies = np.argsort( depth_contour[:,1] )
depth_contour = depth_contour[indicies]

# separately add on the corners of the CMI
# min lon and min lat defined at top
max_lat = np.max(depth_contour[:,1]+2.5) # maximum latitude for corner

# corner points starting from lower left and moving counterclockwise
corner_points = np.array([[min_lon, min_lat, plane_depth], [depth_contour[0,0], min_lat, plane_depth], [min_lon, max_lat, plane_depth], [np.max(depth_contour[:,0]), max_lat, plane_depth]])

# total points along the perimiter of the CMI mesh
mesh_edge = np.concatenate((depth_contour, corner_points))

cx = mesh_edge[:,0]
cy = mesh_edge[:,1]
cz = -1*mesh_edge[:,2] # depth is negative

## BEGIN GMSH

char_len = 0.75 # smaller is good for degrees
n_points = np.shape(depth_contour)[0] # number of depth contour points
num_surf = np.shape(depth_contour)[0]
num_lines = np.shape(mesh_edge)[0] #num lines is the same as the total number of points

if gmsh.isInitialized() == 0:
    gmsh.initialize()
gmsh.option.setNumber("General.Verbosity", 0)
gmsh.clear()

# Define points
gmsh.model.geo.addPoint(cx[-4], cy[-4], cz[-4], char_len, 0) #lower left corner because corner points were added last in the mesh_points
gmsh.model.geo.addPoint(cx[-3], cy[-3], cz[-3], char_len, 1)
for j in range(int(n_points)): # depth contour points
    gmsh.model.geo.addPoint(cx[j], cy[j], cz[j], char_len, j+2) 
gmsh.model.geo.addPoint(cx[-1], cy[-1], cz[-1], char_len, j+3) #upper right corner
gmsh.model.geo.addPoint(cx[-2], cy[-2], cz[-2], char_len, j+4) #upper left corner

# add lines between the points to complete the perimiter
for i in range(int(num_lines-1)):
    gmsh.model.geo.addLine(i, i+1, i)
gmsh.model.geo.addLine(i+1, 0, i+1) #complete the loop

gmsh.model.geo.synchronize()

# define curve loop counterclockwise
gmsh.model.geo.addCurveLoop(list(range(0, i+2)), 1)
gmsh.model.geo.addPlaneSurface([1], 1)

# Finish writing geo attributes
gmsh.model.geo.synchronize()

gmsh.write('horiz' + '.geo_unrolled')

# Generate mesh
gmsh.model.mesh.generate(2) #meshed in spherical because the depth being in km isn't as important when it's flat

gmsh.write('horiz' + '.msh')
gmsh.finalize() 


### EXPAND THE CMI MESH ###
horiz = dict()

# information about the CMI mesh is read in, and stored in a dictionary just like for fault mesh earlier
horizobj = meshio.read("horiz.msh") 
horiz["points"] = horizobj.points
horiz["verts"] = meshio.CellBlock("triangle", horizobj.get_cells_type("triangle")).data

keep_el = np.ones(len(horiz["verts"])).astype(bool)

for i in range(len(horiz["verts"])):
    tri_test = np.shape(np.unique(horiz["points"][horiz["verts"][i,:],:],axis=0))[0]
    if tri_test != 3:
        keep_el[i] = False

horiz["verts"] = horiz["verts"][keep_el,:]

horiz = expandMesh(horiz) # expand mesh coordinates


### INVERSION CODE ###

# READ IN GPS DATA #

colnames = ["station_ID", 'lon', 'lat', 'east_vel', 'north_vel', 'up_vel']
gps = pd.read_table("./cumulative_disp.txt", sep='\s+', header=None, names=colnames)

# Place stations into single array
# lat, lon, dep, but dep is always zero because they're on the ground
# reshape(3, -1) means put it into 3 rows and as many columns as data points
# then T makes it 3 columns, and as many rows as data points
obsv = np.array([gps.lon, gps.lat, 0*gps.lat]).reshape((3, -1)).T.copy()


### CONCATENATE MESHES AND CALCULATE PARTIAL DERIVATIVES ###

# Force meshes into dataclass, using existing fields
class Mesh:
    def __init__(self, d=None):
        if d is not None:
            for key, value in d.items():
                setattr(self, key, value)

# List of classes
meshes = [Mesh(fault), Mesh(horiz)]


# Define indices for meshes in arrays, where each meshes triangle elements begin
n_tri = np.zeros(len(meshes), dtype=int)
for i in range(len(meshes)):
    n_tri[i] = len(meshes[i].lon1)
tri_end_idx = np.cumsum(n_tri) # the last triangle index, the sum of all triangle elem in both meshes
tri_beg_idx = [0, tri_end_idx[0]] # list of indexes, the beginning of the fault mesh elem, beginning of the cmi mesh elem
total_n_tri = tri_end_idx[-1] 
# gets the last element of the summation that makes up total triangles, here, the number of tri elem in the CMI

# function automatically removes tensile rows and columns from the subduction zone matrix in disp mat
# and it then removes teh corresponding rows and columns of smoothing mat, leaving it as a square matrix
disp_mat, smoothing_mat = createDispSmoothMats(gps, np.sum(n_tri), tri_beg_idx, tri_end_idx, meshes)
    # *function should built-in test for flattened elements hopefully


# find elements on the minimum longitude and maximum latitude edge of the CMI to constraint slip on
horiz = findEdgeElem(horiz)


# create constraint matrices

# find edge elem to constrain slip, the meshes are too large
# they're currently set up to constrain the dip slip
celeri.mesh._compute_mesh_edge_elements(fault) 

# find elements above 20 km depth on fault for constraint matrix
too_high = np.where(fault["centroids"][:,2]>-20)[0]

# allocate space for constraint arrays 

# fault constraints
top_elem = sum(fault["top_elements"]) 
top_elem2 = len(too_high) - top_elem 
side_elem = sum(fault["side_elements"]) 

topConstraint = np.zeros(((2*(top_elem)), len(disp_mat[0])))       # top edge elements
topConstraint2 = np.zeros(((2*(top_elem2)), len(disp_mat[0])))     # top 20 km (not including top edge elements)
sideConstraint = np.zeros(((2*(side_elem)), len(disp_mat[0])))     # along the sides of the subduction zone

# find the set difference 
too_high = np.setdiff1d(top_elem, too_high) # so that the top edge elements aren't in both constraint arrays for upper elements

# locations of top and side elements in the fault mesh
idx = np.nonzero(fault["top_elements"])[0].reshape(-1,1)    # returns a row index for each nonzero (true) val
idx2 = np.nonzero(fault["side_elements"])[0].reshape(-1,1)

# add ones in the constraint matrices for where slip is to be minimized/not allowed

# top edge elements
topConstraint = constraintMatrix(idx, topConstraint, False)

# elements above 20 km depth
topConstraint2 = constraintMatrix(too_high, topConstraint2, False)

# elements along the sides of the fault
sideConstraint = constraintMatrix(idx2, sideConstraint, False)


# CMI constraint matrix containing elements on the top and left edge of the plane
edge_elem = sum(horiz["far_north"]) + sum(horiz["far_west"])
horizConstraint = np.zeros(((3*edge_elem), len(disp_mat[0])))

west_idx = np.nonzero(horiz["far_west"])[0].reshape(-1,1) # gives the row number of elements that touch the min lon value
north_idx = np.nonzero(horiz["far_north"])[0].reshape(-1,1)
horiz_rows = np.vstack((west_idx, north_idx)) # compiled rows of north/west elem

# CMI element indicies need to be shifted by the number of fault elements
shift = 2*len(fault["lon1"]) # two not three because tensile col has already been removed from fault elements

horizConstraint = constraintMatrix(horiz_rows, horizConstraint, True, shift=shift)

# modify indexing lists to account for the new elements in constraint arrays
# this is important for later when the weights are saved in an array

# shift for fault top elements 
tri_beg_idx.append(tri_end_idx[len(meshes)-1])
tri_end_idx = np.append(tri_end_idx, [tri_end_idx[len(meshes)-1]+len(topConstraint[0:,])], axis=0)

# shift for upper fault elements 
tri_beg_idx.append(tri_end_idx[len(meshes)])
tri_end_idx = np.append(tri_end_idx, [tri_end_idx[len(meshes)]+len(topConstraint2[0:,])], axis=0)

# shift for fault side elements
tri_beg_idx.append(tri_end_idx[len(meshes)+1])
tri_end_idx = np.append(tri_end_idx, [tri_end_idx[len(meshes)+1]+len(sideConstraint[0:,])], axis=0)

# shift for cmi edge elements
tri_beg_idx.append(tri_end_idx[len(meshes)+2])
tri_end_idx = np.append(tri_end_idx, [tri_end_idx[len(meshes)+2]+len(horizConstraint[0:,])], axis=0)


# create a spatially variable smoothing weight based upon the resolution of the triangles

### option 1 ###
closeness = np.abs(disp_mat.sum(axis=0))# sum is column wise

# then its the sum of all the stations for each triangle ss, ds, and ts (if even) 



### ASSEMBLE MATRICES, CONSTRAINTS, WEIGHTING VECTOR, AND DATA VECTOR ###

# Assemble matrices 

# assembled_mat = np.vstack([disp_mat, smoothing_mat, fault_constraint, horiz_constraint]) # stick constraint array as 3rd argument
assembled_mat = np.vstack([disp_mat, smoothing_mat, topConstraint, topConstraint2, sideConstraint, horizConstraint]) # stick constraint array as 3rd argument


# create new indexing lists to accommodate the differences between 2*fault_elem and 3*cmi elem after removing tensile slip
fault_end = tri_end_idx[0]*2
horiz_end = fault_end + len(horiz["lon1"])*3

top_constraint_end = horiz_end + len(topConstraint[0:,])
top_constraint2_end = top_constraint_end + len(topConstraint2[0:,])
side_constraint_end = top_constraint2_end + len(sideConstraint[0:,])

all_elem_beg = [0, fault_end, horiz_end, top_constraint_end, top_constraint2_end, side_constraint_end]
all_elem_end = [all_elem_beg[1], all_elem_beg[2], all_elem_beg[3], all_elem_beg[4], all_elem_beg[5], all_elem_beg[5]+len(horizConstraint[0:,])]


# set smoothing and constraint weights
# NOTE: smoothing weights set as model inputs at top

if spatiallyVar:
    fault_smoothing = (fault_smooth* np.reciprocal(closeness[0:all_elem_beg[1]])).reshape(-1, 1) # into column vector so it fits in weights
    cmi_smoothing = (fault_smooth* np.reciprocal(closeness[all_elem_beg[1]:])).reshape(-1,1)
else: 
    fault_smoothing = fault_smooth
    cmi_smoothing = cmi_smooth

top_weight = 1e4
upper_weight = 1
side_weight = 1
cmi_side_weight = 1e2

smoothing_weight = [fault_smoothing, cmi_smoothing, top_weight, upper_weight, side_weight, cmi_side_weight] 
if len(smoothing_weight) != (len(meshes)+4): # 2 meshes, and 4 constraint arrays need to all be weighted
    smoothing_weight = smoothing_weight*np.ones(len(meshes)) 

# Assemble weighting vector
# Allocate space for data vector
data_vector = np.zeros((np.shape(assembled_mat)[0], 1)) # by default, the rows corresponding to constraint array are initialized as zeros
# Vector of displacements
disp_array = np.array([gps.east_vel, gps.north_vel, gps.up_vel]).reshape((3,-1)).T.copy()
data_vector[0:np.size(disp_array)] = disp_array.flatten().reshape(-1,1)


# Start with unit uncertainties 
# this puts the smoothing weight for the fault mesh and cmi mesh, and leaves gps stations as 1
weights = np.ones((np.shape(assembled_mat)[0], 1)) # might want to update when adding slip constraint

for mesh_idx in range(len(meshes)+4):
    weights[np.size(disp_array)+ all_elem_beg[mesh_idx]:np.size(disp_array)+all_elem_end[mesh_idx]] = smoothing_weight[mesh_idx]


### PERFORM INVERSION ###

# Calculate model covariance
cov = np.linalg.inv(assembled_mat.T * weights.T @ assembled_mat) 

# Estimate slip using pre-calculated covariance
est_slip = cov @ assembled_mat.T * weights.T @ data_vector 
# Predict displacement at stations
pred_disp = disp_mat.dot(est_slip) 
# run to check sign convention of dip slip (neg = east pos = west on CMI)
# pred_disp = disp_mat[:, 1+all_elem_beg[1]::3].dot(est_slip[1+all_elem_beg[1]::3])



# VISUALIZE RESULTS

slipDist(est_slip, gps, fault, horiz)