# Import libraries
import numpy as np # Numerical analysis
from prepareMeshes import findContour, meshCmi, expandMesh
from createMatrices import findEdgeElem, createDispSmoothMats, createIndexingLists, constrain
import celeri
from results import slipDist, displacements, residualPlot, numericalData, saveConfig, plotLikeDiao, afterslip
import yaml, argparse
from files_io import readMesh, readGPS, getFilename
from runInversion import runInversion, assembleWeights, removeFaultContribution

### SET MODEL CHOICES ###

def parseArgs() :
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", metavar='', default="config.yaml", help="name of config file")
    parser.add_argument("--planeDepth", type=int, metavar='', help="Override plane depth")
    parser.add_argument("--faultSmoothing", type=float, metavar='', help="Override fault smoothing value")
    parser.add_argument("--cmiSmoothing", type=float, metavar='', help="Override cmi smoothing value")
    parser.add_argument("--spatiallyVariable", action='store_true', help="Enable spatially variable smoothing feature")
    parser.add_argument("--saveFigures", action='store_true', help="Enable figure saving")
    parser.add_argument("--saveData", action='store_true', help="Enable numerical data saving")
    parser.add_argument("--testName", type=str, metavar='', help="Override test name")
    parser.add_argument("--gpsFile", type=str, metavar='', help="Override gps file")

    parser.add_argument("--oldResults", action='store_true', help='Process the results of a past test.')
    parser.add_argument("--resultFile", type=str, metavar='', help='Directory for the old test results.')

    parser.add_argument("--afterslipReinvert", action='store_true', help="Invert for cmi after removing fault afterslip displacements.")
    
    return parser.parse_args()

def main() :
    args = parseArgs()

    if (args.oldResults or args.afterslipReinvert) :
        with open(args.resultFile + '/configSettings.txt', 'r') as file:
            config = yaml.safe_load(file)
    else: 
        with open(args.config, "r") as file :
            config = yaml.safe_load(file)

    # override with command line arguments if given
    if (args.planeDepth) :
        config["planeDepth"] = args.planeDepth
    if (args.faultSmoothing) :
        config["smoothing"]["fault"] = args.faultSmoothing
    if (args.cmiSmoothing) :
        config["smoothing"]["cmi"] = args.cmiSmoothing
    if (args.spatiallyVariable) :
        config["spatiallyVariable"] = True
    if (args.saveFigures) :
        config["results"]["saveFigures"] = True
    if (args.saveData) :
        config["results"]["saveData"] = True
    if (args.testName) :
        config["results"]["testName"] = args.testName
    if (args.gpsFile) :
        config["inputs"]["gps"] = args.gpsFile

    print(config)

    # ###  READ IN SUBDUCTION ZONE MESH AND PARSE BEFORE USING IT TO CREATE A DEPTH CONTOUR ###

    mesh = readMesh(config["inputs"]["fault"]) # read in mesh file
    mesh = expandMesh(mesh)    # expand coordinates and calculate properties of mesh

    # Create depth contour points using triangle elements spanning the CMI depth
    # fault mesh automatically clipped to match
    depthContour, fault = findContour(mesh, config["planeDepth"])
    fault = expandMesh(fault) 

    # ### MESH THE CMI BASED ON THE LOWER DEPTH EXTENT OF AFTERSLIP AND THE DEPTH CONTOUR ###

    meshCmi(depthContour, config["cmi"]["minLon"], config["cmi"]["minLat"], config["planeDepth"], config["inputs"]["cmi"])
    horiz = readMesh(config["inputs"]["cmi"] + ".msh") # read in mesh
    horiz = expandMesh(horiz) # expand mesh coordinates


    # ### INVERSION CODE ###

    # # READ IN GPS DATA #
    gps = readGPS(config["inputs"]["gps"])

    # ### CONCATENATE MESHES AND CALCULATE PARTIAL DERIVATIVES ###

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

    elemBegin = [0, n_tri[0]]  # list of indexes, the beginning of the fault mesh elem, beginning of the cmi mesh elem
    elemEnd = [n_tri[0], n_tri[0]+n_tri[1]]

    # function automatically removes tensile rows and columns from the subduction zone matrix in disp mat
    # and it then removes the corresponding rows and columns of smoothing mat, leaving it as a square matrix

    # faultDispMat, faultSmoothingMat = createDispSmoothMats(gps=gps, numTri=len(fault["lon1"]), meshes=[meshes[0]], isFault=True)
    # horizDispMat, horizSmoothingMat = createDispSmoothMats(gps=gps, numTri=len(horiz["lon1"]), meshes=[meshes[1]])

    # # square matrix requires blocks of zeros to hold space
    # upperRightBlock = np.zeros((len(faultSmoothingMat[0]), len(horizSmoothingMat[0])))
    # lowerLeftBlock = np.zeros((len(horizSmoothingMat[0]), len(faultSmoothingMat[0])))

    # faultRow = np.hstack((faultSmoothingMat, upperRightBlock))
    # cmiRow = np.hstack((lowerLeftBlock, horizSmoothingMat))

    # # stick together column wise
    # dispMat = np.hstack((faultDispMat, horizDispMat))
    # smoothingMat = np.vstack((faultRow, cmiRow))
    dispMat, smoothingMat = createDispSmoothMats(gps, np.sum(n_tri), elemBegin, elemEnd, meshes)

    # *function should built-in test for flattened elements hopefully

    # find elements on the minimum longitude and maximum latitude edge of the CMI to constraint slip on
    horiz = findEdgeElem(horiz)

    # create constraint matrices

    # find edge elem to constrain slip, the meshes are too large
    # they're currently set up to constrain the dip slip
    celeri.mesh._compute_mesh_edge_elements(fault) 

    # finds the locations of top and side elements in the fault mesh
    # add ones in the constraint matrices for where slip is to be minimized/not allowed
    faultConstraint = constrain(fault, "top_elements", "side_elements", dispMat, False)

    # CMI element indicies need to be shifted by the number of fault elements
    shift = 2*len(fault["lon1"]) # two not three because tensile col has already been removed from fault elements
    horizConstraint = constrain(horiz, "far_west", "far_north", dispMat, True, shift)

    # create a spatially variable smoothing weight based upon the resolution of the triangles
    closeness = np.abs(dispMat.sum(axis=0))# sum is column wise
    # then its the sum of all the stations for each triangle ss, ds, and ts (if even) 

    # ### ASSEMBLE MATRICES, CONSTRAINTS, WEIGHTING VECTOR, AND DATA VECTOR ###

    assembledMat = np.vstack([dispMat, smoothingMat, faultConstraint, horizConstraint]) # stick constraint array as 3rd argument

    # # update indexing lists to accommodate the differences between 2*fault_elem and 3*cmi elem after removing tensile slip
    allElemBegin, allElemEnd = createIndexingLists(fault, horiz, faultConstraint, horizConstraint)

    # set smoothing and constraint weights
    if config["spatiallyVariable"]:
        faultSmoothing = (config["smoothing"]["fault"] * np.reciprocal(closeness[0:allElemBegin[1]])).reshape(-1, 1) # into column vector so it fits in weights
        cmiSmoothing = (config["smoothing"]["cmi"] * np.reciprocal(closeness[allElemBegin[1]:])).reshape(-1,1)
    else: 
        faultSmoothing = config["smoothing"]["fault"]
        cmiSmoothing = config["smoothing"]["cmi"]

    smoothingWeights = [faultSmoothing, cmiSmoothing, config["constraint"]["faultEdge"], config["constraint"]["cmiEdge"]] 
    if len(smoothingWeights) != (len(meshes)+2): # 2 meshes, and 2 constraint arrays need to all be weighted
        smoothingWeights = smoothingWeights*np.ones(len(meshes)) 

    # Assemble weighting vector
    # Allocate space for data vector
    dataVector = np.zeros((np.shape(assembledMat)[0], 1)) # by default, the rows corresponding to constraint array are initialized as zeros
    # Vector of displacements
    dispArray = np.array([gps.east_vel, gps.north_vel, gps.up_vel]).reshape((3,-1)).T.copy()
    dataVector[0:np.size(dispArray)] = dispArray.flatten().reshape(-1,1)

    weights = assembleWeights(2, assembledMat, dispArray, smoothingWeights, allElemBegin, allElemEnd)

    if (args.oldResults or args.afterslipReinvert) :
        fileName = getFilename(config)
        
        print(fileName)
        with open(fileName + '/estSlip.npy', 'rb') as f:
            estSlip = np.load(f)
        with open(fileName + '/predDisp.npy', 'rb') as f:
            predDisp = np.load(f)
        
        print(np.shape(predDisp))

    else :
        # ### PERFORM INVERSION ###
        estSlip, predDisp = runInversion(assembledMat, dispMat, weights, dataVector)

        with open('estSlip.npy', 'wb') as f:
            np.save(f, estSlip)
        with open('predDisp.npy', 'wb') as f:
            np.save(f, predDisp)

    if (args.afterslipReinvert) :
        newGps = removeFaultContribution(gps, estSlip, dispMat, allElemBegin)
        # create new disp mat with only the cmi, ignoring the fault

        # function wants elemBegin and meshes as lists, so the single data values are in brackets
        newDispMat, newSmoothingMat = createDispSmoothMats(newGps, np.sum(n_tri), [elemBegin[1]], [elemEnd[1]], [meshes[1]])
        # 0 shift because no fault elements
        horizConstraint = constrain(horiz, "far_west", "far_north", newDispMat, True, 0)
        newAssembledMat = np.vstack([newDispMat, newSmoothingMat, horizConstraint]) # stick constraint array as 3rd argument

        newSmoothingWeights = [cmiSmoothing, config["constraint"]["cmiEdge"]] 

        # Assemble weighting vector
        # Allocate space for data vector
        newDataVector = np.zeros((np.shape(newAssembledMat)[0], 1)) # by default, the rows corresponding to constraint array are initialized as zeros
        # Vector of displacements
        newDispArray = np.array([newGps.east_vel, newGps.north_vel, newGps.up_vel]).reshape((3,-1)).T.copy()
        newDataVector[0:np.size(newDispArray)] = newDispArray.flatten().reshape(-1,1)

        newWeights = assembleWeights(0, newAssembledMat, newDispArray, newSmoothingWeights, [0,allElemBegin[1]], [allElemBegin[1], allElemBegin[1]+len(horizConstraint[0:,])])

        # re run the inversion but now only for cmi 
        determinant = np.linalg.det(newAssembledMat.T * newWeights.T @ newAssembledMat)
        if np.isclose(determinant, 0):
            print("Matrix is singular or nearly singular.")
        else:
            estSlip, predDisp = runInversion(newAssembledMat, newDispMat, newWeights, newDataVector)
    

    # # VISUALIZE RESULTS
    length = 0.2 # 5mm
    scale = 0.2 # relative to cm, so 1cm vector scale = 1. 5mm vector scale = 0.5 etc.

    # slipDist(estSlip, gps, fault, horiz, vecScale, config["results"]["saveFigures"], config["results"]["slipDist"])
    # # calls plotRatio
    # displacements(dispMat, allElemBegin, estSlip, predDisp, gps, vecScale, config["results"]["saveFigures"], 
    #             config["results"]["allDisp"], config["results"]["dispSep"], config["results"]["ratioFig"])
    # afterslip(estSlip=estSlip, fault=fault)
    
    plotLikeDiao(gps, predDisp, length, scale, dispMat, estSlip, allElemBegin, 58, config["results"]["saveFigures"], False)
    # residualPlot(gps, predDisp, vecScale, config["results"]["saveFigures"], config["results"]["residFig"])

    # # numerical data
    # numericalData(estSlip, predDisp, gps, allElemBegin, fault, horiz, config["results"]["saveData"])

    # # save config settings, just in case they're forgotten later and images are referenced
    if (config["results"]["saveFigures"] or config["results"]["saveData"]):
        saveConfig(config)


if __name__ == "__main__":
    main()