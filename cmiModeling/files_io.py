# read in subduction zone data
import meshio
import numpy as np
import pandas as pd

def readMesh(filename) :
    mesh = dict()
    meshobj = meshio.read(filename)
    mesh["file_name"] = filename
    mesh["points"] = meshobj.points
    mesh["verts"] = meshio.CellBlock("triangle", meshobj.get_cells_type("triangle")).data

    keep_el = np.ones(len(mesh["verts"])).astype(bool)

    # test for flattened elements

    for i in range(len(mesh["verts"])):
        tri_test = np.shape(np.unique(mesh["points"][mesh["verts"][i,:],:],axis=0))[0]
        if tri_test != 3:
            keep_el[i] = False

    mesh["verts"] = mesh["verts"][keep_el,:]

    return mesh


# read in gps data
def readGPS(filename) :
    colnames = ["station_ID", 'lon', 'lat', 'east_vel', 'north_vel', 'up_vel']
    gps = pd.read_table("./cumulative_disp.txt", sep='\s+', header=None, names=colnames)
    return gps


# parse a filename from the test parameters
def getFilename(config) :
    fileName = './_outputs/' + 'D' + str(config["planeDepth"]) + '_S'
    if(config["spatiallyVariable"]) :
        fileName = fileName + 'V_Testing/'
    else:
        fileName = fileName + 'U_Testing/'

    fileName = fileName + str(config["results"]["testName"])

    return fileName