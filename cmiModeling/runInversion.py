import numpy as np
import pandas as pd


# assemble the weights matrix, inputing the smoothing weights for the fault, cmi, and the weights for the 
# constraint matrices
def assembleWeights(numMeshes, assembledMat, dispArray, smoothingWeights, allElemBegin, allElemEnd) :
    # Start with unit uncertainties 
    # this puts the smoothing weight for the fault mesh and cmi mesh, and leaves gps stations as 1
    weights = np.ones((np.shape(assembledMat)[0], 1)) # might want to update when adding slip constraint

    for mesh_idx in range(numMeshes+2):
        weights[np.size(dispArray) + allElemBegin[mesh_idx]:np.size(dispArray)+allElemEnd[mesh_idx]] = smoothingWeights[mesh_idx]

    return weights

# remove the contribution of predicted fault displacements from the observed gps displacements
def removeFaultContribution(gps, estSlip, dispMat, allElemBegin):
    # calculate displacements by which components

    # observed displacements only east and north because fault has no tensile slip (?)
    observedEast = np.array(gps.east_vel)
    observedNorth = np.array(gps.north_vel)
    observedUp = np.array(gps.up_vel)

    # calc disp from fault
    # multiply disp mat by the estimated slip on the fault
    fault_disp = dispMat[:, allElemBegin[0]:allElemBegin[1]].dot(estSlip[allElemBegin[0]:allElemBegin[1]])
    fault_e = fault_disp[0::3].flatten()
    fault_n = fault_disp[1::3].flatten()

    newEast = observedEast - fault_e
    newNorth = observedNorth - fault_n
    newUp = observedUp

    # same column headers as old gps file
    newGps = pd.DataFrame({"station_ID": np.array(gps.station_ID),
                           "lon": np.array(gps.lon),
                           "lat": np.array(gps.lat),
                           "east_vel": newEast,
                           "north_vel": newNorth,
                           "up_vel": newUp})

    return newGps

def runInversion(assembledMat, dispMat, weights, dataVector) :
    # Calculate model covariance
    cov = np.linalg.inv(assembledMat.T * weights.T @ assembledMat) 

    # Estimate slip using pre-calculated covariance
    estSlip = cov @ assembledMat.T * weights.T @ dataVector 
    # Predict displacement at stations
    predDisp = dispMat.dot(estSlip) 
    # run to check sign convention of dip slip (neg = east pos = west on CMI)
    # pred_disp = disp_mat[:, 1+all_elem_beg[1]::3].dot(est_slip[1+all_elem_beg[1]::3])

    return estSlip, predDisp