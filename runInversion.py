import numpy as np


# assemble the weights matrix, inputing the smoothing weights for the fault, cmi, and the weights for the 
# constraint matrices
def assembleWeights(numMeshes, assembledMat, dispArray, smoothingWeights, allElemBegin, allElemEnd) :
    # Start with unit uncertainties 
    # this puts the smoothing weight for the fault mesh and cmi mesh, and leaves gps stations as 1
    weights = np.ones((np.shape(assembledMat)[0], 1)) # might want to update when adding slip constraint

    for mesh_idx in range(numMeshes+2):
        weights[np.size(dispArray)+ allElemBegin[mesh_idx]:np.size(dispArray)+allElemEnd[mesh_idx]] = smoothingWeights[mesh_idx]

    return weights

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