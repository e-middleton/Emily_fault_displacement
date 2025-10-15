import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import json


def slipDist(estSlip, gps, fault, cmi, vecScale, slipDist=False, saveFigures=False) :

    # plot for visualizing
    xmin = 122
    xmax = 148
    ymin = 32
    ymax = 48
    plt.close('all')

    coast = pd.read_csv("coastline.csv")
    lon_corr = 1
    # dip slip on the CMI represents east-west motion, where east is negative and west is positive
    slip_type = 1 # 0 = strike slip 1 = dip slip, only for CMI, 2 (manual) is vertical/tensile
    end_idx = 2* len(fault["lon1"]) #end of fault elem beginning of cmi elem
    slip_vals = [estSlip[slip_type:end_idx:2]/100, estSlip[slip_type+end_idx::3]/100] # dip slip values for fault and CMI, converted from cm to m

    max_mag_f = np.abs(np.max(slip_vals[0]))
    max_mag_h = np.abs(np.max(slip_vals[1]))
    if max_mag_f > max_mag_h:
        max_mag = max_mag_f
    else:
        max_mag = max_mag_h

    both = {}
    both["points"] = np.vstack((fault["points"], cmi["points"]))
    shift_val = len(fault["points"][:,0])
    both["verts"] = np.vstack((fault["verts"], cmi["verts"]+shift_val))

    fig, ax = plt.subplots(1, 2, figsize=(10,6))
    rso = ax[0].tripcolor(both["points"][:,0],
                        both["points"][:,1], 
                        both["verts"],
                        facecolors=(np.vstack(((slip_vals[0], slip_vals[1])))).flatten(), 
                        vmin=-max_mag, vmax=max_mag)
    #ax[0].tripcolor(horiz["points"][:,0], horiz["points"][:,1], horiz["verts"], facecolors=(slip_vals[1]).flatten())
    #ax[0].quiver(gps.lon, gps.lat, pred_disp[0::3], pred_disp[1::3], scale=vec_scale, color='r')
    cbar1 = fig.colorbar(rso, ax=ax[0], orientation='horizontal')
    ax[0].plot(coast.lon+360*(1-lon_corr), coast.lat, color="k", linewidth=0.5)
    cbar1.set_label("Slip (m)")
    ax[0].set(xlim=(xmin-2, xmax), ylim=(ymin, ymax), aspect='equal')
    ax[0].title.set_text("Fault Slip") #graph 1
    ax[0].set_ylabel("Latitude")
    ax[0].set_xlabel("Longitude")

    rso = ax[1].tripcolor(cmi["points"][:,0], cmi["points"][:,1], cmi["verts"], facecolors=(slip_vals[1]).flatten(), vmin=-max_mag_h, vmax=max_mag_h)
    cbar1 = fig.colorbar(rso, ax=ax[1], orientation='horizontal')
    cbar1.set_label("Slip (m)")
    #ax[1].quiver(gps.lon, gps.lat, pred_disp[0::3], pred_disp[1::3], scale=vec_scale, color='r', label="predicted")
    ax[1].quiver(gps.lon, gps.lat, gps.east_vel, gps.north_vel, scale=vecScale, color='k', label='observed')
    ax[1].set(xlim=(xmin-2, xmax), ylim=(ymin, ymax), aspect='equal')
    ax[1].title.set_text("CMI Slip") #graph 1
    ax[1].set_ylabel("Latitude")
    ax[1].set_xlabel("Longitude")

    if saveFigures and slipDist:
        plt.savefig('slip_dist.pdf') # save the figure
    
    plt.close('all')

    return


# plot displacements, including observed, predicted, and displacements separated
# by the component (i.e., displacement from CMI and displacement from fault)
def displacements(dispMat, allElemBegin, estSlip, predDisp, gps, vecScale, saveFigures=False, allDisp =False, dispSep=False, ratioFig=False) :
    # calculate displacements by which components

    # square the components
    east_disp = np.square(predDisp[0::3]).reshape(1,-1)
    north_disp = np.square(predDisp[1::3]).reshape(1, -1)

    # add together north and east disp, sum down column, take square root for magnitude of horiz disp
    totalDisp = np.sqrt(np.sum(np.vstack((east_disp, north_disp)), axis=0))

    # calc disp from cmi, beginning from the cmi elements to the end of the cmi elements
    cmi_disp = dispMat[:, allElemBegin[1]:allElemBegin[2]].dot(estSlip[allElemBegin[1]:allElemBegin[2]]) 
    cmi_e = np.square(cmi_disp[0::3]).reshape(1,-1) # square components
    cmi_n = np.square(cmi_disp[1::3]).reshape(1,-1)
    totalCmiDisp = np.sqrt(np.sum(np.vstack((cmi_e, cmi_n)), axis=0))

    # fault disp
    fault_disp = dispMat[:, allElemBegin[0]:allElemBegin[1]].dot(estSlip[allElemBegin[0]:allElemBegin[1]])
    fault_e = np.square(fault_disp[0::3]).reshape(1,-1) # square components
    fault_n = np.square(fault_disp[1::3]).reshape(1,-1)
    totalFaultDisp = np.sqrt(np.sum(np.vstack((fault_e, fault_n)), axis=0))

    plotRatio(gps, totalCmiDisp, totalDisp, saveFigures, ratioFig)

    fig, ax = plt.subplots(1, 2, figsize=(10,5))
    Q= ax[0].quiver(gps.lon, gps.lat, gps.east_vel, gps.north_vel, scale=vecScale, color='k', label='observed')
    ax[0].quiverkey(Q, X = 0.3, Y=0.8, U=100, label='100 cm',labelpos='N', color='r')
    ax[0].set_title("Observed displacements (cm)")
    ax[0].set_ylim([32.5, 45])
    ax[0].set_xlim([129, 143])

    Q1 = ax[1].quiver(gps.lon, gps.lat, predDisp[0::3], predDisp[1::3], scale=vecScale, color='r', label='predicted')
    ax[1].quiverkey(Q1, X=0.3, Y=0.8, U=100, label="100 cm", labelpos='N', color='r')
    #ax.quiver(gps.lon, gps.lat, data_vector[0:1497:3], data_vector[1:1497:3], scale=vec_scale, color='b')
    ax[1].set_ylim([32.5, 45])
    ax[1].set_xlim([129, 143])
    plt.title("Predicted Displacements (cm)")

    if saveFigures and allDisp:
        plt.savefig('totalDisp.pdf')


    fig, ax = plt.subplots(1, 2, figsize=(10,5))
    Q2 = ax[0].quiver(gps.lon, gps.lat, cmi_disp[0::3], cmi_disp[1::3], scale=vecScale, color='b', label="cmi contribution")
    ax[0].quiverkey(Q2, X=0.3, Y=0.8, U=100, label="100 cm", labelpos='N', color='b')
    ax[0].set_ylim([32.5, 45])
    ax[0].set_xlim([129, 143])
    ax[0].set_title("Cmi contribution")

    Q3 = ax[1].quiver(gps.lon, gps.lat, fault_disp[0::3], fault_disp[1::3], scale=vecScale, color='g', label="fault contribution")
    ax[1].quiverkey(Q3, X=0.3, Y=0.8, U=100, label="100 cm", labelpos='N', color='g')
    ax[1].set_ylim([32.5, 45])
    ax[1].set_xlim([129, 143])
    plt.title("fault contribution")

    if saveFigures and dispSep:
        plt.savefig('dispByComponent.pdf')
    
    plt.close('all')

    return


# plot the ratio of displacement due to cmi vs total displacements
def plotRatio(gps, totalCmiDisp, totalDisp, saveFigures, ratioFig) :
    ratio = totalCmiDisp / totalDisp
    lon_corr = 1 # longitude correction hardcoded to 1

    coast = pd.read_csv("coastline.csv")

    plt.close('all')
    fig, ax = plt.subplots()
    dots = ax.scatter(gps.lon, gps.lat, c=ratio) # color by ratio of cmi_disp / total disp
    ax.plot(coast.lon+360*(1-lon_corr), coast.lat, color="k", linewidth=0.5)
    ax.set_xlim(130, 145)
    ax.set_ylim(33, 45)
    # Add a customized colorbar
    cbar = fig.colorbar(dots, ax=ax, orientation='vertical', shrink=0.7,
                        label='Ratio of cmi:total', extend='both')

    if saveFigures and ratioFig:
        plt.savefig("ratioCmiToTotal.pdf")
    
    plt.close('all')

    return


def residualPlot(gps, predDisp, vecScale, saveFigures=False, residFig=False) :
    # calculate gps displacement residuals
    # residual = actual - predicted

    residuals = np.empty((len(gps.lon), 3))
    actual = np.hstack((np.array(gps.east_vel).reshape(-1,1), np.array(gps.north_vel).reshape(-1,1), np.array(gps.up_vel).reshape(-1,1)))

    # predicted / calculated values
    calc_east = predDisp[0::3].flatten()
    calc_north = predDisp[1::3].flatten()
    calc_up = predDisp[2::3].flatten()

    residuals[:,0] = actual[:,0] - calc_east
    residuals[:,1] = actual[:,1] - calc_north
    residuals[:,2] = actual[:,2] - calc_up


    plt.close('all')
    fig, ax = plt.subplots(1, 2, figsize=(10,5))
    #ax.quiver(gps.lon, gps.lat, gps.east_vel, gps.north_vel, scale=vec_scale, color='k', label='observed')
    Q = ax[0].quiver(gps.lon, gps.lat, residuals[:,0], residuals[:,1], scale=vecScale/20, color='r', label="residuals")
    ax[0].quiverkey(Q, X=0.3, Y=0.8, U=5, label="5cm", labelpos='N', color='r')
    #ax.quiver(gps.lon, gps.lat, data_vector[0:1497:3], data_vector[1:1497:3], scale=vec_scale, color='b')
    ax[0].set_ylim([32.5, 45])
    ax[0].set_xlim([129, 143])
    ax[0].set_title("Residual displacements (horizontal)")

    Q1 = ax[1].quiver(gps.lon, gps.lat, 0, residuals[:,2], scale=vecScale/20, color='r', label="residuals")
    ax[1].quiverkey(Q1, X=0.3, Y=0.8, U=5, label='5 cm', labelpos='N', color='r')
    ax[1].set_ylim([32.5, 45])
    ax[1].set_xlim([129, 143])
    plt.title("residual displacements (vertical)")

    if saveFigures and residFig:
        plt.savefig("residuals.pdf")
    
    plt.close('all')

    return


def maxSlipMag(estSlip, allElemBegin):
    slip_vals = [estSlip[0:allElemBegin[1]]/100, estSlip[allElemBegin[1]::]/100] # slip values for fault and CMI, converted from cm to m

    maxFaultMag = np.abs(np.max(slip_vals[0]))
    maxCmiMag = np.abs(np.max(slip_vals[1]))

    return maxFaultMag, maxCmiMag

def calcRMSE(predDisp, gps):
    residuals = np.empty((len(gps.lon), 3))
    actual = np.hstack((np.array(gps.east_vel).reshape(-1,1), np.array(gps.north_vel).reshape(-1,1), np.array(gps.up_vel).reshape(-1,1)))

    # predicted / calculated values
    calc_east = predDisp[0::3].flatten()
    calc_north = predDisp[1::3].flatten()
    calc_up = predDisp[2::3].flatten()

    residuals[:,0] = actual[:,0] - calc_east
    residuals[:,1] = actual[:,1] - calc_north
    residuals[:,2] = actual[:,2] - calc_up

    # calc root mean square residuals
    resid_sqr = np.square(residuals) # square all residuals
    # Calculate the Mean Squared Error (MSE)
    mse = np.mean(resid_sqr)
    # Calculate the Root Mean Square Error (RMSE)
    rmse = np.sqrt(mse)

    return rmse

def calcMoment(estSlip, allElemBegin, fault, cmi):
    # calculate the moment for slip on fault vs slip on cmi
    # moment = rigidity x area x slip

    rigidity = 3e10 #N / m^2 (is this the same though for the CMI at depth?)

    # allElemBegin[1] corresponds to the end of the fault elements and beginning of the cmi elements
    # calc slip mag
    fault_d_slip = np.square(estSlip[1:allElemBegin[1]:2]) # one for dip slip
    fault_s_slip = np.square(estSlip[0:allElemBegin[1]:2]) # zero for strike slip
    faultSlip = np.sqrt(np.sum(np.vstack((fault_d_slip, fault_s_slip))))

    faultMoment = rigidity * (fault["area"]) * faultSlip
    faultTotalMoment = np.sum(faultMoment) # in N / m^2

    cmi_d_slip = np.square(estSlip[1+allElemBegin[1]::3])
    cmi_s_slip = np.square(estSlip[0+allElemBegin[1]::3])
    cmiSlip = np.sqrt(np.sum(np.vstack((cmi_d_slip, cmi_s_slip))))

    cmiMoment = rigidity * (cmi["area"]) * cmiSlip
    cmiTotalMoment = np.sum(cmiMoment)

    return faultTotalMoment, cmiTotalMoment


def numericalData(estSlip, predDisp, gps, allElemBegin, fault, cmi, saveData):
    maxFaultMag, maxCmiMag = maxSlipMag(estSlip, allElemBegin)
    rmse = calcRMSE(predDisp, gps)
    faultMoment, cmiMoment = calcMoment(estSlip, allElemBegin, fault, cmi)

    print("Maximum magnitude of fault slip (m): ", maxFaultMag)
    print("Maximum magnitude of slip on cmi (m): ", maxCmiMag)
    # calc done in slip distribution plotting

    print("Fault moment: ", faultMoment)
    print("Horiz moment: ", cmiMoment)

    print(f"root mean square residual: %.3f cm" % rmse)
    # calc done in residual plotting

    if (saveData) :
        results = ["Maximum magnitude of fault slip (m): " + str(maxFaultMag), "Maximum magnitude of slip on cmi (m): " + str(maxCmiMag), 
                "Fault moment: " + str(faultMoment), "CMI moment: " + str(cmiMoment), "root mean square residual (cm): " + str(rmse)]
        with open("numericalResults.txt", "w") as file:
            for item in results:
                file.write("\n")
                file.write(item + "\n")

    return


def saveConfig(config):
    with open("configSettings.txt", 'w') as file:
        file.write(json.dumps(config, indent=4, sort_keys=True))
        
    return
