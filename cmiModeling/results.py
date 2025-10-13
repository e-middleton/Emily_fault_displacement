import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os


def slipDist(estSlip, gps, fault, cmi, vecScale, slipDist=False, saveFigures=False, figPath = "") :

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
        plt.savefig('slip_dist.png') # save the figure
        os.system('mv ./slip_dist.png ' + figPath) # move it into the test output folder

    plt.show()

    return


# plot displacements, including observed, predicted, and displacements separated
# by the component (i.e., displacement from CMI and displacement from fault)
def displacements(dispMat, allElemBegin, estSlip, predDisp, gps, vecScale, saveFigures=False, allDisp =False, dispSep=False, ratio=False, figPath="") :
    # calculate displacements by which components

    # square the components
    east_disp = np.square(predDisp[0::3]).reshape(1,-1)
    north_disp = np.square(predDisp[1::3]).reshape(1, -1)

    # add together north and east disp, sum down column, take square root for magnitude of horiz disp
    total_disp = np.sqrt(np.sum(np.vstack((east_disp, north_disp)), axis=0))

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
        plt.savefig('totalDisp.png')
        os.system('mv ./totalDisp.png ' + figPath) # move it into the test output folder


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
        plt.savefig('dispByComponent.png')
        os.system('mv ./dispByComponent.png ' + figPath)

    plt.show()

    return


# plot the ratio of displacement due to cmi vs total displacements
def plotRatio(gps, totalCmiDisp, totalDisp, saveFigures, ratioFig, figPath) :
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
        plt.savefig("ratioCmiToTotal.png")
        os.system("mv ./ratioCmiToTotal.png " + figPath)

    plt.show()

    return