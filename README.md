# fault_displacement
Wrappers/notebook for calculating displacement from fault slip

***NOTE***: Longitude correction for calculations has been hardcoded to 1 (Hokkaido range), and assumes that the maximum longitude value of the meshes is < 180.

`fault_displacement.ipynb` loads a fault geometry file, comprising triangular dislocation elements and created using [Gmsh](https://gmsh.info), and uses it as the basis for relating displacement to slip in an elastic halfspace. The elastic dislocation calculations are carried out using [cutde](https://github.com/tbenthompson/cutde), and the `environment.yml` file allows creation of a `conda` environment that includes the necessary libraries. 

# cmiModeling directory

#NOTE: Temporarily leaving out the tooHigh constraint (top 20 km of subduction zone) because I'm not sure it was properly implemented in the first place, and having a single constraint matrix for the fault and one for the cmi is clearer.
Originally : tooHigh = np.where(fault["centroids"][:,2]>-20)[0]
#NOTE: Temporarily removing cross section figure from new python files, simplicity and no longer as relevant in result processing.

To read in the results of past testing for new processing in cmiModeling, run
python3 main.py "--oldResults" "--resultFile=/path/to/testing/file"


GPS data is from the first 2 years after the earthquake, from Hu et al. doi:10.1186/1880-5981-66-106 