# fault_displacement
Wrappers/notebook for calculating displacement from fault slip

***NOTE***: Longitude correction for calculations has been hardcoded to 1 (Hokkaido range), and assumes that the maximum longitude value of the meshes is < 180.

`fault_displacement.ipynb` loads a fault geometry file, comprising triangular dislocation elements and created using [Gmsh](https://gmsh.info), and uses it as the basis for relating displacement to slip in an elastic halfspace. The elastic dislocation calculations are carried out using [cutde](https://github.com/tbenthompson/cutde), and the `environment.yml` file allows creation of a `conda` environment that includes the necessary libraries. 
