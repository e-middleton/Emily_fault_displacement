import numpy as np
import pyproj

# Define some basic coordinate transformation functions
GEOID = pyproj.Geod(ellps="WGS84")
KM2M = 1.0e3
RADIUS_EARTH = np.float64((GEOID.a + GEOID.b) / 2)

def sph2cart(lon, lat, radius):
    lon_rad = np.deg2rad(lon)
    lat_rad = np.deg2rad(lat)
    x = radius * np.cos(lat_rad) * np.cos(lon_rad)
    y = radius * np.cos(lat_rad) * np.sin(lon_rad)
    z = radius * np.sin(lat_rad)
    return x, y, z

def cart2sph(x, y, z):
    azimuth = np.arctan2(y, x)
    elevation = np.arctan2(z, np.sqrt(x ** 2 + y ** 2))
    r = np.sqrt(x ** 2 + y ** 2 + z ** 2)
    return azimuth, elevation, r

def wrap2360(lon):
    lon[np.where(lon < 0.0)] += 360.0
    return lon


# function to expand mesh coordinates
# calculates normal vectors, cartesian normal vectors,
# latitudes, longitudes, and depths of the triangles in the mesh
# as well as calculates the area of the mesh
def expandMesh(mesh) : # where mesh is a dictionary object 

    mesh["lon1"] = mesh["points"][mesh["verts"][:, 0], 0]
    mesh["lon2"] = mesh["points"][mesh["verts"][:, 1], 0]
    mesh["lon3"] = mesh["points"][mesh["verts"][:, 2], 0]
    mesh["lat1"] = mesh["points"][mesh["verts"][:, 0], 1]
    mesh["lat2"] = mesh["points"][mesh["verts"][:, 1], 1]
    mesh["lat3"] = mesh["points"][mesh["verts"][:, 2], 1]
    mesh["dep1"] = mesh["points"][mesh["verts"][:, 0], 2]
    mesh["dep2"] = mesh["points"][mesh["verts"][:, 1], 2]
    mesh["dep3"] = mesh["points"][mesh["verts"][:, 2], 2]
    mesh["centroids"] = np.mean(mesh["points"][mesh["verts"], :], axis=1)


    # Cartesian coordinates in meters
    mesh["x1"], mesh["y1"], mesh["z1"] = sph2cart(
        mesh["lon1"],
        mesh["lat1"],
        RADIUS_EARTH + KM2M * mesh["dep1"],
    )
    mesh["x2"], mesh["y2"], mesh["z2"] = sph2cart(
        mesh["lon2"],
        mesh["lat2"],
        RADIUS_EARTH + KM2M * mesh["dep2"],
    )
    mesh["x3"], mesh["y3"], mesh["z3"] = sph2cart(
        mesh["lon3"],
        mesh["lat3"],
        RADIUS_EARTH + KM2M * mesh["dep3"],
    )
    # Cartesian triangle centroids
    mesh["x_centroid"] = (mesh["x1"] + mesh["x2"] + mesh["x3"]) / 3.0
    mesh["y_centroid"] = (mesh["y1"] + mesh["y2"] + mesh["y3"]) / 3.0
    mesh["z_centroid"] = (mesh["z1"] + mesh["z2"] + mesh["z3"]) / 3.0

    # Cross products for orientations
    tri_leg1 = np.transpose([np.deg2rad(mesh["lon2"] - mesh["lon1"]), np.deg2rad(mesh["lat2"] - mesh["lat1"]), (1 + KM2M * mesh["dep2"] / RADIUS_EARTH) - (1 + KM2M * mesh["dep1"] / RADIUS_EARTH)])
    tri_leg2 = np.transpose([np.deg2rad(mesh["lon3"] - mesh["lon1"]), np.deg2rad(mesh["lat3"] - mesh["lat1"]), (1 + KM2M * mesh["dep3"] / RADIUS_EARTH) - (1 + KM2M * mesh["dep1"] / RADIUS_EARTH)])
    mesh["nv"] = np.cross(tri_leg1, tri_leg2)
    azimuth, elevation, r = cart2sph(mesh["nv"][:, 0], mesh["nv"][:, 1], mesh["nv"][:, 2])
    mesh["strike"] = wrap2360(-np.rad2deg(azimuth))
    mesh["dip"] = 90 - np.rad2deg(elevation)
    mesh["dip_flag"] = mesh["dip"] != 90

    # calc mesh areas
    # Convert coordinates

    # Hokkaido range
    xs = np.linspace(120, 145, 200)
    ys = np.linspace(40, 45, 200)

    # Set up transformation ## LON_CORR HARDCODED ##
    lon_corr = 1
    # Check longitude convention of mesh
    if np.max(xs) > 180:
        lon_corr = 0

    utmzone=int(32700-(np.sign(np.mean(ys))+1)/2 * 100+np.floor((lon_corr*180 + np.mean(xs))/6) + 1)
    target_crs = 'epsg:'+str(utmzone) # Coordinate system of the file
    source_crs = 'epsg:4326' # Global lat-lon coordinate system
    latlon_to_utm = pyproj.Transformer.from_crs(source_crs, target_crs)

    meshxy = np.array(latlon_to_utm.transform(mesh["points"][:, 1], mesh["points"][:, 0])).T/1e3

    cart_pts = np.zeros_like(mesh["points"])
    cart_pts[:, 0:2] = meshxy
    cart_pts[:, 2] = mesh["points"][:, 2]

    cart_leg1 = cart_pts[mesh["verts"][:,1]] - cart_pts[mesh["verts"][:,0]]
    cart_leg2 = cart_pts[mesh["verts"][:,2]] - cart_pts[mesh["verts"][:,1]]
    mesh["cart_nv"] = np.cross(cart_leg1, cart_leg2)

    mesh["area"] = ((np.linalg.norm(mesh["cart_nv"],axis=1))/2) * (1e6)

    return mesh # return the modified mesh dictionary