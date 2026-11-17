import numpy as np

def rotation_matrix_x(theta):
    """
    Generate a rotation matrix for rotation about the x-axis.

    Parameters
    ----------
    theta (float) : Angle in radians.
    """
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[1, 0, 0],
                     [0, c, s],
                     [0, -s,  c]])


def rotation_matrix_z(theta):
    """
    Generate a rotation matrix for rotation about the z-axis.
    
    Parameters
    ----------
    theta (float) : Angle in radians.
        
    """
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, s, 0],
                     [-s,  c, 0],
                     [0,  0, 1]])

def R_pCi(inc, Omega, omega):
    """
    Transformation matrix from perifocal to inertial coordinates.

    Parameters
    ----------
    inc (float) : Inclination in radians.
    Omega (float) : Longitude of ascending node in radians.
    omega (float) : Argument of periapsis in radians.
    """
    R = rotation_matrix_z(Omega) @ rotation_matrix_x(inc) @ rotation_matrix_z(omega)
    return R

def ang_momentum_vector(R_pCi):
    """
    Parameters
    ----------
    R_pCi (np.ndarray) : Transformation matrix from perifocal to inertial coordinates.

    Returns 
    -------
    np.ndarray
        Angular momentum vector in inertial coordinates.

    """
    return R_pCi[2] / np.linalg.norm(R_pCi[2])

def orthogonal_to_h(h_vector):
    '''
    Find the orthogonal vectors to the ang. momentum vector and the 
    rotation matrix to align points perpendicularly to h_vector
    
    Parameters
    ----------
    h_vector (np.ndarray) : Angular momentum vector

    Returns
    -------
    h_hat, v1, v2, rotation : np.ndarrays
        Unit vector for angular momentum, two orthogonal vectors to h (v1, v2), and the rotation
        to apply to points to align them perpendicularly to h
    '''
    # Normalize ang momentum vector
    h_hat = h_vector / np.linalg.norm(h_vector)

    # Find two orthogonal vectors in the planet orthogonal to h_hat
    if np.allclose(h_hat, [0, 0, 1]) or np.allclose(h_hat, [0, 0, -1]):
        # case where h is aligned with z-axis
        v1 = np.array([1, 0, 0])
    else:
        v1 = np.cross(h_hat, [0, 0, 1])
        v1 /= np.linalg.norm(v1)

    v2 = np.cross(h_hat, v1)
    v2 /= np.linalg.norm(v2)

    # Rotate points to be perpendicular to angular momentum vector
    rotation = np.column_stack((v1, v2, h_hat))

    return v1, rotation  
 
def obliquity_rot_matrix(obliquitiy_rad, v1):
    '''
    Generates rotation matrix for the obliquity angle

    Parameters
    ----------
    obliquity_rad (float) : Obliquity in radians

    Returns
    -------
    np.ndarray : Rotation matrix to apply for obliquity rotation
    '''
    cos_epsilon = np.cos(obliquitiy_rad)
    sin_epsilon = np.sin(obliquitiy_rad)
    u_x, u_y, u_z = v1 # v1 is parallel to the orbit plane, so use as rotation axis

    obl_rotation_matrix = np.array([ [cos_epsilon + u_x**2 * (1 - cos_epsilon), \
                                  u_x * u_y * (1 - cos_epsilon) - u_z * sin_epsilon, \
                                    u_x * u_z * (1 - cos_epsilon) + u_y * sin_epsilon],
                                 [ u_y * u_x * (1 - cos_epsilon) + u_z * sin_epsilon, \
                                  cos_epsilon + u_y**2 * (1 - cos_epsilon), \
                                    u_y * u_z * (1 - cos_epsilon) - u_x * sin_epsilon],
                                 [u_z * u_x * (1 - cos_epsilon) - u_y * sin_epsilon, \
                                   u_z * u_y * (1 - cos_epsilon) + u_x * sin_epsilon, \
                                    cos_epsilon + u_z**2 * (1 - cos_epsilon)]
                                 ])
    
    return obl_rotation_matrix

def get_ring_normal(ring_points):
    '''
    Determine the normal vector to the ring plane from the sampled ring points.

    Parameters
    ----------
    ring_points (np.ndarray) : Array of shape (N, 3) containing the 3D coordinates of points sampled on the ring plane.
    '''
    # Take 3 random points from the ring
    p1 = np.array([ring_points[10]])
    p2 = np.array([ring_points[4500]])
    p3 = np.array([ring_points[45000]])

    # Compute two vectors in the plane of the ring
    v1 = p2 - p1
    v2 = p3 - p1

    # Compute the normal vector using the cross product
    ring_normal = np.cross(v1, v2)
    ring_normal /= np.linalg.norm(ring_normal)  
    return -ring_normal[0] # flip normal to point away from observer

def kepler_orbit(a, e, inc, Omega, omega, f, theta):
    """
    Cartesian planet position from Keplerian elements.

    Parameters
    ----------
    a (float): Semi-major axis
    e (float): Orbital eccentricity
    inc (float): Inclination in radians
    Omega (float): Longitude of ascending node in radians
    omega (float): Argument of periapsis in radians
    f (float): True anomaly in radians
    theta (float): true_anomaly + omega
    """
    r = a * (1 - e**2) / (1 + e * np.cos(f))
    x = r * np.cos(f)
    y = r * np.sin(f)
    z = 0.0

    # compute projected separation w.r.t. observer
    s = r * np.sqrt( 1 - np.sin(inc)**2 * np.sin(theta)**2 )

    # Rotate to inertial frame
    R = rotation_matrix_z(Omega) @ rotation_matrix_x(inc) @ rotation_matrix_z(omega)

    return R @ np.array([x, y, z]), s

def full_orbit(a, e, inc, Omega, w, true_anomaly_arr=np.linspace(0, 2*np.pi, 200)):
    """
    Calculate full orbit.

    Parameters
    ----------
    a (float): Semi-major axis
    e (float): Orbital eccentricity
    inc (float): Inclination in radians
    Omega (float): Longitude of ascending node in radians
    w (float): Argument of periapsis in radians
    true_anomaly_arr (np.ndarray): Array of true anomaly values to sample
    """
    inc = np.pi - inc  
    orbit_coords = []
    for v in true_anomaly_arr:
        r = a * (1 - e**2) / (1 + e * np.cos(v))  # radial distance at each true anomaly
        x = r * (np.cos(Omega) * np.cos(w + v) - np.sin(Omega) * np.sin(w + v) * np.cos(inc))
        y = r * (np.sin(Omega) * np.cos(w + v) + np.cos(Omega) * np.sin(w + v) * np.cos(inc))
        z = r * np.sin(w + v) * np.sin(inc)
        orbit_coords.append([x, y, z])

    return np.array(orbit_coords)

def ring_normal_from_obliquity(obliquity, spin_longitude=0):
    n = np.array([
        np.sin(obliquity) * np.cos(spin_longitude),
        np.sin(obliquity) * np.sin(spin_longitude),
        np.cos(obliquity)
    ])
    return n / np.linalg.norm(n)

def sample_sphere(n_theta=60, n_phi=120):
    '''
    Sample points and normals on a unit sphere
    
    Parameters
    ----------
    n_theta (int) : Number of polar angle divisions
    n_phi (int) : Number of azimuthal angle divisions
    '''
    pts = []
    normals = []

    thetas = np.linspace(0, np.pi, n_theta)
    phis = np.linspace(0, 2*np.pi, n_phi)

    for t in thetas:
        for p in phis:
            x = np.sin(t)*np.cos(p)
            y = np.sin(t)*np.sin(p)
            z = np.cos(t)
            n = np.array([x, y, z])
            pts.append(n)
            normals.append(n)

    return np.array(pts), np.array(normals)

def sample_ring(r_inner, r_outer, n_r=50, n_phi=180):
    '''
    Sample points on a ring
    
    Parameters
    ----------
    r_inner (float): Inner radius of the ring
    r_outer (float): Outer radius of the ring
    n_r (int): Number of radial divisions on the ring
    n_phi (int): Number of azimuthal divisions on the ring
    '''
    pts = []

    rs = np.linspace(r_inner, r_outer, n_r)
    phis = np.linspace(0, 2*np.pi, n_phi)

    for r in rs:
        for p in phis:
            pts.append([r*np.cos(p), r*np.sin(p), 0])

    return np.array(pts)