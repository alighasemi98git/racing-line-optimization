from   racetrack_tools import RaceTrack
import numpy as np
import cvxpy as cp


def compute_racing_line(racetrack: RaceTrack, num_points: int = 1000):
    """
    Computes a minimum-acceleration racing line within track boundaries using
    convex optimization. Returns the vehicle positions.
    """
    racetrack  = racetrack
    s_values   = np.linspace(0, racetrack.length, num_points)
    s_delta    = racetrack.length / (num_points - 1)
    center     = np.array([racetrack.position(t) for t in s_values])
    num_points = num_points


    # TODO: Define cvxpy variables for vehicle positions (num_points x 2) and
    # the centerline offsets (vector of length num_points)
    center  = np.vstack([racetrack.position(s)       for s in s_values])  
    normals = np.vstack([racetrack.normal_vector(s)  for s in s_values])  
    
    positions = cp.Variable((num_points, 2))
    n         = cp.Variable(num_points)

    # TODO: Add constraint on centerline offset
    constraints = [cp.abs(n) <= racetrack.track_width / 2]

    # TODO: Relate positions to centerline offset (Eq. (1))
    constraints += [positions == center + cp.multiply(n[:, None], normals)]

    # TODO: Define acceleration objective (Eq. (4))
    N = num_points
    e = np.ones(N)
    D2 = (np.diag(e[:-2], k=0) - 2*np.diag(e[1:-1], k=0) + np.diag(e[2:], k=0))
    #difference matrix
    D2 = np.zeros((N-2, N))
    for i in range(1, N-1):
        D2[i-1, i-1] = 1.0
        D2[i-1, i  ] = -2.0
        D2[i-1, i+1] = 1.0

    
    acc = (D2 @ positions) / (s_delta**2)  

    # Objective
    objective = cp.sum(cp.norm(acc, axis=1)) * s_delta

    # TODO: Define cvxpy problem and solve it using MOSEK
    min_acc_problem = cp.Problem(cp.Minimize(objective), constraints)
    min_acc_problem.solve(solver=cp.MOSEK)

    return positions.value


def compute_lap_time(positions, g, mu, s_delta):
    v = np.diff(positions, n=1, axis=0)[:-1] / s_delta
    a = np.diff(positions, n=2, axis=0) / s_delta ** 2

    # TODO: Compute the curvature at each index i (Eq. (5))
    curvature = np.abs(a[:,1]*v[:,0] - a[:,0]*v[:,1]) / (v[:,0]**2 + v[:,1]**2)**(3/2)
    
    # TODO: Compute the absolute velocity at each index i
    V = np.sqrt(mu * g / curvature)

    # TODO: Compute the lap time
    time_deltas = s_delta / V
    lap_time = np.sum(time_deltas)

    return lap_time

