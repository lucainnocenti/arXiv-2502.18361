import numpy as np

# Define ns
ns = 5

def basis(s, n):    
    if n >= ns or n < 0:
        b = np.zeros((2 * ns,))  
    else:
        b = np.eye(2 * ns)[2 * n + s]  
    return b

# Define QP
def QP2(alpha, delta):
    result = 0
    for n in range(ns):
        term1 = np.cos(delta / 2) * (np.outer(basis(0,n), basis(0,n)) + np.outer(basis(1,n), basis(1,n)))
        term2 = 1j * np.sin(delta / 2) * (
            np.exp(2j * alpha) * np.outer(basis(1,n), basis(0,n + 1)) +
            np.exp(-2j * alpha) * np.outer(basis(0,n), basis(1,n - 1))
        )
        result += term1 + term2
    return result

# Define HWP
def HWP2(theta):
    return np.kron(np.eye(ns), np.array([[0, np.exp(2j * theta)], [np.exp(-2j * theta), 0]]))

# Define QWP
def QWP2(phi):
    return np.kron(np.eye(ns), np.array([
        [1/2 + 1j/2, (1/2 - 1j/2) * np.exp(2j * phi)],
        [(1/2 - 1j/2) * np.exp(-2j * phi), 1/2 + 1j/2]
    ]))

# RLtoHV = np.kron(np.eye(ns), np.array([[1, 1],[-1j, 1j]]) / np.sqrt(2))

def single_walker_isometry(alpha1, delta1, zeta2, theta2, phi2, alpha2, delta2, theta_p, phi_p):
    projection_matrix = QWP2(phi_p) @ HWP2(theta_p)
    evolution_matrix = QP2(alpha2, delta2) @ QWP2(phi2) @ HWP2(theta2) @ QWP2(zeta2) @ QP2(alpha1, delta1)
    U = projection_matrix @ evolution_matrix
    # return U projected over 0+1
    return (1 / np.sqrt(2)) * (U[0::2][:, ns-1:ns+1] + U[1::2][:, ns-1:ns+1])


def isometry(which_one='optimal'):
    if which_one == 'optimal':
        # this is the one that gives the minimal average MSE. Datasets like 2025-01-09 and 2025-01-21 use this one

        alpha1 = 19 * np.pi / 180
        delta1 = np.pi / 2
        alpha2 = 77 * np.pi / 180
        delta2 = np.pi
        zeta2 = 1.0070977275570985
        theta2 = 1.5570696348907915
        phi2 = 2.908160255954825
        theta_p = 4.539779498163959
        phi_p = 1.5707963284790725

        V1 = single_walker_isometry(alpha1, delta1, zeta2, theta2, phi2, alpha2, delta2, theta_p, phi_p)

        alpha1 = 336 * np.pi / 180
        delta1 = np.pi / 2
        alpha2 = 163 * np.pi / 180
        delta2 = np.pi
        zeta2 = 1.095755783171672
        theta2 = 1.5937676381596888
        phi2 = 2.89289820797498
        theta_p = 4.54821017223903
        phi_p = 1.5707963277463037

        V2 = single_walker_isometry(alpha1, delta1, zeta2, theta2, phi2, alpha2, delta2, theta_p, phi_p)

        return np.kron(V1, V2)
    
    elif which_one == 'old':
        # same as new but with zeta2 and phi2 swapped. 2024 datasets use this one

        alpha1 = 19 * np.pi / 180
        delta1 = np.pi / 2
        alpha2 = 77 * np.pi / 180
        delta2 = np.pi
        zeta2 = 2.908160255954825
        theta2 = 1.5570696348907915
        phi2 = 1.0070977275570985
        theta_p = 4.539779498163959
        phi_p = 1.5707963284790725

        V1 = single_walker_isometry(alpha1, delta1, zeta2, theta2, phi2, alpha2, delta2, theta_p, phi_p)

        alpha1 = 336 * np.pi / 180
        delta1 = np.pi / 2
        alpha2 = 163 * np.pi / 180
        delta2 = np.pi
        zeta2 = 2.89289820797498
        theta2 = 1.5937676381596888
        phi2 = 1.095755783171672
        theta_p = 4.54821017223903
        phi_p = 1.5707963277463037

        V2 = single_walker_isometry(alpha1, delta1, zeta2, theta2, phi2, alpha2, delta2, theta_p, phi_p)

        return np.kron(V1, V2)