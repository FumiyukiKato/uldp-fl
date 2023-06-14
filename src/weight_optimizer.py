import cvxpy as cp
import numpy as np


def optimize(S, U, e):
    # Define the size of S and U
    num_S = len(S)
    num_U = len(U)

    # Define the weights as a variable in cvxpy
    w = cp.Variable((num_S, num_U))

    # Define the objective function
    objective = cp.Minimize(
        cp.sum(cp.abs(cp.multiply(w, e) - cp.sum(cp.multiply(w, e)) / (num_S * num_U)))
    )

    # Define the constraints
    constraints = [cp.sum(w, axis=0) == 1, w >= 0]

    # Define the problem and solve it
    prob = cp.Problem(objective, constraints)
    opt_val = prob.solve()
    print("opt val", opt_val)
    print("mean val", np.mean(w.value * E))

    # Print the optimal w
    print(w.value)
    return w.value


if __name__ == "__main__":
    # silos
    S = [0, 1]

    # users
    U = [0, 1, 2]

    # E(s,u) = C/max(C, norm_of_delta(s,u,t))
    E = [
        [5, 5, 1],
        [5, 5, 5],
    ]
    # Define e as a matrix, replace this with your actual e values
    e = np.array(E)
    e = 1.0 / e

    optimize(S, U, e)
