import numpy as np
import matplotlib.pyplot as plt
import asyncio


def barrier_phi(x, D):
    # Log-barrier function for Euclidean ball ||x|| < D
    norm_sq = np.dot(x, x)
    if norm_sq >= D ** 2:
        raise ValueError("x is outside the domain (||x||^2 >= D^2)")
    return -np.log(1 - norm_sq / D**2)


def hessian_phi(x, D):
    # Hessian of the log-barrier function
    norm_sq = np.dot(x, x)
    if norm_sq >= D ** 2:
        raise ValueError("x is outside the domain (||x||^2 >= D^2)")
    
    d = len(x)
    denom = (1 - norm_sq / D**2)
    outer = np.outer(x, x)

    # Hessian = scaled identity + rank-1 correction
    H = (2 / D**2) * np.eye(d) / denom + (4 / (D**4 * denom**2)) * outer
    return H


def update_x(grad_sum, eta, D, delta, ShrunkScrible=0):
    # Compute the FTRL-style update using closed-form solution
    g = grad_sum
    G_sq = np.dot(g, g)
    
    # Avoid division by zero
    if G_sq == 0:
        return np.zeros(len(g))
    
    # Solve quadratic equation for alpha
    sqrt_term = np.sqrt(4 + 4 * (eta**2) * G_sq * (D**2))
    alpha = (-2 + sqrt_term) / (2 * eta * G_sq)
    
    x_t = -alpha * g

    # Optional shrinking to stay strictly inside feasible region
    radius = (1 - delta) * D
    norm = np.linalg.norm(x_t)
    
    
    if norm > radius and ShrunkScrible:
        x_t = x_t / norm * radius
        print('isShrunk')
    return x_t


def sigma(x, D, epsilon):
    """
    Adversarial perturbation term
    """
    norm = np.linalg.norm(x)
    diff = max(D - norm, 1e-6)   # not 0
    sinx = np.sin(1/(diff))

    return epsilon * abs(sinx)


async def ScriblePlay(epsilon, theta, u, d, T, D, G, g2, x2):
    # Standard SCRiBLe
    x_t = x2
    grad_sum = g2

    delta = 1 / T
    eta = np.sqrt(np.log(T)) / (4 * d * np.sqrt(T*(G*D+epsilon)))
    All_reward2 = 0.0
    
        # Compute local geometry (Hessian of barrier)
    hessian_barrier = hessian_phi(x_t, D)

    # Eigen-decomposition for inverse square root
    eigvals, eigvecs = np.linalg.eigh(hessian_barrier)
    eigvals = np.clip(eigvals, 1e-8, None)
    A = eigvecs @ np.diag(eigvals**(-0.5)) @ eigvecs.T

    # Sample direction from Dikin ellipsoid
    u = u.reshape(-1)
    y_t = x_t + np.dot(A, u)

    if np.linalg.norm(y_t) >= D:
        print("error: outside feasible set")
        
    epsilon3 = sigma(y_t, D, epsilon)

    # Observed reward
    theta = theta.reshape(-1)
    reward = np.dot(theta, y_t) + epsilon3

    # Gradient estimator
    A_inv = np.linalg.inv(A)
    g = d * reward * np.dot(A_inv, u)
    
    grad_sum += g

    # Update decision
    x_t = update_x(grad_sum, eta, D, delta)
    All_reward2 += reward
    return All_reward2, grad_sum, x_t


async def ShrunkScrible(epsilon, theta, u, d, T, D, G, g1, x1):
    # Shrunk SCRiBLe (keeps iterates away from boundary)
    x_t = x1
    grad_sum = g1

    delta = T**(-1)
    if epsilon != 0:
        delta = np.sqrt(epsilon)

    eta = np.sqrt(np.log(1 / delta)) / (4 * d * np.sqrt(T*(G*D+epsilon)))
    All_reward1 = 0.0
    
    hessian_barrier = hessian_phi(x_t, D)

    eigvals, eigvecs = np.linalg.eigh(hessian_barrier)
    eigvals = np.clip(eigvals, 1e-8, None)
    A = eigvecs @ np.diag(eigvals**(-0.5)) @ eigvecs.T
    
    u = u.reshape(-1)
    y_t = x_t + np.dot(A, u)
    if np.linalg.norm(y_t) >= D:
        print("error: outside feasible set")

    theta = theta.reshape(-1)
    epsilon3 = sigma(y_t, D, epsilon)

    reward = np.dot(theta, y_t) + epsilon3

    A_inv = np.linalg.inv(A)
    g = d * reward * np.dot(A_inv, u)
    
    grad_sum += g

    x_t = update_x(grad_sum, eta, D, delta, ShrunkScrible=1)
    All_reward1 += reward

    return All_reward1, grad_sum, x_t


async def TompspmSampling(epsilon, theta, d, D, rng, V, b,theta_hat):
    # Thompson Sampling baseline
    nu = 1.0
    All_reward3 = 0
    # Sample parameter
    cov = (nu ** 2) * np.linalg.inv(V)
    theta_tilde = rng.multivariate_normal(theta_hat, cov)

    # Optimal action under sampled parameter
    norm_theta = np.linalg.norm(theta_tilde)

    if norm_theta == 0:
        x_t = np.zeros(d)
    else:
        x_t = -D * theta_tilde / norm_theta

    # Observe reward
    epsilon2 = sigma(x_t, D, epsilon)
    theta = theta.reshape(-1)
    r_t = x_t @ theta + epsilon2

    All_reward3 += r_t

    # Update posterior
    V += np.outer(x_t, x_t)
    b += r_t * x_t
    theta_hat = np.linalg.solve(V, b)

    return All_reward3, V, b, theta_hat
    

async def main():
    C_loss_list1 = []
    C_loss_list2 = []
    C_loss_list3 = []
    
    d = 10
    T = 2000
    D = 1
    G = 10    
    n = 3
    C = 800
    epsilon = C / T
    d_list = []
    
    rng = np.random.default_rng(seed=42)
        
    for k in range(n):
        All_reward1 = 0
        All_reward2 = 0
        All_reward3 = 0
        
        g1 = np.zeros(d)
        g2 = np.zeros(d)
        x1 = np.zeros(d)
        x2 = np.zeros(d)
        lambda_ = 1.0
        V = lambda_ * np.eye(d)
        b = np.zeros(d)
        theta_hat = np.zeros(d)
        for t in range(T):
            theta = rng.normal(size=(1, d))
            # Normalize theta to fixed norm G
            Theta_norms = np.linalg.norm(theta, axis=1, keepdims=True)
            theta = theta / Theta_norms * G

            # Random directions (normalized)
            u = rng.normal(size=(1, d))
            # uniform distribution on the sphere
            u = u / np.linalg.norm(u, axis=1, keepdims=True)

            reward1, g1, x1 = await ShrunkScrible(epsilon, theta, u, d, T, D, G, g1, x1)
            reward2, g2, x2 = await ScriblePlay(epsilon, theta, u, d, T, D, G, g2, x2)
            reward3, V, b, theta_hat = await TompspmSampling(epsilon, theta, d, D, rng, V, b, theta_hat)

            All_reward1 += reward1
            All_reward2 += reward2
            All_reward3 += reward3
            
            print("Process", t)
        C_loss_list1.append(All_reward1)
        C_loss_list2.append(All_reward2)
        C_loss_list3.append(All_reward3)
        d_list.append(d)
        d = d * 10


    # results
    width = 0.2
    x = np.arange(len(d_list))
    plt.bar(x - width, C_loss_list1, width=width, color="orange", label="Shrunk-SCRiBLe")
    plt.bar(x, C_loss_list2, width=width, color="blue", label="SCRiBLe")
    plt.bar(x + width, C_loss_list3, width=width, color="green", label="Thompson Sampling")
    
    plt.xticks(x, d_list)

    plt.xlabel("d")
    plt.ylabel("Cumulative loss")
    plt.legend()
    plt.grid(axis='y', alpha=0.3)

    plt.savefig("figure.eps", format='eps', dpi=300, bbox_inches='tight')
    plt.show()


asyncio.run(main())
