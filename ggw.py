import numpy as np
import ot
from itertools import product
import time
from scipy.optimize import linprog

def init_extreme_points(lower_bounds,upper_bounds):
    """
    Initiate the extreme points for constraints: lower_bounds[i]<=wi<=upper_bounds[i]
    Input
    lower_bounds: ndarray(n,)
    upper_bounds: ndarray(n,)
    
    Return
    E: Extreme points ndarray(m,d)
    B: Bool matrix ndarray(n,m) with binary elements in which the element B_ij= 1 if extreme point j satisfies constraint i with equality
    D: Bool matrix ndarray(m,m) with binary elements in which the element D_ij= 1 if extreme point i and j are adjacent, i.e. extreme points that satisfies the same r − 1 constraints with equality, where r is the rank of the 
    """
    n = len(lower_bounds)
    
    A = np.r_[np.eye(n),-np.eye(n)]
    b = np.r_[upper_bounds, -lower_bounds]

    bounds = [(l, u) for l,u in zip(lower_bounds,upper_bounds)]
    
    E = np.array(list(product(*bounds)))
    
    # Binary matrix B indicating if a constraint is met with equality
    B = (A @ E.T ==  b[:,None] @ np.ones([1,len(E)]))

    # Rank of the problem, r
    #r = np.linalg.matrix_rank(A)
    r = E.shape[1]

    # Adjacency matrix D
    D = ((B.astype(int)).T @ B == r-1)

    return E,B,D

def extreme_points_update(E, B, D, A, b):
    """ Add a new constraint A @ x \le b to the problem and update the extreme points and the corresponding matrix B, D.
    Input
    E: Extreme points ndarray(n,d)
    B: Bool matrix with binary elements in which the element B_ij= 1 if extreme point j satisfies constraint i with equality
    D: Bool matrix with binary elements in which the element D_ij= 1 if extreme point i and j are adjacent, i.e. extreme points that satisfies the same r − 1 constraints with equality, where r is the rank of the problem.
    A: ndarray(1,d)
    b: ndarray(1,1) 
    
    Output
    E: New extreme points ndarray(m,d)
    B: Constraint bool matrix
    D: Adjacent bool matrix
    """
    
    new_constraint_values = A @ E.T - b[:,None] @ np.ones([1,len(E)])
    infeasible_indices = np.where(np.squeeze(new_constraint_values > 0))[0]

    # Collect feasible adjacent extreme points
    new_extreme_points = []
    C = []
    O = []

    for i in infeasible_indices:
        feasible_adjacent_indices = np.where(D[i] == 1)[0]
        for j in feasible_adjacent_indices:
            if j not in infeasible_indices:
                lamb = (b - A @ E[j].T)/(A @ (E[i] - E[j]).T)
                new_point = (1-lamb) *  E[j] + lamb * E[i]
                new_extreme_points.append(new_point)
                C.append(B[:, i] & B[:, j])
                new_O = np.zeros(len(E),dtype=bool)
                new_O[j] = True
                new_O[i] = True
                O.append(new_O)
                
    # Convert new_extreme_points to a numpy array and append to E
    if new_extreme_points:
        mask = np.ones(len(E), dtype=bool)
        mask[infeasible_indices] = False
        
        E = E[mask]
        B = B[:,mask]
        
        P = np.array(new_extreme_points)
        E = np.r_[E, P]
        b_row = np.zeros((1, B.shape[1])).astype(bool)
        b_row = np.column_stack((b_row, np.ones((1, len(new_extreme_points))).astype(bool)))
        C = np.array(C).T
        B = np.c_[B,C]
        B = np.r_[B, b_row]
        
        # Update adjacency matrix D
        r = E.shape[1]
        N = ((C.astype(int)).T @ C == r-2) | (np.eye(len(C.T),dtype=bool))
        O = np.array(O).T
        
        D = D[:,mask][mask]
        O = O[mask,:]
        D = np.block([[D,O],[O.T,N]])

    return E,B,D

def gw_obj(Cx,Cy,Pi):
    """
    Input:
    Cx: ndarray(nx,nx) Cx = \{\|x_i-x_j\|^2\}
    Cy: ndarray(ny,ny) Cy = \{\|y_i-y_j\|^2\}
    Pi: ndarray(nx,ny), permutation matrix sum(Pi,axis=0) = [1,1,...,1],sum(Pi,axis=1) = [1,1,...,1] 
    
    Return:
    GW value
    """
    return np.sum(Cx**2) + np.sum(Cy**2) - 2* np.sum((Cx @ Pi) * (Pi @ Cy))

def optimal_extreme_point(E):
    """
    \min_i -\|W_i\|^2-w_i+c
    
    Input:
    E: extreme points ndarray(n,d), where Ei = [w_i,vec(W_i)]
    
    Return:
    Ei: Minimizer
    Fi: Minimum value
    """
    F = - E[:,0] -  np.sum(E[:,1:]**2,axis=1)
    index = np.argmin(F)
    return E[index],F[index]

def gw_global(X,Y,epsilon=1e-6,IterMax=100,verbose=False,log=False, dist=False):
    """
    Globally solving the Gromov-Wasserstein problem for point clouds in low dimensional Euclidean spaces [1].
    
    Input:
    X: ndarray(nx,lx)
    Y: ndarray(ny,ly)
    epsilon: Stop threshold on bound gap
    IterMax: Max number of iterations
    verbose: Print information along iterations
    log: Record log if True
    
    Return:
    Pi: ndarray(nx,ny) Global optimal solution
    logs:  log dictionary return only if log==True in parameters
    
    [1] Ryner M, Kronqvist J, Karlsson J. Globally solving the Gromov-Wasserstein problem for point clouds in low dimensional Euclidean spaces[J]. Advances in Neural Information Processing Systems, 2024, 36.
    """
    # X ndarray (nx,lx)
    # Y ndarray (ny,ly)

    start_time = time.time()
    
    X = X.T
    Y = Y.T
        
    lx,nx = X.shape
    ly,ny = Y.shape
    l_bound = -np.inf
    u_bound = np.inf
    
    mx = (np.linalg.norm(X,axis=0)**2).reshape(-1,1)
    my = (np.linalg.norm(Y,axis=0)**2).reshape(-1,1)
    vec_1x = np.ones_like(mx)
    vec_1y = np.ones_like(my)
    
    L = (nx + ny) * mx @ my.T - 4 * mx @ vec_1y.T @ Y.T @ Y - 4 * X.T @ X @ vec_1x @ my.T

    Cx = vec_1x @ mx.T - 2 * X.T @ X + mx @ vec_1x.T
    Cy = vec_1y @ my.T - 2 * Y.T @ Y + my @ vec_1y.T
    c0 = np.sum(Cx**2) + np.sum(Cy**2) - 2 * np.sum(mx) * np.sum(my)
    
    W_lower = np.zeros([lx,ly])
    W_upper = np.zeros([lx,ly])

    a = np.ones(nx)
    b = np.ones(ny)
    
    if verbose:
        print('Iter |Bound gap'+ '\n' + '-' * 22)
    
    if log:
        E_cache = []
        E0_cache = []
        gap_cache = []
        Pi_cache = []
        l_cache = []
        u_cache = []
        c_cache = []
        time_cache = []
        obj_cache = []
    
    
    w_lower,w_upper = ot.emd2(a,b,L), -ot.emd2(a,b,-L)
    
    for i in range(lx):
        for j in range(ly):
            M = 2 * X[i,None].T @ Y[j,None]
            W_lower[i,j] = ot.emd2(a,b,M)
            W_upper[i,j] = - ot.emd2(a,b,-M)
    
    lower_bounds = np.r_[np.array([w_lower]),W_lower.reshape(-1)]
    upper_bounds = np.r_[np.array([w_upper]),W_upper.reshape(-1)]
    
    E,B,D = init_extreme_points(lower_bounds,upper_bounds)
    
    end_time = time.time()
    initialization_time = end_time - start_time
    
    for niter in range(IterMax):
        start_time = time.time()
        
        wW_cache,l_bound = optimal_extreme_point(E)
        l_bound = l_bound + c0
        Wn = wW_cache[1:].reshape(lx,ly) 
        
        M = (4*X.T @ Wn @ Y + L)
        
        Pi = ot.emd(a,b,-M)
        bound = - np.sum((2*X @ Pi @ Y.T)**2) - np.sum(L * Pi) + c0
        u_bound = np.min([u_bound,bound])

        if log:
            E_cache.append(E)
            gap_cache.append(u_bound - l_bound)
            l_cache.append(l_bound)
            u_cache.append(u_bound)
            Pi_cache.append(Pi)
            E0_cache.append(wW_cache)
            obj_cache.append(gw_obj(Cx,Cy,Pi))
            if niter==0:
                c_cache.append((lower_bounds,upper_bounds))
                time_cache.append(initialization_time)
            else:
                c_cache.append((A_,b_))
                time_cache.append(iteration_time)
        
        if u_bound - l_bound < epsilon:
            if verbose:
                print(f'{niter:5d}|{(u_bound-l_bound):8e}')
            break
        
        if verbose:
            if niter % 1 == 0:
                print(f'{niter:5d}|{(u_bound-l_bound):8e}')

        Zn = 2*Wn.reshape(-1)
        alphan = 1
        
        betan = np.sum(M*Pi)

        A_ = np.r_[np.array([alphan]),Zn]
        b_ = np.array([betan])
        
        E, B, D = extreme_points_update(E, B, D, A_, b_)
        
        end_time = time.time()
        iteration_time = end_time - start_time
    
    if u_bound - l_bound > epsilon:
        print('Warning: algorithm does not converge. Try larget IterMax.')
    
    if log:
        cum_time = np.cumsum(time_cache)
        logs = {'niter':niter+1,'obj_cache':obj_cache,'E_cache':E_cache,'gap_cache':gap_cache,'Pi_cache':Pi_cache,'E0_cache':E0_cache,'l_cache':l_cache,'u_cache':u_cache,'c_cache':c_cache,'time_cache':time_cache,'cum_time':cum_time,'L':L}
        return Pi,logs
    elif dist:
        return Pi, Cx, Cy
    else:
        return Pi


def gw_global_normalized(X,Y,epsilon=1e-6,IterMax=100,verbose=False,log=False, dist=False):
    """
    Globally solving the Gromov-Wasserstein problem for point clouds in low dimensional Euclidean spaces [1].
    
    Input:
    X: ndarray(nx,lx)
    Y: ndarray(ny,ly)
    epsilon: Stop threshold on bound gap
    IterMax: Max number of iterations
    verbose: Print information along iterations
    log: Record log if True
    
    Return:
    Pi: ndarray(nx,ny) Global optimal solution
    logs:  log dictionary return only if log==True in parameters
    
    [1] Ryner M, Kronqvist J, Karlsson J. Globally solving the Gromov-Wasserstein problem for point clouds in low dimensional Euclidean spaces[J]. Advances in Neural Information Processing Systems, 2024, 36.
    """
    # X ndarray (nx,lx)
    # Y ndarray (ny,ly)

    start_time = time.time()
    
    X = X.T
    Y = Y.T
        
    lx,nx = X.shape
    ly,ny = Y.shape
    l_bound = -np.inf
    u_bound = np.inf
    
    mx = (np.linalg.norm(X,axis=0)**2).reshape(-1,1)
    my = (np.linalg.norm(Y,axis=0)**2).reshape(-1,1)
    vec_1x = np.ones_like(mx)
    vec_1y = np.ones_like(my)
    
    # ORIGINAL
    # L = (nx + ny) * mx @ my.T - 4 * mx @ vec_1y.T @ Y.T @ Y - 4 * X.T @ X @ vec_1x @ my.T
    # Adjust L computation
    L = (nx + ny) * nx * ny * mx @ my.T - 4 * nx * ny * (mx @ vec_1y.T @ Y.T @ Y + X.T @ X @ vec_1x @ my.T)

    Cx = vec_1x @ mx.T - 2 * X.T @ X + mx @ vec_1x.T
    Cy = vec_1y @ my.T - 2 * Y.T @ Y + my @ vec_1y.T
    c0 = np.sum(Cx**2) + np.sum(Cy**2) - 2 * np.sum(mx) * np.sum(my)
    
    W_lower = np.zeros([lx,ly])
    W_upper = np.zeros([lx,ly])

    # MODIFIED:
    a = np.ones(nx) / nx
    b = np.ones(ny) / ny
    
    if verbose:
        print('Iter |Bound gap'+ '\n' + '-' * 22)
    
    if log:
        E_cache = []
        E0_cache = []
        gap_cache = []
        Pi_cache = []
        l_cache = []
        u_cache = []
        c_cache = []
        time_cache = []
        obj_cache = []
    
    
    w_lower,w_upper = ot.emd2(a,b,L), -ot.emd2(a,b,-L)
    
    for i in range(lx):
        for j in range(ly):
            M = 2 * X[i,None].T @ Y[j,None]
            W_lower[i,j] = ot.emd2(a,b,M)
            W_upper[i,j] = - ot.emd2(a,b,-M)
    
    lower_bounds = np.r_[np.array([w_lower]),W_lower.reshape(-1)]
    upper_bounds = np.r_[np.array([w_upper]),W_upper.reshape(-1)]
    
    E,B,D = init_extreme_points(lower_bounds,upper_bounds)
    
    end_time = time.time()
    initialization_time = end_time - start_time
    
    for niter in range(IterMax):
        start_time = time.time()
        
        wW_cache,l_bound = optimal_extreme_point(E)
        l_bound = l_bound + c0
        Wn = wW_cache[1:].reshape(lx,ly) 
        
        M = (4*X.T @ Wn @ Y + L)
        
        Pi = ot.emd(a,b,-M)
        bound = - np.sum((2*X @ Pi @ Y.T)**2) - np.sum(L * Pi) + c0
        u_bound = np.min([u_bound,bound])

        if log:
            E_cache.append(E)
            gap_cache.append(u_bound - l_bound)
            l_cache.append(l_bound)
            u_cache.append(u_bound)
            Pi_cache.append(Pi)
            E0_cache.append(wW_cache)
            obj_cache.append(gw_obj(Cx,Cy,Pi))
            if niter==0:
                c_cache.append((lower_bounds,upper_bounds))
                time_cache.append(initialization_time)
            else:
                c_cache.append((A_,b_))
                time_cache.append(iteration_time)
        
        if u_bound - l_bound < epsilon:
            if verbose:
                print(f'{niter:5d}|{(u_bound-l_bound):8e}')
            break
        
        if verbose:
            if niter % 1 == 0:
                print(f'{niter:5d}|{(u_bound-l_bound):8e}')

        Zn = 2*Wn.reshape(-1)
        alphan = 1
        
        betan = np.sum(M*Pi)

        A_ = np.r_[np.array([alphan]),Zn]
        b_ = np.array([betan])
        
        E, B, D = extreme_points_update(E, B, D, A_, b_)
        
        end_time = time.time()
        iteration_time = end_time - start_time
    
    if u_bound - l_bound > epsilon:
        print('Warning: algorithm does not converge. Try larget IterMax.')
    
    if log:
        cum_time = np.cumsum(time_cache)
        logs = {'niter':niter+1,'obj_cache':obj_cache,'E_cache':E_cache,'gap_cache':gap_cache,'Pi_cache':Pi_cache,'E0_cache':E0_cache,'l_cache':l_cache,'u_cache':u_cache,'c_cache':c_cache,'time_cache':time_cache,'cum_time':cum_time,'L':L}
        return Pi,logs
    elif dist:
        return Pi, Cx, Cy
    else:
        return Pi

def gw_global_testing(X,Y,epsilon=1e-6,IterMax=100,verbose=False,log=False, dist=False):
    """
    Globally solving the Gromov-Wasserstein problem for point clouds in low dimensional Euclidean spaces [1].
    
    Input:
    X: ndarray(nx,lx)
    Y: ndarray(ny,ly)
    epsilon: Stop threshold on bound gap
    IterMax: Max number of iterations
    verbose: Print information along iterations
    log: Record log if True
    
    Return:
    Pi: ndarray(nx,ny) Global optimal solution
    logs:  log dictionary return only if log==True in parameters
    
    [1] Ryner M, Kronqvist J, Karlsson J. Globally solving the Gromov-Wasserstein problem for point clouds in low dimensional Euclidean spaces[J]. Advances in Neural Information Processing Systems, 2024, 36.
    """
    # X ndarray (nx,lx)
    # Y ndarray (ny,ly)

    start_time = time.time()
    
    X = X.T
    Y = Y.T
        
    lx,nx = X.shape
    ly,ny = Y.shape
    l_bound = -np.inf
    u_bound = np.inf
    
    mx = (np.linalg.norm(X,axis=0)**2).reshape(-1,1)
    my = (np.linalg.norm(Y,axis=0)**2).reshape(-1,1)
    vec_1x = np.ones_like(mx)
    vec_1y = np.ones_like(my)
    
    L = (nx + ny) * mx @ my.T - 4 * mx @ vec_1y.T @ Y.T @ Y - 4 * X.T @ X @ vec_1x @ my.T

    Cx = vec_1x @ mx.T - 2 * X.T @ X + mx @ vec_1x.T
    Cy = vec_1y @ my.T - 2 * Y.T @ Y + my @ vec_1y.T
    c0 = np.sum(Cx**2) + np.sum(Cy**2) - 2 * np.sum(mx) * np.sum(my)
    
    W_lower = np.zeros([lx,ly])
    W_upper = np.zeros([lx,ly])

    a = np.ones(nx)
    b = np.ones(ny)
    
    if verbose:
        print('Iter |Bound gap'+ '\n' + '-' * 22)
    
    if log:
        E_cache = []
        E0_cache = []
        gap_cache = []
        Pi_cache = []
        l_cache = []
        u_cache = []
        c_cache = []
        time_cache = []
        obj_cache = []
    
    # MODIFIED: NUMERICAL STABILITY THROUGH SCALING AND EARLY STOPPING USING ADAPTIVE EPSILON
    def scaled_emd2(a,b,M):
        scale = max(1.0, np.max(np.abs(M)))
        M_scaled = M/scale
        return scale * ot.emd2(a,b,M_scaled)
    
    def adaptive_epsilon(iter_num, base_epsilon=1e-6):
        return base_epsilon * (1 + 0.1 * iter_num)
    
    w_lower,w_upper = scaled_emd2(a,b,L), -scaled_emd2(a,b,-L)
    
    for i in range(lx):
        for j in range(ly):
            M = 2 * X[i,None].T @ Y[j,None]
            W_lower[i,j] = scaled_emd2(a,b,M)
            W_upper[i,j] = - scaled_emd2(a,b,-M)
    
    lower_bounds = np.r_[np.array([w_lower]),W_lower.reshape(-1)]
    upper_bounds = np.r_[np.array([w_upper]),W_upper.reshape(-1)]
    
    E,B,D = init_extreme_points(lower_bounds,upper_bounds)
    
    end_time = time.time()
    initialization_time = end_time - start_time
    
    for niter in range(IterMax):
        start_time = time.time()
        
        wW_cache,l_bound = optimal_extreme_point(E)
        l_bound = l_bound + c0
        Wn = wW_cache[1:].reshape(lx,ly) 
        
        M = (4*X.T @ Wn @ Y + L)
        
        Pi = ot.emd(a,b,-M)
        bound = - np.sum((2*X @ Pi @ Y.T)**2) - np.sum(L * Pi) + c0
        u_bound = np.min([u_bound,bound])

        if log:
            E_cache.append(E)
            gap_cache.append(u_bound - l_bound)
            l_cache.append(l_bound)
            u_cache.append(u_bound)
            Pi_cache.append(Pi)
            E0_cache.append(wW_cache)
            obj_cache.append(gw_obj(Cx,Cy,Pi))
            if niter==0:
                c_cache.append((lower_bounds,upper_bounds))
                time_cache.append(initialization_time)
            else:
                c_cache.append((A_,b_))
                time_cache.append(iteration_time)
        
        # MODIFIED: 
        current_epsilon = adaptive_epsilon(niter, epsilon)

        if u_bound - l_bound < current_epsilon:
            if verbose:
                print(f'{niter:5d}|{(u_bound-l_bound):8e}')
            break
        
        if verbose:
            if niter % 1 == 0:
                print(f'{niter:5d}|{(u_bound-l_bound):8e}')

        Zn = 2*Wn.reshape(-1)
        alphan = 1
        
        betan = np.sum(M*Pi)

        A_ = np.r_[np.array([alphan]),Zn]
        b_ = np.array([betan])
        
        E, B, D = extreme_points_update(E, B, D, A_, b_)
        
        end_time = time.time()
        iteration_time = end_time - start_time
    
    if u_bound - l_bound > epsilon:
        print('Warning: algorithm does not converge. Try larget IterMax.')
    
    if log:
        cum_time = np.cumsum(time_cache)
        logs = {'niter':niter+1,'obj_cache':obj_cache,'E_cache':E_cache,'gap_cache':gap_cache,'Pi_cache':Pi_cache,'E0_cache':E0_cache,'l_cache':l_cache,'u_cache':u_cache,'c_cache':c_cache,'time_cache':time_cache,'cum_time':cum_time,'L':L}
        return Pi,logs
    elif dist:
        return Pi, Cx, Cy
    else:
        return Pi




# def gw_global_part_wrong(X,Y,epsilon=1e-6,IterMax=100,verbose=False,log=False, dist=False):
#     """
#     Globally solving the Gromov-Wasserstein problem for point clouds in low dimensional Euclidean spaces [1].
    
#     Input:
#     X: ndarray(nx,lx)
#     Y: ndarray(ny,ly)
#     epsilon: Stop threshold on bound gap
#     IterMax: Max number of iterations
#     verbose: Print information along iterations
#     log: Record log if True
    
#     Return:
#     Pi: ndarray(nx,ny) Global optimal solution
#     logs:  log dictionary return only if log==True in parameters
    
#     [1] Ryner M, Kronqvist J, Karlsson J. Globally solving the Gromov-Wasserstein problem for point clouds in low dimensional Euclidean spaces[J]. Advances in Neural Information Processing Systems, 2024, 36.
#     """
#     # X ndarray (nx,lx)
#     # Y ndarray (ny,ly)

#     start_time = time.time()
    
#     X = X.T
#     Y = Y.T
        
#     lx,nx = X.shape
#     ly,ny = Y.shape
#     l_bound = -np.inf
#     u_bound = np.inf
    
#     mx = (np.linalg.norm(X,axis=0)**2).reshape(-1,1)
#     my = (np.linalg.norm(Y,axis=0)**2).reshape(-1,1)
#     vec_1x = np.ones_like(mx)
#     vec_1y = np.ones_like(my)
    
#     L = (nx + ny) * mx @ my.T - 4 * mx @ vec_1y.T @ Y.T @ Y - 4 * X.T @ X @ vec_1x @ my.T

#     Cx = vec_1x @ mx.T - 2 * X.T @ X + mx @ vec_1x.T
#     Cy = vec_1y @ my.T - 2 * Y.T @ Y + my @ vec_1y.T
#     c0 = np.sum(Cx**2) + np.sum(Cy**2) - 2 * np.sum(mx) * np.sum(my)
    
#     W_lower = np.zeros([lx,ly])
#     W_upper = np.zeros([lx,ly])

#     a = np.ones(nx)
#     b = np.ones(ny)
    
#     if verbose:
#         print('Iter |Bound gap'+ '\n' + '-' * 22)
    
#     if log:
#         E_cache = []
#         E0_cache = []
#         gap_cache = []
#         Pi_cache = []
#         l_cache = []
#         u_cache = []
#         c_cache = []
#         time_cache = []
#         obj_cache = []
    
#     # ORIGINAL:
#     # w_lower,w_upper = ot.emd2(a,b,L), -ot.emd2(a,b,-L)
#     # MODIFIED:
#     w_lower = np.zeros([lx,ly])
#     w_upper = np.zeros([lx,ly])
#     reg = 0.01
#     L_min = L.min()
#     L_shifted = L - L_min
#     w_lower_scalar = float(ot.sinkhorn2(a,b,L_shifted, reg)) + L_min
#     w_upper_scalar = float(ot.sinkhorn2(a,b,-L_shifted, reg)) - L_min

#     # ORIGINAL:
#     # for i in range(lx):
#     #     for j in range(ly):
#     #         M = 2 * X[i,None].T @ Y[j,None]
#     #         W_lower[i,j] = ot.emd2(a,b,M)
#     #         W_upper[i,j] = - ot.emd2(a,b,-M)
#     # MODIFIED:
#     for i in range(lx):
#         for j in range(ly):
#             M = 2 * X[i,None].T @ Y[j,None]
#             M_min = M.min()
#             M_shifted = M - M_min
#             w_lower[i,j] = float(ot.sinkhorn2(a,b,M_shifted,reg)) + M_min
#             w_upper[i,j] = float(ot.sinkhorn2(a,b,-M_shifted,reg)) - M_min
    
#     lower_bounds = np.r_[np.array([w_lower_scalar]),W_lower.reshape(-1)]
#     upper_bounds = np.r_[np.array([w_upper_scalar]),W_upper.reshape(-1)]
    
#     E,B,D = init_extreme_points(lower_bounds,upper_bounds)
    
#     end_time = time.time()
#     initialization_time = end_time - start_time
    
#     for niter in range(IterMax):
#         start_time = time.time()
        
#         wW_cache,l_bound = optimal_extreme_point(E)
#         l_bound = l_bound + c0
#         Wn = wW_cache[1:].reshape(lx,ly) 
        
#         M = (4*X.T @ Wn @ Y + L)
        
#         # ORIGINAL:
#         # Pi = ot.emd(a,b,-M)
#         # MODIFIED:
#         M_min = M.min()
#         M_shifted = M - M_min
#         Pi = ot.sinkhorn(a,b,M_shifted,reg)

#         bound = - np.sum((2*X @ Pi @ Y.T)**2) - np.sum(L * Pi) + c0
#         u_bound = np.min([u_bound,bound])

#         if log:
#             E_cache.append(E)
#             gap_cache.append(u_bound - l_bound)
#             l_cache.append(l_bound)
#             u_cache.append(u_bound)
#             Pi_cache.append(Pi)
#             E0_cache.append(wW_cache)
#             obj_cache.append(gw_obj(Cx,Cy,Pi))
#             if niter==0:
#                 c_cache.append((lower_bounds,upper_bounds))
#                 time_cache.append(initialization_time)
#             else:
#                 c_cache.append((A_,b_))
#                 time_cache.append(iteration_time)
        
#         if u_bound - l_bound < epsilon:
#             if verbose:
#                 print(f'{niter:5d}|{(u_bound-l_bound):8e}')
#             break
        
#         if verbose:
#             if niter % 1 == 0:
#                 print(f'{niter:5d}|{(u_bound-l_bound):8e}')

#         Zn = 2*Wn.reshape(-1)
#         alphan = 1
        
#         betan = np.sum(M*Pi)

#         A_ = np.r_[np.array([alphan]),Zn]
#         b_ = np.array([betan])
        
#         E, B, D = extreme_points_update(E, B, D, A_, b_)
        
#         end_time = time.time()
#         iteration_time = end_time - start_time
    
#     if u_bound - l_bound > epsilon:
#         print('Warning: algorithm does not converge. Try larget IterMax.')
    
#     if log:
#         cum_time = np.cumsum(time_cache)
#         logs = {'niter':niter+1,'obj_cache':obj_cache,'E_cache':E_cache,'gap_cache':gap_cache,'Pi_cache':Pi_cache,'E0_cache':E0_cache,'l_cache':l_cache,'u_cache':u_cache,'c_cache':c_cache,'time_cache':time_cache,'cum_time':cum_time,'L':L}
#         return Pi,logs
#     elif dist:
#         return Pi, Cx, Cy
#     else:
#         return Pi

# def gw_global_sinkhorn2(X,Y,epsilon=1e-6,IterMax=100,verbose=False,log=False, dist=False):
#     """
#     Globally solving the Gromov-Wasserstein problem for point clouds in low dimensional Euclidean spaces [1].
    
#     Input:
#     X: ndarray(nx,lx)
#     Y: ndarray(ny,ly)
#     epsilon: Stop threshold on bound gap
#     IterMax: Max number of iterations
#     verbose: Print information along iterations
#     log: Record log if True
    
#     Return:
#     Pi: ndarray(nx,ny) Global optimal solution
#     logs:  log dictionary return only if log==True in parameters
    
#     [1] Ryner M, Kronqvist J, Karlsson J. Globally solving the Gromov-Wasserstein problem for point clouds in low dimensional Euclidean spaces[J]. Advances in Neural Information Processing Systems, 2024, 36.
#     """
#     # X ndarray (nx,lx)
#     # Y ndarray (ny,ly)

#     start_time = time.time()
    
#     X = X.T
#     Y = Y.T
        
#     lx,nx = X.shape
#     ly,ny = Y.shape
#     l_bound = -np.inf
#     u_bound = np.inf
    
#     mx = (np.linalg.norm(X,axis=0)**2).reshape(-1,1)
#     my = (np.linalg.norm(Y,axis=0)**2).reshape(-1,1)
#     vec_1x = np.ones_like(mx)
#     vec_1y = np.ones_like(my)
    
#     L = (nx + ny) * mx @ my.T - 4 * mx @ vec_1y.T @ Y.T @ Y - 4 * X.T @ X @ vec_1x @ my.T

#     Cx = vec_1x @ mx.T - 2 * X.T @ X + mx @ vec_1x.T
#     Cy = vec_1y @ my.T - 2 * Y.T @ Y + my @ vec_1y.T
#     c0 = np.sum(Cx**2) + np.sum(Cy**2) - 2 * np.sum(mx) * np.sum(my)
    
#     W_lower = np.zeros([lx,ly])
#     W_upper = np.zeros([lx,ly])

#     a = np.ones(nx)
#     b = np.ones(ny)
    
#     if verbose:
#         print('Iter |Bound gap'+ '\n' + '-' * 22)
    
#     if log:
#         E_cache = []
#         E0_cache = []
#         gap_cache = []
#         Pi_cache = []
#         l_cache = []
#         u_cache = []
#         c_cache = []
#         time_cache = []
#         obj_cache = []


#     ######
#     c = L.flatten()
#     A_eq = np.ones((a.shape[0], a.shape[0] * b.shape[0]))
#     for i in range(a.shape[0]):
#         A_eq[i, i*b.shape[0]:(i+1)*b.shape[0]] = 1

#     b_eq = np.concatenate((a, b))
#     bounds = [(0, None) for _ in range(len(c))]

#     res = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=bounds)

#     # Extract the optimal transport plan
#     optimal_plan = np.array(res.x).reshape(L.shape)

#     # Calculate w_lower as the sum of the element-wise product of optimal_plan and L
#     w_lower = np.sum(optimal_plan * L)

#     # Calculate w_upper as the sum of the element-wise product of optimal_plan and -L
#     w_upper = -np.sum(optimal_plan * (-L))
    
#     # reg = 1e-6
#     # w_lower, w_upper = ot.bregman.sinkhorn2(a,b,L,reg), -ot.emd2(a,b,-L,reg)
#     print(w_lower)
#     print(w_upper)

#     #######
    
#     for i in range(lx):
#         for j in range(ly):
#             M = 2 * X[i,None].T @ Y[j,None]

#             c = M.flatten()
#             res = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=bounds)
#             optimal_plan = np.array(res.x).reshape(L.shape)

#             # W_lower[i,j] = ot.bregman.sinkhorn2(a,b,M,reg)
#             # W_upper[i,j] = - ot.emd2(a,b,-M,reg)
#             W_lower[i,j] = np.sum(optimal_plan * M)
#             W_upper[i,j] = -np.sum(optimal_plan * (-M))
    
#     lower_bounds = np.r_[np.array([w_lower]),W_lower.reshape(-1)]
#     upper_bounds = np.r_[np.array([w_upper]),W_upper.reshape(-1)]
    
#     E,B,D = init_extreme_points(lower_bounds,upper_bounds)
    
#     end_time = time.time()
#     initialization_time = end_time - start_time
    
#     for niter in range(IterMax):
#         start_time = time.time()
        
#         wW_cache,l_bound = optimal_extreme_point(E)
#         l_bound = l_bound + c0
#         Wn = wW_cache[1:].reshape(lx,ly) 
        
#         M = (4*X.T @ Wn @ Y + L)
        
#         c = -M.flatten()
#         A_eq = np.ones((a.shape[0], a.shape[0] * b.shape[0]))
#         for i in range(a.shape[0]):
#             A_eq[i, i * b.shape[0] : (i + 1) * b.shape[0]] = 1

#         b_eq = np.concatenate((a, b))
#         bounds = [(0, None) for _ in range(len(c))]
#         res = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=bounds)
#         Pi = np.array(res.x).reshape(M.shape)

#         # Pi = ot.emd(a,b,-M,reg)
#         bound = - np.sum((2*X @ Pi @ Y.T)**2) - np.sum(L * Pi) + c0
#         u_bound = np.min([u_bound,bound])

#         if log:
#             E_cache.append(E)
#             gap_cache.append(u_bound - l_bound)
#             l_cache.append(l_bound)
#             u_cache.append(u_bound)
#             Pi_cache.append(Pi)
#             E0_cache.append(wW_cache)
#             obj_cache.append(gw_obj(Cx,Cy,Pi))
#             if niter==0:
#                 c_cache.append((lower_bounds,upper_bounds))
#                 time_cache.append(initialization_time)
#             else:
#                 c_cache.append((A_,b_))
#                 time_cache.append(iteration_time)
        
#         if u_bound - l_bound < epsilon:
#             if verbose:
#                 print(f'{niter:5d}|{(u_bound-l_bound):8e}')
#             break
        
#         if verbose:
#             if niter % 1 == 0:
#                 print(f'{niter:5d}|{(u_bound-l_bound):8e}')

#         Zn = 2*Wn.reshape(-1)
#         alphan = 1
        
#         betan = np.sum(M*Pi)

#         A_ = np.r_[np.array([alphan]),Zn]
#         b_ = np.array([betan])
        
#         E, B, D = extreme_points_update(E, B, D, A_, b_)
        
#         end_time = time.time()
#         iteration_time = end_time - start_time
    
#     if u_bound - l_bound > epsilon:
#         print('Warning: algorithm does not converge. Try larget IterMax.')
    
#     if log:
#         cum_time = np.cumsum(time_cache)
#         logs = {'niter':niter+1,'obj_cache':obj_cache,'E_cache':E_cache,'gap_cache':gap_cache,'Pi_cache':Pi_cache,'E0_cache':E0_cache,'l_cache':l_cache,'u_cache':u_cache,'c_cache':c_cache,'time_cache':time_cache,'cum_time':cum_time,'L':L}
#         return Pi,logs
#     elif dist:
#         return Pi, Cx, Cy
#     else:
#         return Pi


# def gw_global_batch(X,Y,epsilon=1e-6,IterMax=100,batch_size=5,verbose=False,log=False, dist=False):
#     """
#     Globally solving the Gromov-Wasserstein problem for point clouds in low dimensional Euclidean spaces [1].
    
#     Input:
#     X: ndarray(nx,lx)
#     Y: ndarray(ny,ly)
#     epsilon: Stop threshold on bound gap
#     IterMax: Max number of iterations
#     verbose: Print information along iterations
#     log: Record log if True
    
#     Return:
#     Pi: ndarray(nx,ny) Global optimal solution
#     logs:  log dictionary return only if log==True in parameters
    
#     [1] Ryner M, Kronqvist J, Karlsson J. Globally solving the Gromov-Wasserstein problem for point clouds in low dimensional Euclidean spaces[J]. Advances in Neural Information Processing Systems, 2024, 36.
#     """
#     # X ndarray (nx,lx)
#     # Y ndarray (ny,ly)

#     start_time = time.time()
    
#     X = X.T
#     Y = Y.T
        
#     lx,nx = X.shape
#     ly,ny = Y.shape
#     l_bound = -np.inf
#     u_bound = np.inf
    
#     mx = (np.linalg.norm(X,axis=0)**2).reshape(-1,1)
#     my = (np.linalg.norm(Y,axis=0)**2).reshape(-1,1)
#     vec_1x = np.ones_like(mx)
#     vec_1y = np.ones_like(my)
    
#     L = (nx + ny) * mx @ my.T - 4 * mx @ vec_1y.T @ Y.T @ Y - 4 * X.T @ X @ vec_1x @ my.T

#     Cx = vec_1x @ mx.T - 2 * X.T @ X + mx @ vec_1x.T
#     Cy = vec_1y @ my.T - 2 * Y.T @ Y + my @ vec_1y.T
#     c0 = np.sum(Cx**2) + np.sum(Cy**2) - 2 * np.sum(mx) * np.sum(my)
    
#     W_lower = np.zeros([lx,ly])
#     W_upper = np.zeros([lx,ly])

#     a = np.ones(nx)
#     b = np.ones(ny)
    
#     if verbose:
#         print('Iter |Bound gap'+ '\n' + '-' * 22)
    
#     if log:
#         E_cache = []
#         E0_cache = []
#         gap_cache = []
#         Pi_cache = []
#         l_cache = []
#         u_cache = []
#         c_cache = []
#         time_cache = []
#         obj_cache = []
    
    
#     w_lower,w_upper = ot.emd2(a,b,L), -ot.emd2(a,b,-L)
    
#     for i in range(lx):
#         for j in range(ly):
#             M = 2 * X[i,None].T @ Y[j,None]
#             W_lower[i,j] = ot.emd2(a,b,M)
#             W_upper[i,j] = - ot.emd2(a,b,-M)
    
#     lower_bounds = np.r_[np.array([w_lower]),W_lower.reshape(-1)]
#     upper_bounds = np.r_[np.array([w_upper]),W_upper.reshape(-1)]
    
#     E,B,D = init_extreme_points(lower_bounds,upper_bounds)
    
#     end_time = time.time()
#     initialization_time = end_time - start_time
    
#     for niter in range(IterMax):
#         start_time = time.time()
        
#         wW_cache,l_bound = optimal_extreme_point(E)
#         l_bound = l_bound + c0
#         Wn = wW_cache[1:].reshape(lx,ly) 
        
#         M = (4*X.T @ Wn @ Y + L)
        
#         Pi = ot.emd(a,b,-M)
#         bound = - np.sum((2*X @ Pi @ Y.T)**2) - np.sum(L * Pi) + c0
#         u_bound = np.min([u_bound,bound])

#         if log:
#             E_cache.append(E)
#             gap_cache.append(u_bound - l_bound)
#             l_cache.append(l_bound)
#             u_cache.append(u_bound)
#             Pi_cache.append(Pi)
#             E0_cache.append(wW_cache)
#             obj_cache.append(gw_obj(Cx,Cy,Pi))
#             if niter==0:
#                 c_cache.append((lower_bounds,upper_bounds))
#                 time_cache.append(initialization_time)
#             else:
#                 # c_cache.append((A_,b_))
#                 c_cache.append((A_batch, b_batch))
#                 time_cache.append(iteration_time)
        
#         if u_bound - l_bound < epsilon:
#             if verbose:
#                 print(f'{niter:5d}|{(u_bound-l_bound):8e}')
#             break
        
#         if verbose:
#             if niter % 1 == 0:
#                 print(f'{niter:5d}|{(u_bound-l_bound):8e}')
        
#         A_batch = []
#         b_batch = []
#         for _ in range(batch_size):
#             Zn = 2 * Wn.reshape(-1)
#             alphan = 1
#             betan = np.sum(M * Pi)

#             A_batch.append(np.r_[np.array([alphan]), Zn])
#             b_batch.append(betan)

#         # Zn = 2*Wn.reshape(-1)
#         # alphan = 1
        
#         # betan = np.sum(M*Pi)

#         # A_ = np.r_[np.array([alphan]),Zn]
#         # b_ = np.array([betan])

#         A_batch = np.vstack(A_batch)
#         b_batch = np.hstack(b_batch)
        
#         # E, B, D = extreme_points_update(E, B, D, A_, b_)
#         E, B, D = extreme_points_update(E, B, D, A_batch, b_batch)
        
#         end_time = time.time()
#         iteration_time = end_time - start_time
    
#     if u_bound - l_bound > epsilon:
#         print('Warning: algorithm does not converge. Try larget IterMax.')
    
#     if log:
#         cum_time = np.cumsum(time_cache)
#         logs = {'niter':niter+1,'obj_cache':obj_cache,'E_cache':E_cache,'gap_cache':gap_cache,'Pi_cache':Pi_cache,'E0_cache':E0_cache,'l_cache':l_cache,'u_cache':u_cache,'c_cache':c_cache,'time_cache':time_cache,'cum_time':cum_time,'L':L}
#         return Pi,logs
#     elif dist:
#         return Pi, Cx, Cy
#     else:
#         return Pi


# def gw_global_drot(X,Y,epsilon=1e-6,IterMax=100,verbose=False,log=False, dist=False):
#     """
#     Globally solving the Gromov-Wasserstein problem for point clouds in low dimensional Euclidean spaces [1].
    
#     Input:
#     X: ndarray(nx,lx)
#     Y: ndarray(ny,ly)
#     epsilon: Stop threshold on bound gap
#     IterMax: Max number of iterations
#     verbose: Print information along iterations
#     log: Record log if True
    
#     Return:
#     Pi: ndarray(nx,ny) Global optimal solution
#     logs:  log dictionary return only if log==True in parameters
    
#     [1] Ryner M, Kronqvist J, Karlsson J. Globally solving the Gromov-Wasserstein problem for point clouds in low dimensional Euclidean spaces[J]. Advances in Neural Information Processing Systems, 2024, 36.
#     """
#     # X ndarray (nx,lx)
#     # Y ndarray (ny,ly)

#     start_time = time.time()
    
#     X = X.T
#     Y = Y.T
        
#     lx,nx = X.shape
#     ly,ny = Y.shape
#     l_bound = -np.inf
#     u_bound = np.inf
    
#     mx = (np.linalg.norm(X,axis=0)**2).reshape(-1,1)
#     my = (np.linalg.norm(Y,axis=0)**2).reshape(-1,1)
#     vec_1x = np.ones_like(mx)
#     vec_1y = np.ones_like(my)
    
#     L = (nx + ny) * mx @ my.T - 4 * mx @ vec_1y.T @ Y.T @ Y - 4 * X.T @ X @ vec_1x @ my.T

#     Cx = vec_1x @ mx.T - 2 * X.T @ X + mx @ vec_1x.T
#     Cy = vec_1y @ my.T - 2 * Y.T @ Y + my @ vec_1y.T
#     c0 = np.sum(Cx**2) + np.sum(Cy**2) - 2 * np.sum(mx) * np.sum(my)
    
#     W_lower = np.zeros([lx,ly])
#     W_upper = np.zeros([lx,ly])

#     a = np.ones(nx)
#     b = np.ones(ny)
    
#     if verbose:
#         print('Iter |Bound gap'+ '\n' + '-' * 22)
    
#     if log:
#         E_cache = []
#         E0_cache = []
#         gap_cache = []
#         Pi_cache = []
#         l_cache = []
#         u_cache = []
#         c_cache = []
#         time_cache = []
#         obj_cache = []
    
#     stepsize = 2. / sum(L.shape)
#     maxiters = 100000
#     eps = 1e-5

#     # result = fast_ot.drot(C, p, q, C.shape[0], C.shape[1], stepsize, maxiters, eps, False, True)
#     # print(f'Iteration stops at #{len(result.objectives)}')
#     w_lower = fast_ot.drot(L, a, b, L.shape[0], L.shape[1], stepsize, maxiters, eps, False, True)
#     w_upper = -fast_ot.drot(-L, a, b, L.shape[0], L.shape[1], stepsize, maxiters, eps, False, True)

    
#     w_lower,w_upper = ot.emd2(a,b,L), -ot.emd2(a,b,-L)
    
#     for i in range(lx):
#         for j in range(ly):
#             M = 2 * X[i,None].T @ Y[j,None]
#             W_lower[i,j] = ot.emd2(a,b,M)
#             W_upper[i,j] = - ot.emd2(a,b,-M)
    
#     lower_bounds = np.r_[np.array([w_lower]),W_lower.reshape(-1)]
#     upper_bounds = np.r_[np.array([w_upper]),W_upper.reshape(-1)]
    
#     E,B,D = init_extreme_points(lower_bounds,upper_bounds)
    
#     end_time = time.time()
#     initialization_time = end_time - start_time
    
#     for niter in range(IterMax):
#         start_time = time.time()
        
#         wW_cache,l_bound = optimal_extreme_point(E)
#         l_bound = l_bound + c0
#         Wn = wW_cache[1:].reshape(lx,ly) 
        
#         M = (4*X.T @ Wn @ Y + L)
        
#         Pi = ot.emd(a,b,-M)
#         bound = - np.sum((2*X @ Pi @ Y.T)**2) - np.sum(L * Pi) + c0
#         u_bound = np.min([u_bound,bound])

#         if log:
#             E_cache.append(E)
#             gap_cache.append(u_bound - l_bound)
#             l_cache.append(l_bound)
#             u_cache.append(u_bound)
#             Pi_cache.append(Pi)
#             E0_cache.append(wW_cache)
#             obj_cache.append(gw_obj(Cx,Cy,Pi))
#             if niter==0:
#                 c_cache.append((lower_bounds,upper_bounds))
#                 time_cache.append(initialization_time)
#             else:
#                 c_cache.append((A_,b_))
#                 time_cache.append(iteration_time)
        
#         if u_bound - l_bound < epsilon:
#             if verbose:
#                 print(f'{niter:5d}|{(u_bound-l_bound):8e}')
#             break
        
#         if verbose:
#             if niter % 1 == 0:
#                 print(f'{niter:5d}|{(u_bound-l_bound):8e}')

#         Zn = 2*Wn.reshape(-1)
#         alphan = 1
        
#         betan = np.sum(M*Pi)

#         A_ = np.r_[np.array([alphan]),Zn]
#         b_ = np.array([betan])
        
#         E, B, D = extreme_points_update(E, B, D, A_, b_)
        
#         end_time = time.time()
#         iteration_time = end_time - start_time
    
#     if u_bound - l_bound > epsilon:
#         print('Warning: algorithm does not converge. Try larget IterMax.')
    
#     if log:
#         cum_time = np.cumsum(time_cache)
#         logs = {'niter':niter+1,'obj_cache':obj_cache,'E_cache':E_cache,'gap_cache':gap_cache,'Pi_cache':Pi_cache,'E0_cache':E0_cache,'l_cache':l_cache,'u_cache':u_cache,'c_cache':c_cache,'time_cache':time_cache,'cum_time':cum_time,'L':L}
#         return Pi,logs
#     elif dist:
#         return Pi, Cx, Cy
#     else:
#         return Pi