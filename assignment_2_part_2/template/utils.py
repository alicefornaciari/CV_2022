import numpy as np
import scipy
import math
import scipy.sparse as sparse
from scipy.sparse.linalg import lsmr
from collections import defaultdict


def get_normalization_matrix(x): #from part 1
    """
    - get_normalization_matrix Returns the transformation matrix used to normalize the inputs x
    -  Normalization corresponds to subtracting mean-position and positions have a mean distance of sqrt(2) to the center    
    """
    # Input: x 3*N  --> I want the first three rows and all columns of the inputs 2D points
    #I consider x as general input, that the first time called is x1 and the second is x2
    x_2D = x[:2,:] #2D point

    # Get centroid and mean-distance to centroid (as studied in theory)
    #mean --> µ = (µx, µy) has two dimensions--> I compute mean over the columns (axis=1) and I set keepdims=True so the axes which are reduced are left in the result as dimensions with size one 
    # In this way the result will broadcast correctly against the input array. 
    centroid= np.mean(x_2D, axis= 1, keepdims=True) 
      
    #mean standard deviation --> The standard deviation is the square root of the average of the squared deviations from the mean
    #I perform the mean standard deviation along axis 0 (rows)
    distance = x_2D - centroid
    msd=np.mean(np.sqrt(np.sum((distance) ** 2, axis=0)))
    centroid = centroid.flatten() #I need to do this to have the right dimensions for the T matrix
    #print("msd = ", msd)
    
    # Output: T 3x3 transformation matrix of points
    #TRANSFORMATION MATRIX used to normalize the inputs x (constructed following the theoretical knowledge)
    T = np.array([[(np.sqrt(2) / msd), 0, (-centroid[0] * np.sqrt(2) / msd)], #µx
                  [0, (np.sqrt(2) / msd), (-centroid[1] * np.sqrt(2) / msd)], #µy
                  [0, 0, 1]])

    return T #returns the transformation matrix used to normalize the inputs


def eight_points_algorithm(x1, x2, normalize=True):#True by default --> also form part 1
    """
    Calculates the fundamental matrix between two views using the normalized 8 point algorithm
    Inputs:
                    x1      3xN     homogeneous coordinates of matched points in view 1
                    x2      3xN     homogeneous coordinates of matched points in view 2
    Outputs:
                    F       3x3     fundamental matrix
    """
    N = x1.shape[1] #I look at the second dimension of x1

    if normalize: 
        #Call the funciton get_normalization_matrix(x) to obtain the matrices T1 and T2 to normalize the coordinates
        T1= get_normalization_matrix(x1)
        T2= get_normalization_matrix(x2)

        # Normalize the inputs --> matrix multiplication between T(3x3) @ x (3xN)
        x1_n= T1 @ x1 #x1 normalized
        x2_n= T2 @ x2 #x2 normalized

    # Construct matrix A encoding the constraints on x1 and x2 for matrix F --> homogenous system with  9 unknowns
    A = np.stack((x2_n[0, :] * x1_n[0, :],
                  x2_n[0, :] * x1_n[1, :],
                  x2_n[0, :],
                  x2_n[1, :] * x1_n[0, :],
                  x2_n[1, :] * x1_n[1, :],
                  x2_n[1, :],
                  x1_n[0, :],
                  x1_n[1, :],
                  np.ones((N,))), axis= 1)
    
    # Solve for f using SVD --> application of the singualr value decomposition A=USV^t
    U, S , V = np.linalg.svd(A, full_matrices=True)

    #F is the last column of V transpose, vector corresponding to the smallest singular value 
    F = V.transpose()[:, 8].reshape(3,3) 

    # Enforce that rank(F)=2 --> verify that the fundamental matrix F has rank 2
    # I enforce rank 2 by zeroing out the last singular value
    U, S, V = np.linalg.svd(F, full_matrices=True) 
    # I am zeroing down the last singualr value of S (position 2--> 3rd element)
    S[2] = 0 

    #I restore F as the matrix product of the matrices obtained with the singular value decomposition:
    F_n = U @ np.diag(S) @ V

    if normalize:
        #Transform F back --> I need to go from F_n back to F, by applying:  F= T2^T*F_n*T1
        T2_t= T2.transpose()
        # Matrix product
        F = T2_t @ F_n @ T1
    return F

#Function used to compute the distance of the rest of the input points to the sampled ones, hence from the model
def reprojection_error(F_sampled,x1,x2):
    #find an intercept of a normal from the given point to the model
    x1_transf = F_sampled @ x1
    x2_transf = F_sampled.T @ x2 #x2.T @ F_sampled
    #calculation of the distance (error)
    err = (np.linalg.norm(x1 - x1_transf, axis=0) ** 2)+ (np.linalg.norm(x2 - x2_transf, axis=0) ** 2)
    #normalization of the error
    err = err /(np.linalg.norm(err))
    return err

#Function used to compute the distance of the rest of the input points to the sampled ones, hence from the model
def reprojection_error(F_sampled,x1,x2):
    #find an intercept of a normal from the given point to the model
    x1_transf = F_sampled @ x1
    x2_transf = F_sampled.T @ x2 #x2.T @ F_sampled
    #calculation of the distance (error)
    err = (np.linalg.norm(x1 - x1_transf, axis=0) ** 2)+ (np.linalg.norm(x2 - x2_transf, axis=0) ** 2)
    #normalization of the error
    err = err /(np.linalg.norm(err))
    return err


def ransac(x1, x2, threshold, num_steps=1000, random_seed=42):
    """
    Output:
    - F: best fundamental matrix
    - inliers: record the best number of inliers so far at each iteration
    """
    N = x1.shape[1]
    if random_seed is not None:
        np.random.seed(random_seed)  # we are using a random seed to make the results reproducible
    
    #initialization of the variables to count the inliers
    max_inlier_count = 0
    num_inlier_count = 0

    #probabilities/ratio of inliers and outlier
    prob =0.99
    e_out = 0.5 
    #min number of points needed to fit the model (8-point algorithm)
   # min_num =8 
    num_samp =8
    num_steps = int(round(np.log(1 - prob) / np.log(1- (1 - e_out)**num_samp)))

    #iterative approach  
    for _ in range(num_steps):
        #definition of the number of smaples needed for the ransac approach, randomly generated within a range from 0 and N
        #num_samp = int(round(np.log(1 - prob) / np.log(1- (1 - e_out)**min_num)))  #should give 1177 number of sampling points with min_num=8
        s = np.random.randint(low=0,high=N,size=num_samp) 
        #find the new x1 and x2 sampled points
        x1_sampled= x1[..., s]
        x2_sampled= x2[..., s]

        # Fit the model F to these sampled points
        F_sampled = eight_points_algorithm(x1_sampled, x2_sampled)
        #calculation of the distance from the sampled points to the rest of the input points
        distance = reprojection_error(F_sampled, x1,x2) 

        #creation of a dictionary to collect the actual inliers for both the given inputs, considered when distance<threshold
        inliers_couple = defaultdict(lambda: None)
        x1_inl = x1[..., distance < threshold]
        x2_inl = x2[..., distance < threshold]
        
        assert len(x1_inl) == len(x2_inl) 
        #no error is retured so we have the same number of inliers, which are also the same given that the intrinsics of the camera are the same (two identical cameras)

        #update number of inliers and definition of best ones wrt the max number of inliers overall 
        num_inlier_count = len(x1_inl)
        if num_inlier_count > max_inlier_count:
            #update of the dictionary (same values for x1 and x2)
            inliers_couple["x1"] = x1_inl
            inliers_couple["x2"] = x2_inl
            max_inlier_count = num_inlier_count

            #estimate F with all the inliers --> reapply 8-point alg
            F = eight_points_algorithm(inliers_couple["x1"], inliers_couple["x2"])
            inliers = inliers_couple["x2"] # right points
           
    return F, inliers 


#Computation of the essential matrix E using the known intrinsics of the cameras (K1=K2=K)
def get_essential_matrix(F, K1, K2):
    #E = K2.T.dot(F).dot(K1) alternatively
    E = K2.T @ F @ K1
    return E


# Decomposition of the essential matrix to obtain the posisble rotations and translations
def decompose_essential_matrix(E, x1, x2):
    """
    Decomposes E into a rotation and translation matrix using the normalized corresponding points x1 and x2.
    """
    # Fix left camera-matrix
    Rl = np.eye(3)
    tl = np.array([[0, 0, 0]]).T
    Pl = np.concatenate((Rl, tl), axis=1)

    #Apply singualr value decomposition to E
    U, S, V = np.linalg.svd(E, full_matrices=True) 
    #Enforcing rank-2 constraint:I am zeroing down the last singualr value of S
    S[2] = 0 
   # W = np.array([0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0]).reshape(3, 3)
    W = np.array([[0.0, -1.0, 0.0], 
                  [1.0, 0.0, 0.0],
                  [0.0, 0.0, 1.0]])


    #Translations
    t1 = U[:, 2] 
    if (np.linalg.norm(t1)) != 0:
        t1 = t1/(np.linalg.norm(t1))
    t1 = t1[..., np.newaxis]
    t2= - t1

    #Rotations
    R1 = U * W * V
    R2 = U * W.T * V
    # I want to be sure that the returned rotation matrices are valid, hence with det=1 and not det= -1 --> if <0 then I invert the sign of the matrix
    det1 = scipy.linalg.det(R1)
    det2 = scipy.linalg.det(R2)
    if det1 < 0: 
        R1 = -R1  
    if det2 < 0:
        R2 = -R2
    



    # Four possibilities
    Pr = [np.concatenate((R1, t1), axis=1),
          np.concatenate((R1, t2), axis=1),
          np.concatenate((R2, t1), axis=1),
          np.concatenate((R2, t2), axis=1)]

    # Compute reconstructions for all possible right camera-matrices
    X3Ds = [infer_3d(x1[:, 0:1], x2[:, 0:1], Pl, x) for x in Pr]

    # Compute projections on image-planes and find when both cameras see point
    test = [np.prod(np.hstack((Pl @ np.vstack((X3Ds[i], [[1]])), Pr[i] @ np.vstack((X3Ds[i], [[1]])))) > 0, 1) for i in range(4)]
    test = np.array(test)
    idx = np.where(np.hstack((test[0, 2], test[1, 2], test[2, 2], test[3, 2])) > 0.)[0][0]

    # Choose correct matrix
    Pr = Pr[idx]

    return Pl, Pr


# Function used to reconstruct the 3D points using the obtained rotation and translation
def infer_3d(x1, x2, Pl, Pr):
    # INFER3D Infers 3d-positions of the point-correspondences x1 and x2, using
    # the rotation matrices Rl, Rr and translation vectors tl, tr. Using a
    # least-squares approach.

    M = x1.shape[1]
    # Extract rotation and translation
    Rl = Pl[:3, :3]
    tl = Pl[:3, 3]
    Rr = Pr[:3, :3]
    tr = Pr[:3, 3]

    # Construct matrix A with constraints on 3d points
    row_idx = np.tile(np.arange(4 * M), (3, 1)).T.reshape(-1)
    col_idx = np.tile(np.arange(3 * M), (1, 4)).reshape(-1)

    A = np.zeros((4 * M, 3))
    A[:M, :3] = x1[0:1, :].T @ Rl[2:3, :] - np.tile(Rl[0:1, :], (M, 1))
    A[M:2 * M, :3] = x1[1:2, :].T @ Rl[2:3, :] - np.tile(Rl[1:2, :], (M, 1))
    A[2 * M:3 * M, :3] = x2[0:1, :].T @ Rr[2:3, :] - np.tile(Rr[0:1, :], (M, 1))
    A[3 * M:4 * M, :3] = x2[1:2, :].T @ Rr[2:3, :] - np.tile(Rr[1:2, :], (M, 1))

    A = sparse.csr_matrix((A.reshape(-1), (row_idx, col_idx)), shape=(4 * M, 3 * M))

    # Construct vector b
    b = np.zeros((4 * M, 1))
    b[:M] = np.tile(tl[0], (M, 1)) - x1[0:1, :].T * tl[2]
    b[M:2 * M] = np.tile(tl[1], (M, 1)) - x1[1:2, :].T * tl[2]
    b[2 * M:3 * M] = np.tile(tr[0], (M, 1)) - x2[0:1, :].T * tr[2]
    b[3 * M:4 * M] = np.tile(tr[1], (M, 1)) - x2[1:2, :].T * tr[2]

    # Solve for 3d-positions in a least-squares way
    w = lsmr(A, b)[0]
    x3d = w.reshape(M, 3).T

    return x3d

# Calculates rotation matrix to euler angles

def rotationMatrixToEulerAngles(R) :
    sy = math.sqrt(R[0,0] * R[0,0] +  R[1,0] * R[1,0])
 
    singular = sy < 1e-6
 
    if  not singular :
        x = math.atan2(R[2,1] , R[2,2])
        y = math.atan2(-R[2,0], sy)
        z = math.atan2(R[1,0], R[0,0])
    else :
        x = math.atan2(-R[1,2], R[1,1])
        y = math.atan2(-R[2,0], sy)
        z = 0
 
    return np.array([x, y, z])