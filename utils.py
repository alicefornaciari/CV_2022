import numpy as np
import matplotlib.pyplot as plt

def get_normalization_matrix(x):
    """
    - get_normalization_matrix Returns the transformation matrix used to normalize the inputs x
    -  Normalization corresponds to subtracting mean-position and positions have a mean distance of sqrt(2) to the center    
    """
    # Input: x 3*N  --> I want the first three rows and all columns of the inputs 2D points
    #I consider x as general input, that the first time called is is x1 and the second is x2
    x_2D = x[:2,:] 

    # Get centroid and mean-distance to centroid
    #mean --> µ = (µx, µy) has two dimensions--> mean over the columns (axis=1)
    """ I want to compute the mean along the 1 axis, and I set keepdims=True so the axes which are reduced are left
        in the result as dimensions with size one --> so the result will broadcast correctly against the input array.
    """
    centroid= np.mean(x_2D, axis= 1, keepdims=True) 
      
    #mean standard deviation --> The standard deviation is the square root of the average of the squared deviations from the mean
    #I perform the mean standard deviation along axis 0 --> rows
    distance = x_2D - centroid
    msd=np.mean(np.sqrt(np.sum((distance) ** 2, axis=0)))
    centroid = centroid.flatten()
    print("msd = ", msd)
    
    # Output: T 3x3 transformation matrix of points
    #TRANSFORMATION MATRIX used to normalize the inputs x
    T = np.array([[(np.sqrt(2) / msd), 0, (-centroid[0] * np.sqrt(2) / msd)], #µx
                  [0, (np.sqrt(2) / msd), (-centroid[1] * np.sqrt(2) / msd)], #µy
                  [0, 0, 1]])

    return T #returns the transformation matrix used to normalize the inputs


def eight_points_algorithm(x1, x2, normalize=True):
    """
    Calculates the fundamental matrix between two views using the normalized 8 point algorithm
    --> it's called as: F = eight_points_algorithm(x1, x2)
    Inputs:
                    x1      3xN     homogeneous coordinates of matched points in view 1
                    x2      3xN     homogeneous coordinates of matched points in view 2
    Outputs:
                    F       3x3     fundamental matrix
    """

    N = x1.shape[1] #I look at the second dimension of x1
    #print("N=", N)

    if normalize: 
        #CALL the funciton get_normalization_matrix(x) that I have used aboe
        # Construct transformation matrices to normalize the coordinates
        T1= get_normalization_matrix(x1)
        T2= get_normalization_matrix(x2)

        # Normalize inputs --> matrix multiplication of T(3x3) @ x (3xN)
        x1_n= T1 @ x1 #x1 normalized
        x2_n= T2 @ x2 #x2 normalized

      
    """ Epipolar constraint calibrated case, when Kand K' of the two cameras are known: (xʹ)dot[(t)cross(Rx)]=0 
    In matrix formulaiton I have [tx]*R = E, which is the definition of the essential matrix E that gives us the 
    relative rotation and translation between the cameras, or their extrinsic parameters
    Where [Tx]= [0 -tz ty; tz 0 -tx; -ty tx 0]
     --> in this way I have that: (x'^T) * E * x=0 ---> EPIPOLAR CONSTRAINT

     --> dot as inner product that gives 0 when the two vectors are orthogonal as in this case, since x' 
        belongs to the epipolar plane, it's always perpendicular to the norm of the plane, defined by the cross product
     --> the cross product returns a vector perpendicular to both (t) and (Rx) --> Infact I have that
        vectors Rx, t and ' are coplanar
    Ex is the epipolar line l'=Ex associated with x
    E^Tx' is the epipoalr line l= E^Tx' associated with x'
    E has rank 2 and 5 doF

    The 8-point algorithm assumes that all entries are independent from one another.
    Matrix E has 9 unknown entries but one can be derived from the other 8.
    For 'n' points is possible to write an homogeneous system of n equations 
    --> in this case to derive non trivial solutions for QE=0 -->(x'^T) * E * x=0 (Q same role as A for F)
    I need at least 8 point correspondences, each one providing one independent equation
    """

    # Construct matrix A encoding the constraints on x1 and x2 for matrix F --> homogenous system with  9 unknowns
    # Each point pair (according to epipolar constraint) contributes only to one scalar equation --> We need at least 8 points, the 9th equation can be derived from the other eight

    #I construct A as a 9-rows matrix where each row is defined by the constraint over x1 and x2 
    # I set the `axis` parameter  to 1 to specify that I want to stack the input arrays along the columns
    # the value of the last column is all 1, since it counts for the elements of the F matrix in the line equation
    
   
    # I obtian a system of 9 equations as the 9 entries of F that can be summarized in matrix form as Af=0, where ||f||^2=1
    #I solve this lienar system by minimizing ||Af||^2 subject to ||f||^2=1
    #This equation can be represented in matrix notation as Af= 0, where A is the point correspondence matrix and vector f is the flattened fundamental matrix.

    A = np.stack((x2_n[0, :] * x1_n[0, :],
            x2_n[0, :] * x1_n[1, :],
            x2_n[0, :],
            x2_n[1, :] * x1_n[0, :],
            x2_n[1, :] * x1_n[1, :],
            x2_n[1, :],
            x1_n[0, :],
            x1_n[1, :],
            np.ones((N,))), axis= 1)
    
    # it returns a stacked array with one more dimension than the input arrays

    """
    For Uncalibrated cases, so when matrices K and K' intrinsics of the camera are unknown --> the epipolar
    constraint can be rewritten in terms of unknown normalized coordinates --> the fundamental matrix F is found
    (x_n'^T) * E * x_n=0 --> where x_n= (K^-1) * x ;  and x_n'= (K'^-1) * x'
    so I get that:
    (x')^T *(K'^-1)^T * E * (K^-1) * x =0 
    Which can be rewritten as:
    (x')^T * F * x =0
    where F = (K^-1')^T * E * (K^-1) is the fundamental matrix that works directly in pixel coordinate
    Fx is the epipolar line l'=Fx associated with x
    F^Tx' is the epipoalr line l= F^Tx' associated with x'
    F is singualr with rank 2 and 7doF

    PLEASE_NOTE --> If I know the calibration matrices of the two cameras, I can estimate the essential matrix: E = K’^T*F*K, inverse approach

    The 8-point algorithm minimizes the error:
    (sum of i over N of ((xi')^T * F * xi)**2), under the constraint ||F||^2 =1
    This error can be interpreted as:
    - the sum of squared algebraic distances between points xi' and epipolar lines F*xi (or points xi and epipolar lines F^T*xi')
    - Nonlinear approach: minimize sum of squared geometric distances

    --> the poor numerical conditioning due tue the gap between quadratic and linear terms, that makes results very sensitive to noise, 
    can be solved by applying the normalized 8-point algorithm that rescales the data in the range [-1,1], following these three steps;
    1. Normalization of the point correspondences; p1_n= T1*p1 , p2_n= T2*p2
    2. Use of normalized coordiantes '1_n and p2_n to estimatermalized F_n with 8-point algorithm -->p2_n^T * F_n * p1_n=0
    3. Compute un-normlaized F form F_n --> p2_n^T= p2^T*T2^T and p1_n= T1*p1 --> F= T2^T*F_n*T1

    """

    # Solve for f using SVD --> application of singualr value decomposition A=USV^t
    # It's obtianed by using the linear algebra function 'svd' on the numpy libray
    # A= matrix of dimensions (MxN) 
    # I consider by default full_matrices=True so , U and V have the shapes (..., M, M) and (..., N, N), respectively.
    _, _ , V = np.linalg.svd(A, full_matrices=True)
    # print("V", V)

    #what matters for me is V^T
    #V_t= V.transpose()

    #F is the last column of V, vector corresponding to the smallest singular value
    F_n = V.transpose()[:, 8].reshape(3,3) #column 9, but since I count from 0 the last one is 8--> V.transpose()[:, -1].reshape(3,3)
    # Iget a vector that I need to rehsape in a 3x3 matrix, the normalized Functional matrix
    print("F normalized = ", F_n) 

    # Enforce that rank(F)=2 --> verify that the fundamental matrix F has rank 2
    # I enforce rank 2 by zeroing out last singular value
    U, S, V = np.linalg.svd(F_n, full_matrices=True)  #???????????dire cosa sono S V U
    S[2] = 0 # I am zeroing down the last singualr value of S (position 2--> 3rd element)
    #I restore F as the matrix product of the matrices obtained with the singular value decomposition:
    F_n = U @ np.diag(S) @ V
    print("F_n =", F_n)

    if normalize:
        # Transform F back --> I need to go from F_n back to F, by applying:  F= T2^T*F_n*T1
        T2_t= T2.transpose()
        #matrix product
        F = T2_t @ F_n @ T1
    return F


def right_epipole(F, type="right"):
    """
    Computes the (right) epipole from a fundamental matrix F
    To find the right epipole I consider the epipolar lines computed from the fundamental matrix F (remind that Fx is the epipolar line l'=Fx associated with x)
    SO, I compute the linear least square estimate of F which is its last singular value.

    --> epipole is a point where all the epipolar lines in an image meet. 
    --> Mathematically it can be represented as: l1^T*e=0, l2^T*e=0,... where l1, l2 are the epipolar lines and e the epipole
    •  Epipoles = intersections of baseline with image planes // projections of the optical center on the other camera image  // vanishing points of the baseline (motion direction)
    •  Epipolar Lines =intersections of epipolar plane with image planes (always come in corresponding pairs)
    PLEASE_NOTE --> a stereo camera has two epipoles, and all epipolar lines intersect at the epipole.
    
    To find the epipole for the other image, I find the linear least square estimate of F transpose.
    (Use with F.T for left epipole --> to compute the other epipole)

    The function compute_epipole is used to calculate the epipoles for a given fundamental matrix and corner point correspondences in the two images.

    """
    if type == "right": #default value is used for the right epipole
        # The epipole is the null space of F (F * e = 0) --> linear least square estimate of F
        _, _, V = np.linalg.svd(F)
    elif type== "left":#1 is used for the left epipole  
        _, _, V = np.linalg.svd(F.T)   

    e = V[-1, :] # e = V[-1] -> addresses the last row of V all the columns
    # TODO check what it does 
    e = e / e[2]
    return e

# def left_epipole(F):
#     """
#     Computes the (right) epipole from a fundamental matrix F.
#     (Use with F.T for left epipole --> to compute the other epipole)
#     When the function is called, it's pass in F.T
#     """

#     # The epipole is the null space of F (F * e = 0)
#     _, _, V = np.linalg.svd(F)
#     e = V[-1, :] # e = V[-1] --> addresses the last row of V all the columns
#     e = e / e[2]
#     print ("Left epipole =", e)

#     return e

def plot_epipolar_line(im, F, x, e, show_epipole=True): #ax=None,
    """
    Plot the epipole and epipolar line F*x=0 in an image. 
    F is the fundamental matrix, passed transpose
    and x a point in the other image.
    """

    m, n = im.shape[:2] #These are the min and max value, that set the limits within which I consider the points
    #range = np.ptp(x, 1) * 0.01 #returns the range of x (max-min) over axis=1, so returns values subtracting max and min for each row
    # line = np.dot(F, x)  ---> same as F @ x 
    # l1= Fx is the epipolar line associated with x
    l1= F @ x

    # I need an evenly spaced number of samples over the interval defined by [0,n] image space -> or find max e min and use themas limit
    samp= np.linspace(0, n, 150)
    val = np.array([(l1[2] + l1[0] * s) / (-l1[1]) for s in samp])   #values of the line to be plot --> simpler if I use list comprehension

    # I need to limit the points I am taking by considering the image dimension m
    # --> i want only the points inside the image
    p_in = (val >= 0) & (val < m)
    plt.plot(samp[p_in], val[p_in], linewidth=1)
    # if ax is None:
    #     ax = plt.plot(samp[p_in], val[p_in], linewidth=1) #samp[p_in],
    #    # plt.plot(x[0], x[1], 'ro')

    if show_epipole is True:
        plt.plot(e[0] / e[2], e[1] / e[2], 'bx') #plot blue x-es where there are the epipolar points



