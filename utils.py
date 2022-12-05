import numpy as np
import matplotlib.pyplot as plt

def get_normalization_matrix(x):
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
    print("msd = ", msd)
    
    # Output: T 3x3 transformation matrix of points
    #TRANSFORMATION MATRIX used to normalize the inputs x (constructed following the theoretical knowledge)
    T = np.array([[(np.sqrt(2) / msd), 0, (-centroid[0] * np.sqrt(2) / msd)], #µx
                  [0, (np.sqrt(2) / msd), (-centroid[1] * np.sqrt(2) / msd)], #µy
                  [0, 0, 1]])

    return T #returns the transformation matrix used to normalize the inputs


def eight_points_algorithm(x1, x2, normalize=True):#True by default
    """
    Calculates the fundamental matrix between two views using the normalized 8 point algorithm
    Inputs:
                    x1      3xN     homogeneous coordinates of matched points in view 1
                    x2      3xN     homogeneous coordinates of matched points in view 2
    Outputs:
                    F       3x3     fundamental matrix

    The poor numerical conditioning due tue the gap between quadratic and linear terms, which makes results very sensitive to noise, 
    can be solved by applying the normalized 8-point algorithm that rescales the data in the range [-1,1], following these three steps;
    1. Normalization of the point correspondences; x1_n= T1*x1 , x2_n= T2*x2
    2. Use of normalized coordiantes x1_n and x2_n to estimatermalized F_n with 8-point algorithm -->x2_n^T * F_n * x1_n=0
    3. Compute un-normlaized F form F_n --> x2_n^T= x2^T*T2^T and x1_n= T1*p1 --> F= T2^T*F_n*T1
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
   
    # Each point pair (according to epipolar constraint) contributes only to one scalar equation --> We need at least 8 points, the 9th equation can be derived from the other eight
    # - I construct A as a 9-rows matrix where each row is defined by the constraint over x1 and x2 
    # - I set the `axis` parameter  to 1 to specify that I want to stack the input arrays along the columns
    # - The value of the last column is all 1, since it counts for the elements of the F matrix in the line equation
    
   
    # I obtian a system of 9 equations as the 9 entries of F that can be summarized in matrix form as Af=0,subject to ||f||^2=1.
    # And where A is the point correspondence matrix and vector f is the flattened fundamental matrix.
    # ---> I solve this lienar system by minimizing ||Af||^2 subject to ||f||^2=1
    
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
    # It's obtianed by using the linear algebraic function 'svd' on the numpy libray
    # I consider by default full_matrices=True so for example if A is a matrix of dimensions (MxN), then U and V have the shapes (..., M, M) and (..., N, N), respectively.
    # _, _, V would work as well sincec I need only V (actually I am interested in V.T)
    U, S , V = np.linalg.svd(A, full_matrices=True)

    #F is the last column of V transpose, vector corresponding to the smallest singular value 
    #I am interested in the last column (V.transpose() is like doing V.T)
    # I get a vector that I need to rehsape in a 3x3 matrix, the normalized Functional matrix
    F = V.transpose()[:, 8].reshape(3,3) 
    print("F solved with SVD = ", F) 

    # Enforce that rank(F)=2 --> verify that the fundamental matrix F has rank 2
    # I enforce rank 2 by zeroing out the last singular value
    U, S, V = np.linalg.svd(F, full_matrices=True) 

    # I am zeroing down the last singualr value of S (position 2--> 3rd element)
    S[2] = 0 

    #I restore F as the matrix product of the matrices obtained with the singular value decomposition:
    F_n = U @ np.diag(S) @ V
    print("F_n =", F_n)

    if normalize:
        #Transform F back --> I need to go from F_n back to F, by applying:  F= T2^T*F_n*T1
        T2_t= T2.transpose()
        # Matrix product
        F = T2_t @ F_n @ T1
    return F

    
    """ FROM THEORY WE KNOW THAT
    The eight point algorithm follows the epipolar constraint for either the calibrated or uncalibrated case.
    The calibrated case is applied when K and K', the intrinsics of the two cameras, are known: (xʹ)dot[(t)cross(Rx)]=0 
   
    For example for the Essential matrix E:
    in matrix formulaiton I have [tx]*R = E, which is the definition of the essential matrix E that gives us the 
    relative rotation and translation between the cameras, or their extrinsic parameters
    Where [Tx]= [0 -tz ty; tz 0 -tx; -ty tx 0]
     --> in this way I have that: (x'^T) * E * x=0 ---> EPIPOLAR CONSTRAINT on matrix E
     --> the cross product returns a vector perpendicular to both (t) and (Rx) --> Infact I have that vectors Rx, t and ' are coplanar
    Ex is the epipolar line l'=Ex associated with x
    E^Tx' is the epipoalr line l= E^Tx' associated with x'
    E has rank 2 and 5 doF

    The 8-point algorithm assumes that all entries are independent from one another.
    Matrix E has 9 unknown entries but one can be derived from the other 8.
    For 'n' points is possible to write an homogeneous system of n equations 
    --> in this case to derive non trivial solutions for QE=0 -->(x'^T) * E * x=0 (Q same role as A for F)
    I need at least 8 point correspondences, each one providing one independent equation
    PLEASE_NOTE --> If I know the calibration matrices of the two cameras, I can estimate the essential matrix: E = K’^T*F*K, inverse approach


    For Uncalibrated cases, so when matrices K and K' intrinsics of the camera are unknown --> the epipolar
    constraint can be rewritten in terms of unknown normalized coordinates --> in this way the fundamental matrix F is found
    (x_n'^T) * E * x_n=0 --> where x_n= (K^-1) * x ;  and x_n'= (K'^-1) * x'
    so I get that:
    (x')^T *(K'^-1)^T * E * (K^-1) * x =0 
    Which can be rewritten as:
    (x')^T * F * x =0
    where F = (K^-1')^T * E * (K^-1) is the fundamental matrix that works directly in pixel coordinate
    Fx is the epipolar line l'=Fx associated with x
    F^Tx' is the epipoalr line l= F^Tx' associated with x'
    F is singualr with rank 2 and 7doF

    The 8-point algorithm applied to F minimizes the error:
    (sum of i over N of ((xi')^T * F * xi)**2), under the constraint ||F||^2 =1
    This error can be interpreted as:
    - the sum of squared algebraic distances between points xi' and epipolar lines F*xi (or points xi and epipolar lines F^T*xi')
    - Non linear approach: minimize sum of squared geometric distances

    """


def right_epipole(F):
    """
    Computes the (right) epipole from a fundamental matrix F
    To find the right epipole I consider the epipolar lines computed from the fundamental matrix F (remind that Fx is the epipolar line l'=Fx associated with x)
    So, I compute the linear least square estimate of F which is its last singular value.

    FROM THEORY:
    •  Epipoles = intersections of baseline with image planes // projections of the optical center on the other camera image  // vanishing points of the baseline (motion direction)
       --> epipole is a point where all the epipolar lines in an image meet. 
       --> Mathematically it can be represented as: l1^T*e=0, l2^T*e=0,... where l1, l2 are the epipolar lines and e the epipole
       PLEASE_NOTE --> a stereo camera has two epipoles, and all epipolar lines intersect at the epipole.
    •  Epipolar Lines =intersections of epipolar plane with image planes (always come in corresponding pairs)
    
    This function is used to calculate the right_epipole for a given fundamental matrix and corner point correspondences in the two images.
    To find the epipole for the other image, I find the linear least square estimate of F transpose --> I pass F.T to find the left epiopole. ALternatively I could have passed F and a type argument set either to "right" or "left to define the two cases. 
    """
    
    # The epipole is the null space of F (F * e = 0) --> linear least square estimate of F
    _, _, V = np.linalg.svd(F)
    e = V[-1, :] # e = V[-1] -> addresses the last row of V all the columns
    e = e / e[2] # the peipole must be normalized
    return e



def plot_epipolar_line(im, F, x, e): #ax=None,
    """
    Plot the epipole and epipolar line F*x=0 in an image. 
    F is the fundamental matrix, passed transpose and x a point in the other image.
    """

    #Find the min and max value of the image, the limits within which I consider the points (could also use range instead to find max-min value)
    m, n = im.shape[:2] 

    # Find l1, the epipolar line associated with x ---> I could also do l1= np.dot(F, x)
    l1= F @ x

    # I need an evenly spaced number of samples (150 e.g.) over the interval defined by [0,n] image space -> or I could also use min, max as limits
    samp= np.linspace(0, n, 150)

    # I find the values of the line to be plot --> simpler by using list comprehension
    val = np.array([(l1[2] + l1[0] * s) / (-l1[1]) for s in samp])   

    # I need to limit the points by considering the image dimension m --> i want only the points inside the image
    p_in = (val >= 0) & (val < m)

    #I plot the epipolar line
    plt.plot(samp[p_in], val[p_in], linewidth=2)

    #In case I want to plot the epipole, I would have to pass another argument "show_epipole", true by default
    # if show_epipole is True:
    #     plt.plot(e[0] / e[2], e[1] / e[2], 'bx' ) #plot blue x-es where there are the epipolar points



