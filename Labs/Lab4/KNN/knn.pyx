import numpy as np
cimport cython

# Define the data types for arrays
DTYPE = np.int_
DTYPE2 = np.double

@cython.boundscheck(False)  # Deactivate bounds checking for performance
@cython.wraparound(False)    # Deactivate negative indexing for performance
cpdef double[:] knn_cython(double[:, :] x, Py_ssize_t K, double[:, :] data_train, long[:] class_train):
    """
    KNN classifier.

    Numpy implementation of a K-nearest neighbours (KNN) classifier. Assumes
    samples are classified as row vectors.

    Args:
    x (array): input features to be classified (each sample corresponds to
               a row).
    K (int): number of neighbours for the KNN classification.
    data_train (array): training data (each sample corresponds to a row).
    class_train (array of int): class associated with each example in the
                                 training data.

    Returns:
    class_pred (array): predicted class for each input vector x[q,:].
    """

    # Check input dimensions
    assert data_train.shape[0] == class_train.shape[0]
    assert x.shape[1] == data_train.shape[1]

    # Initialize the prediction array
    cdef long[:] class_pred = np.zeros(x.shape[0], dtype=DTYPE)

    # Preallocate distance and labels arrays
    cdef Py_ssize_t n_test = x.shape[0]
    cdef Py_ssize_t n_train = data_train.shape[0]
    cdef double[:] distance = np.zeros(n_train, dtype=DTYPE2)
    cdef long[:] labels = np.zeros(K, dtype=DTYPE)

    cdef Py_ssize_t i, j, k

    # Iterate through each test sample
    for i in range(n_test):
        # Compute distances from the current test point to all training points
        for j in range(n_train):
            distance[j] = 0.0
            for k in range(x.shape[1]):
                distance[j] += (data_train[j, k] - x[i, k]) ** 2
            distance[j] = np.sqrt(distance[j])  # Taking the square root to get actual distance

    return distance
