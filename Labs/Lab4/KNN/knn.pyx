import numpy as np
cimport numpy as np


# We now need to fix a datatype for our arrays. I've used the variable
# DTYPE for this, which is assigned to the usual NumPy runtime
# type info object.
DTYPE = np.intc


# cdef means here that this function is a plain C function (so faster).
# To get all the benefits, we type the arguments and the return value.


cdef distance(double[:,:] x_train_view,Py_ssize_t N_train, double[:,:] x_test_view,Py_ssize_t i):

    cdef Py_ssize_t j, d
    cdef double dist, diff
    distances = np.zeros(N_train, dtype=float)
    cdef double[:] distances_view = distances

    for j in range(N_train):
        dist = 0
        for d in range(x_train_view.shape[1]):
            diff = x_train_view[j, d] - x_test_view[i, d]
            dist += diff * diff
        distances_view[j] = dist
    return distances_view


def knn(double[:,:] x_train, int[:] class_train, double[:,:] x_test, Py_ssize_t k):

    cdef Py_ssize_t N_test = x_test.shape[0]
    cdef Py_ssize_t N_train = x_train.shape[0]
    cdef Py_ssize_t i,j,d, neighbor
    cdef double dist, diff
    cdef int[:] class_test = np.zeros(N_test, dtype=int)
    cdef double[:,:] x_train_view = x_train
    cdef double[:,:] x_test_view = x_test
    # define sorted_indices
    cdef Py_ssize_t[:] sorted_indices

    # Temporary array to store distances
    cdef double[:] distances
    cdef int[:] nearest_classes = np.zeros(k, dtype=int)

    for i in range(N_test):
        distances = distance(x_train_view, N_train, x_test_view, i)

        # Find the 'k' smallest distances and corresponding classes
        sorted_indices = np.argsort(distances)[:k]
        for j in range(k):
            nearest_classes[j] = class_train[sorted_indices[j]]

        class_test[i] = np.argmax(np.bincount(nearest_classes))

    return np.asarray(class_test)
