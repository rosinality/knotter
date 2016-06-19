import numpy as np

def uniform_cover(points, N, overlap=.5):
    minimum = points.min()
    maximum = points.max()
    
    interval = (maximum - minimum) / (N - (N - 1) * overlap)
    step = interval * (1 - overlap)
    intervals = []
    
    m = np.full(N, float(minimum))
    m += step * np.arange(N)
    M = m + interval
    
    return np.dstack((m, M))[0]

def balanced_cover(points, N, overlap=.5):
    sorted_points = np.sort(points)
    length = sorted_points.shape[0]
    interval = int(length / (N - (N - 1) * overlap))
    step = int(interval * overlap)
    m = np.empty(N)
    M = np.empty(N)

    for i in range(N):
        start = i * (interval - step)
        
        # If this is last step in the loop, include all the rest points.
        if i == N - 1:
            subset = sorted_points[start:]
        else:
            subset = sorted_points[start:start + interval]
            
        m[i] = subset.min()
        
        # Get next point of the max point in the interval.
        try:
            next_point = sorted_points[start + interval]
            
        except:
            next_point = 0.0

        subset_max = subset.max()
        
        # Add difference between max and next element for
        # include this point in the m <= points < M condition.
        # See point_in_interval function.
        M[i] = subset_max + (next_point - subset_max) / 2

    return np.dstack((m, M))[0]

