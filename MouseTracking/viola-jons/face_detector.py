import cv2
import numpy as np
from scipy.ndimage import label

# step size for sliding window in the image of size N*M
step_size = 4
positive_pretection = 1
negative_pretection = 0
def integral_image(I):
    """
    Computes the integral image of an input image I.
    
    Parameters:
    I (numpy.ndarray): An image of size N x M.
    
    Returns:
    numpy.ndarray: The integral image of the same size.
    """
    N, M = I.shape
    II = np.zeros((N, M))
    
    # Set II(1, 1) = I(1, 1)
    II[0, 0] = I[0, 0]
    
    # Compute the first row
    for j in range(1, M):
        II[0, j] = I[0, j] + II[0, j-1]
    
    # Compute the first column
    for i in range(1, N):
        II[i, 0] = I[i, 0] + II[i-1, 0]
    
    # Compute the rest of the integral image
    for i in range(1, N):
        for j in range(1, M):
            II[i, j] = I[i, j] + II[i, j-1] + II[i-1, j] - II[i-1, j-1]
    
    return II

def sum_region(ii, x1, y1, x2, y2):
    """
    Computes the sum of the region from (x1, y1) to (x2, y2) using the integral image.
    """
    A = ii[x1-1, y1-1] if x1 > 0 and y1 > 0 else 0
    B = ii[x1-1, y2] if x1 > 0 else 0
    C = ii[x2, y1-1] if y1 > 0 else 0
    D = ii[x2, y2]
    return D - B - C + A

def compute_haar_like_features(img,II):
    """
    Computes Haar-like features for a 24x24 image.
    
    Parameters:
    img (numpy.ndarray): A 24x24 image with zero mean and unit variance.
    
    Returns:
    numpy.ndarray: A feature vector of Haar-like features.
    """
    assert img.shape == (24, 24), "Input image must be 24x24"
    
    features = []
    f = 0
    
    # Feature type (a)
    for i in range(1, 25):  # Range from 1 to 24 inclusive
        for j in range(1, 25):
            for w in range(1, (25 - j) // 2 + 1):
                for h in range(1, 25 - i + 1):
                    S1 = sum_region(II, i-1, j-1, i-1+h-1, j-1+w-1)
                    S2 = sum_region(II, i-1, j-1+w, i-1+h-1, j-1+2*w-1)
                    features.append(S1 - S2)
                    f += 1
    
    # Feature type (b)
    for i in range(1, 25):
        for j in range(1, 25):
            for w in range(1, (25 - j) // 3 + 1):
                for h in range(1, 25 - i + 1):
                    S1 = sum_region(II, i-1, j-1, i-1+h-1, j-1+w-1)
                    S2 = sum_region(II, i-1, j-1+w, i-1+h-1, j-1+2*w-1)
                    S3 = sum_region(II, i-1, j-1+2*w, i-1+h-1, j-1+3*w-1)
                    features.append(S1 - S2 + S3)
                    f += 1
    
    # Feature type (c)
    for i in range(1, 25):
        for j in range(1, 25):
            for w in range(1, 25 - j + 1):
                for h in range(1, (25 - i) // 2 + 1):
                    S1 = sum_region(II, i-1, j-1, i-1+h-1, j-1+w-1)
                    S2 = sum_region(II, i-1+h, j-1, i-1+2*h-1, j-1+w-1)
                    features.append(S1 - S2)
                    f += 1
    
    # Feature type (d)
    for i in range(1, 25):
        for j in range(1, 25):
            for w in range(1, 25 - j + 1):
                for h in range(1, (25 - i) // 3 + 1):
                    S1 = sum_region(II, i-1, j-1, i-1+h-1, j-1+w-1)
                    S2 = sum_region(II, i-1+h, j-1, i-1+2*h-1, j-1+w-1)
                    S3 = sum_region(II, i-1+2*h, j-1, i-1+3*h-1, j-1+w-1)
                    features.append(S1 - S2 + S3)
                    f += 1
    
    # Feature type (e)
    for i in range(1, 25):
        for j in range(1, 25):
            for w in range(1, (25 - j) // 2 + 1):
                for h in range(1, (25 - i) // 2 + 1):
                    S1 = sum_region(II, i-1, j-1, i-1+h-1, j-1+w-1)
                    S2 = sum_region(II, i-1+h, j-1, i-1+2*h-1, j-1+w-1)
                    S3 = sum_region(II, i-1, j-1+w, i-1+h-1, j-1+2*w-1)
                    S4 = sum_region(II, i-1+h, j-1+w, i-1+2*h-1, j-1+2*w-1)
                    features.append(S1 - S2 - S3 + S4)
                    f += 1
    
    return np.array(features)

def haar_feature_scaling(image, feature_type, i, j, w, h):
    e = image.shape[0]
    assert e >= 24, "Image size should be at least 24x24"

    def round_nearest_integer(z):
        return int(np.round(z))

    if feature_type == 'a':
        a = 2 * w * h
        i = round_nearest_integer(i * e / 24)
        j = round_nearest_integer(j * e / 24)
        h = round_nearest_integer(h * e / 24)
        w = max(k for k in range(1, round_nearest_integer(1 + 2 * w * e / 24) // 2 + 1) if 2 * k <= e - j + 1)
        S1 = np.sum(image[i:i + h, j:j + w])
        S2 = np.sum(image[i:i + h, j + w:j + 2 * w])
        return (S1 - S2) * a / (2 * w * h)

    elif feature_type == 'b':
        a = 3 * w * h
        i = round_nearest_integer(i * e / 24)
        j = round_nearest_integer(j * e / 24)
        h = round_nearest_integer(h * e / 24)
        w = max(k for k in range(1, round_nearest_integer(1 + 3 * w * e / 24) // 3 + 1) if 3 * k <= e - j + 1)
        S1 = np.sum(image[i:i + h, j:j + w])
        S2 = np.sum(image[i:i + h, j + w:j + 2 * w])
        S3 = np.sum(image[i:i + h, j + 2 * w:j + 3 * w])
        return (S1 - S2 + S3) * a / (3 * w * h)

    elif feature_type == 'c':
        a = 2 * w * h
        i = round_nearest_integer(i * e / 24)
        j = round_nearest_integer(j * e / 24)
        w = round_nearest_integer(w * e / 24)
        h = max(k for k in range(1, round_nearest_integer(1 + 2 * h * e / 24) // 2 + 1) if 2 * k <= e - i + 1)
        S1 = np.sum(image[i:i + h, j:j + w])
        S2 = np.sum(image[i + h:i + 2 * h, j:j + w])
        return (S1 - S2) * a / (2 * w * h)

    elif feature_type == 'd':
        a = 3 * w * h
        i = round_nearest_integer(i * e / 24)
        j = round_nearest_integer(j * e / 24)
        w = round_nearest_integer(w * e / 24)
        h = max(k for k in range(1, round_nearest_integer(1 + 3 * h * e / 24) // 3 + 1) if 3 * k <= e - i + 1)
        S1 = np.sum(image[i:i + h, j:j + w])
        S2 = np.sum(image[i + h:i + 2 * h, j:j + w])
        S3 = np.sum(image[i + 2 * h:i + 3 * h, j:j + w])
        return (S1 - S2 + S3) * a / (3 * w * h)

    elif feature_type == 'e':
        a = 4 * w * h
        i = round_nearest_integer(i * e / 24)
        j = round_nearest_integer(j * e / 24)
        w = max(k for k in range(1, round_nearest_integer(1 + 2 * w * e / 24) // 2 + 1) if 2 * k <= e - j + 1)
        h = max(k for k in range(1, round_nearest_integer(1 + 2 * h * e / 24) // 2 + 1) if 2 * k <= e - i + 1)
        S1 = np.sum(image[i:i + h, j:j + w])
        S2 = np.sum(image[i + h:i + 2 * h, j:j + w])
        S3 = np.sum(image[i:i + h, j + w:j + 2 * w])
        S4 = np.sum(image[i + h:i + 2 * h, j + w:j + 2 * w])
        return (S1 - S2 - S3 + S4) * a / (4 * w * h)

    else:
        raise ValueError("Unknown feature type")

def decision_stump(X, y, weights, feature_index):
    n = len(y)
    # Sort by the feature
    sorted_indices = np.argsort(X[:, feature_index])
    X_sorted = X[sorted_indices]
    y_sorted = y[sorted_indices]
    weights_sorted = weights[sorted_indices]

    # Initialize variables
    tau = X_sorted[0, feature_index] - 1
    W_pos_above = np.sum(weights_sorted[y_sorted == 1])
    W_neg_above = np.sum(weights_sorted[y_sorted == -1])
    W_pos_below = 0
    W_neg_below = 0
    E = 2
    M = 0
    curr_M = 0
    for j in range(n):
        # Calculate weighted errors
        error_pos = W_neg_above + W_pos_below
        error_neg = W_pos_above + W_neg_below
        # print(error_pos, error_neg)
        if error_pos <= error_neg:
            error = error_pos
            toggle = 1
        else:
            error = error_neg
            toggle = -1

        # Update best decision stump
        if error < E or (error == E and curr_M > M):
            E = error
            best_tau = tau
            best_toggle = toggle
            M = curr_M

        while True:
            # Update weights
            if y_sorted[j] == -1:
                W_neg_below += weights_sorted[j]
                W_neg_above -= weights_sorted[j]
            else:
                W_pos_below += weights_sorted[j]
                W_pos_above -= weights_sorted[j]
            
            if j < n - 1 and X_sorted[j, feature_index] == X_sorted[j + 1, feature_index]:
                j += 1
            else:
                break

        if(j < n - 1):
            tau = (X_sorted[j, feature_index] + X_sorted[j + 1, feature_index]) / 2
            curr_M = X_sorted[j+1, feature_index] - X_sorted[j, feature_index]
        else :
            curr_M = 0
            tau = np.max([X_sorted[i, feature_index]for i in range(len(X_sorted[:,feature_index]))])

    return best_tau, best_toggle, E, M

def best_stump(X, y, weights, num_features):
    best_E = 2
    best_stump = decision_stump(X, y, weights, 0)
    for f in range(num_features):
        tau, toggle, E, M = decision_stump(X, y, weights, f)
        if E < best_E or (E == best_E and M > best_stump[3]):
            best_E = E
            best_stump = (tau, toggle, E, M)

    return best_stump

def adaboost(X, y, T):
    n = X.shape[0]
    weights = np.ones(n) / n
    learners = []
    alphas = []

    for t in range(T):
        tau, toggle, error, _ = best_stump(X, y, weights, X.shape[1])
        if error == 0:
            learners.append((tau, toggle))
            alphas.append(1)
            break

        alpha = 0.5 * np.log((1 - error) / error)
        learners.append((tau, toggle))
        alphas.append(alpha)

        predictions = toggle * np.sign(X[:, 0] - tau)
        weights *= np.exp(-alpha * y * predictions)
        weights /= np.sum(weights)

    def strong_learner(x):
        print(learners, alphas)
        return np.sign(sum(alpha * toggle * np.sign(x[0] - tau) for (tau, toggle), alpha in zip(learners, alphas)))

    return strong_learner


# Algorithm 7: Detecting faces with an Adaboost trained cascade classifier
def detect_faces(image, cascade_classifier, window_scale_multiplier=1.25):
    global step_size
    global positive_pretection
    global negative_pretection
    # Convert image to grayscale if it's not already
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Step 4: Initialize P with all possible windows
    M, N = image.shape
    P = []
    scale = 24
    
    while scale <= min(M, N):
        for i in range(0, M - scale + 1, step_size):  # Step size of 4 to speed up
            for j in range(0, N - scale + 1, step_size):
                P.append((i, j, scale))
        scale = int(scale * window_scale_multiplier)
    
    # Step 5-17: Iterate through layers of cascade classifier
    for l in range(len(cascade_classifier)):
        for window in P[:]:
            i, j, e = window
            windowed_image = image[i:i+e, j:j+e]
            mean = np.mean(windowed_image)
            std_dev = np.std(windowed_image)
            
            if std_dev > 1:
                normalized_image = (windowed_image - mean) / std_dev
                features = compute_features(normalized_image, cascade_classifier[l])
                
                if  cascade_classifier[l].predict(features)==negative_pretection:
                    P.remove(window)
            else:
                P.remove(window)
    
    return P

# Algorithm 8: Downsampling a square image
def downsample_image(image):
    e = image.shape[0]
    if e <= 24:
        return image
    
    # Step 3: Blur the image
    sigma = 0.6 * np.sqrt((e / 24) ** 2 - 1)
    blurred_image = cv2.GaussianBlur(image, (0, 0), sigma)
    
    # Step 4-12: Allocate a 24x24 matrix and compute the downsampled image
    downsampled_image = np.zeros((24, 24))
    for i in range(24):
        for j in range(24):
            i_scaled = int((e-1) * (i + 1) / 25)
            j_scaled = int((e-1) * (j + 1) / 25)
            i_min = max(0, i_scaled - 1)
            i_max = min(e - 1, i_scaled + 1)
            j_min = max(0, j_scaled - 1)
            j_max = min(e - 1, j_scaled + 1)
            downsampled_image[i, j] = (blurred_image[i_min, j_min] + blurred_image[i_min, j_max] + blurred_image[i_max, j_min] + blurred_image[i_max, j_max]) / 4
    
    return downsampled_image

# Algorithm 9: Collecting false positive examples for training a cascade’s (L + 1)-th layer
def collect_false_positives(images, cascade_classifier, window_scale_multiplier=1.25):
    V = []
    
    for image in images:
        Q = detect_faces(image, cascade_classifier, window_scale_multiplier)
        
        for (i, j, scale) in Q:
            windowed_image = image[i:i+scale, j:j+scale]
            
            if scale > 24:
                downsampled_image = downsample_image(windowed_image)
                if detect_faces(downsampled_image, cascade_classifier, window_scale_multiplier):
                    V.append(downsampled_image)
            else:
                V.append(windowed_image)
    
    return V

# Auxiliary function to compute features using Algorithm 3
def compute_features(image, classifier_layer):
    # Placeholder implementation (the actual implementation should follow Algorithm 3)
    features = []
    for feature in classifier_layer.features:
        feature_type, i, j, w, h = feature
        features.append(haar_feature_scaling(image, feature_type, i, j, w, h))
    return features

# Placeholder class for a classifier layer in the cascade
class ClassifierLayer:
    def __init__(self, features):
        self.features = features
    
    def predict(self, features):
        # Dummy prediction logic (to be replaced with actual logic)
        return np.sum(features) > 0

def attentional_cascade(n, m, training_positives, validation_positives, training_negatives, validation_negatives, gamma_o, gamma_l, beta_l):
    gamma_bo = 1
    layer_count = 0
    cascade = []

    while gamma_bo > gamma_o:
        layer_count += 1
        Tl = 1
        Sl = 0
        
        negative_training_examples = np.random.choice(training_negatives, 10 * n, replace=False)
        negative_validation_examples = np.random.choice(validation_negatives, m, replace=False)

        training_set = np.concatenate([training_positives, negative_training_examples])
        labels = np.concatenate([np.ones(len(training_positives)), -np.ones(len(negative_training_examples))])

        classifier = adaboost(training_set, labels, num_weak_classifiers=Tl)
        cascade.append(classifier)

        false_positives = 0
        for example in validation_negatives:
            if classifier(example) == 1:
                false_positives += 1

        gamma_bo = false_positives / len(validation_negatives)
    
    return cascade

# n = 100
# m = 50
# training_positives = np.random.rand(n, 24, 24)
# validation_positives = np.random.rand(m, 24, 24)
# training_negatives = np.random.rand(1000, 24, 24)
# validation_negatives = np.random.rand(200, 24, 24)
# gamma_o = 0.01
# gamma_l = 0.5
# beta_l = 0.5

# cascade = attentional_cascade(n, m, training_positives, validation_positives, training_negatives, validation_negatives, gamma_o, gamma_l, beta_l)

def post_processing(G, M, N, r):
    # Step 4: Create an M × N matrix E filled with zeros
    E = np.zeros((M, N))
    
    # Step 5-7: Populate matrix E with window sizes
    for (i, j, e) in G:
        E[i, j] = e

    # Step 8: Run a connected component algorithm on E
    labeled_array, num_features = label(E)

    P = []

    # Step 9-13: Process each component
    for component in range(1, num_features + 1):
        component_mask = (labeled_array == component)
        component_size = np.sum(component_mask)
        eC = np.unique(E[component_mask])[0]
        
        if component_size * (eC**-1) > r:
            coordinates = np.argwhere(component_mask)
            representative_window = (coordinates[0][0], coordinates[0][1], eC)
            P.append(representative_window)

    # Step 14: Sort the elements in P in ascending order of window size
    P = sorted(P, key=lambda x: x[2])

    # Step 15-25: Filter windows
    for i in range(len(P)):
        for j in range(i + 1, len(P)):
            if j >= len(P):
                break
            window_i = P[i]
            window_j = P[j]
            center_i = (window_i[0] + window_i[2] // 2, window_i[1] + window_i[2] // 2)
            center_j = (window_j[0] + window_j[2] // 2, window_j[1] + window_j[2] // 2)

            if (0 <= center_i[0] < M and 0 <= center_i[1] < N and 
                window_i[2] > window_j[2] and 
                window_i[0] <= center_j[0] < window_i[0] + window_i[2] and 
                window_i[1] <= center_j[1] < window_i[1] + window_i[2]):
                P.pop(j)
            elif (0 <= center_j[0] < M and 0 <= center_j[1] < N and 
                  window_j[2] > window_i[2] and 
                  window_j[0] <= center_i[0] < window_j[0] + window_j[2] and 
                  window_j[1] <= center_i[1] < window_j[1] + window_j[2]):
                P.pop(i)
                break
    
    return P

def rotate_image(image, angle):
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h))
    return rotated

def face_detection_with_rotation(image, theta, cascade_classifier):
    P = detect_faces(image, cascade_classifier)

    I_theta = rotate_image(image, theta)
    I_minus_theta = rotate_image(image, -theta)
    
    P_theta = detect_faces(I_theta, cascade_classifier)
    P_minus_theta = detect_faces(I_minus_theta, cascade_classifier)
    
    M, N = image.shape[:2]
    
    for w in P_theta:
        i, j, e = w
        center = (N // 2, M // 2)
        i_prime = int(center[0] + (i - center[0]) * np.cos(-np.radians(theta)) - (j - center[1]) * np.sin(-np.radians(theta)))
        j_prime = int(center[1] + (i - center[0]) * np.sin(-np.radians(theta)) + (j - center[1]) * np.cos(-np.radians(theta)))
        i_prime = min(max(0, i_prime), M - 1)
        j_prime = min(max(0, j_prime), N - 1)
        if not any((i_prime, j_prime, e) == x for x in P):
            P.append((i_prime, j_prime, e))

    for w in P_minus_theta:
        i, j, e = w
        center = (N // 2, M // 2)
        i_prime = int(center[0] + (i - center[0]) * np.cos(np.radians(theta)) - (j - center[1]) * np.sin(np.radians(theta)))
        j_prime = int(center[1] + (i - center[0]) * np.sin(np.radians(theta)) + (j - center[1]) * np.cos(np.radians(theta)))
        i_prime = min(max(0, i_prime), M - 1)
        j_prime = min(max(0, j_prime), N - 1)
        if not any((i_prime, j_prime, e) == x for x in P):
            P.append((i_prime, j_prime, e))
    
    return P

def skin_test(I):
    c = 0
    N = I.shape[0]
    M = I.shape[1]
    for i in range(N):
        for j in range(M):
            if I[i, j, 1] < I[i, j, 0] and I[i, j, 2] < I[i, j, 0]:
                c += 1
    return c / (N * M) > 0.4
# img = cv2.imread('imgs/color1.png')
# img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# print(skin_test(img))
# def train_next_layer(positives, negatives, cascade_classifier, num_weak_classifiers, target_false_positive_rate):
#     """
#     Trains the next layer (L+1) of the cascade classifier.
    
#     Parameters:
#     positives (list of numpy.ndarray): List of positive training examples.
#     negatives (list of numpy.ndarray): List of negative training examples.
#     cascade_classifier (list of ClassifierLayer): The current cascade classifier.
#     num_weak_classifiers (int): The number of weak classifiers to train in this layer.
#     target_false_positive_rate (float): The target false positive rate for this layer.
    
#     Returns:
#     ClassifierLayer: The trained (L+1)-th layer.
#     """
#     # Combine positives and negatives into a single training set
#     X = np.array(positives + negatives)
#     y = np.array([1] * len(positives) + [-1] * len(negatives))
    
#     # Initialize weights
#     weights = np.ones(len(y)) / len(y)
    
#     # Train the strong classifier using AdaBoost
#     strong_classifier = adaboost(X, y, num_weak_classifiers)
    
#     # Initialize variables for false positive detection
#     false_positives = []
    
#     # Evaluate the classifier on the training set
#     for i in range(len(X)):
#         if y[i] == -1 and strong_classifier(X[i]) == 1:
#             false_positives.append(X[i])
    
#     # Collect false positives
#     while len(false_positives) / len(negatives) > target_false_positive_rate:
#         new_negatives = collect_false_positives(false_positives, cascade_classifier)
#         X = np.concatenate([X, np.array(new_negatives)])
#         y = np.concatenate([y, -np.ones(len(new_negatives))])
#         weights = np.concatenate([weights, np.ones(len(new_negatives)) / len(y)])
        
#         strong_classifier = adaboost(X, y, num_weak_classifiers)
        
#         false_positives = []
#         for i in range(len(X)):
#             if y[i] == -1 and strong_classifier(X[i]) == 1:
#                 false_positives.append(X[i])
    
#     # Return the trained layer
#     return ClassifierLayer(strong_classifier)

# Example usage:
# X = np.array([[1], [2], [3], [4]])
# y = np.array([1, 1, -1, -1])
# strong_learner = adaboost(X, y, 10)
# print(strong_learner([3.5]))



# # Example usage:
# cascade_classifier = [
#     ClassifierLayer([('a', 0, 0, 12, 12), ('b', 4, 4, 8, 8)]),
#     ClassifierLayer([('c', 0, 0, 12, 12), ('d', 4, 4, 8, 8)]),
#     # Add more layers as needed
# ]

# image = cv2.imread('imgs/face00001.png', cv2.IMREAD_GRAYSCALE)
# detected_windows = detect_faces(image, cascade_classifier)
# print("Detected windows:", detected_windows)


# # read image imgs/face00001.png
# img = cv2.imread('imgs/face00001.png')
# # convert to gray scale
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# # resize the image to 24x24
# img = cv2.resize(gray, (24, 24))

# II = integral_image(img)
# print(len(compute_haar_like_features(gray,II)))
