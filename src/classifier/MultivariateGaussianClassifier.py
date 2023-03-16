import numpy as np

class GaussianNaiveBayesClassifier:
    
    # declaring variables
    def __init__(self):
        self.__n_features = 0
        self.__classes = []
        self.__class_probabs = []
        self.__class_cov_matrix = []
        self.__class_means = []
        
    # function to find parameters required by classifier
    def fit(self, X, y, cov_matrix = "different"):
        self.__n_features = X.shape[1]
        self.__classes, class_counts= np.unique(y, return_counts=True)
        self.__class_probabs = class_counts/len(y)
        self.__class_means = np.zeros((len(self.__classes), self.__n_features))
        for class_idx in range(len(self.__classes)):
            self.__class_means[class_idx] = np.array([np.mean(X[y==self.__classes[class_idx]][:,feature_num]) for feature_num in range(self.__n_features)])
        
        # initializing covariance matrix
        self.__class_cov_matrix = np.zeros((len(self.__classes), self.__n_features, self.__n_features))
        if cov_matrix == "identity": # all classes cov_matrix is identity matrix
            cov_matrix = np.identity(self.__n_features)
            self.__class_cov_matrix = np.array([cov_matrix for i in range(self.__classes.shape[0])])

        elif cov_matrix == "same": # all classes cov_matrix is global covariance matrix
            global_cov_matrix = np.identity(self.__n_features)
            for f_1 in range(self.__n_features):
                for f_2 in range(f_1, self.__n_features):
                    cov = np.cov(X[:, f_1], X[:, f_2])
                    global_cov_matrix[f_1][f_2] = cov[0][1]
                    global_cov_matrix[f_2][f_1] = cov[1][0]
                    global_cov_matrix[f_1][f_1] = cov[0][0]
                    global_cov_matrix[f_2][f_2] = cov[1][1]

            self.__class_cov_matrix = np.array([global_cov_matrix for i in range(self.__classes.shape[0])])

        elif cov_matrix == "different": # all classes have different cov matrix
            for class_idx in range(len(self.__classes)):
                class_cov_matrix = np.identity(self.__n_features)
                x_class = X[y==self.__classes[class_idx]]
                for f_1 in range(self.__n_features):
                    for f_2 in range(f_1, self.__n_features):
                        cov = np.cov(x_class[:, f_1], x_class[:, f_2])
                        class_cov_matrix[f_1][f_2] = cov[0][1]
                        class_cov_matrix[f_2][f_1] = cov[1][0]
                        class_cov_matrix[f_1][f_1] = cov[0][0]
                        class_cov_matrix[f_2][f_2] = cov[1][1]

                self.__class_cov_matrix[class_idx] = class_cov_matrix

        
    def __gaussian(self, x, mean, cov_matrix):
        return 1.0/(np.power(2*np.pi, self.__n_features/2) * np.power(np.linalg.det(cov_matrix), 0.5)) * np.exp(-0.5 * (x- mean).T @ np.linalg.inv(cov_matrix) @ (x-mean))

    # predict values P(Ci|X) = p(X|Ci) * P(Ci) / P(X)
    def predict(self, X):
        # P(C|X)
        p_c_gvn_x = []
        for x in X:
            
            # p(Ci|x)
            p_ci_gvn_x = []
            for idx in range(len(self.__classes)):
                numr = self.__class_probabs[idx]
                p_x_gvn_ci = self.__gaussian(x, self.__class_means[idx], self.__class_cov_matrix[idx])
                numr *= p_x_gvn_ci
                p_ci_gvn_x.append(numr)
            
            p_c_gvn_x.append(self.__classes[np.argmax(np.array(p_ci_gvn_x))])
        return np.array(p_c_gvn_x)