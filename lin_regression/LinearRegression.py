import numpy as np

class LinearRegression:
    def __init__(self, learning_rate=0.01, mode=1, n_iteration=2000, epsilon=1e-6):
        self.learning_rate = learning_rate
        self.mode = mode
        self.n_iteration = n_iteration
        self.epsilon = epsilon
        self.theta = None

    def compute_hypothesis(self, theta, x):
        '''
        Compute and return the value of h(x) = theta^T x
        theta: coefficient vector (pesi)
        x: feature vector
        '''
        #h = 0
        #for i in range(len(theta)):
        #    h = h + theta[i] * x[i]
        #return h
        return np.dot(theta, x)

    def compute_derivative(self, j, theta, x, y, m):
        '''
        Compute the derivative of the cost function
        '''
        der=0
        
        for i in range(m):
            h = self.compute_hypothesis(theta,x[i])
            #print('h= ',h)
            der = der + (h - y[i]) * x[i,j]
            
        return der / m
    
    def mean_squared_error(self, x, y, theta, m):
        '''
        Compute the mean square error
        '''
        error = 0
        for i in range(m):
            h = self.compute_hypothesis(theta, x[i])
            error += (h - y[i]) ** 2
        return error / (2 * m)
    
    def gradient_descent(self, X, y_true):
        '''
        Perform the gradient descent
        '''
        m, f = X.shape
        # Inizializza i pesi con valori casuali piccoli
        theta = np.random.randn(f) * 0.1  
        theta = np.round(theta, 5)
        prev_cost = float('inf')
        
        #si parte da valori casuali di theta
        #si continua a cambiare theta per ridurre J(t) fino a che si minimizza la funzione
        #passaggi ripetuti iterativamente fino a una condizione di terminazione
        if self.mode == 1:
            for _ in range(self.n_iteration):
                temp=np.copy(theta)

                for j in range(0,len(theta)):        
                    temp[j] = theta[j] - self.learning_rate * self.compute_derivative(j,theta,X,y_true,m)

                theta = np.copy(temp)

                #print(theta)

        elif self.mode == 2:
            while True:

                temp=np.copy(theta)

                for j in range(0,len(theta)):        
                    temp[j] = theta[j] - self.learning_rate * self.compute_derivative(j,theta,X,y_true,m)

                theta = np.copy(temp)

                # Calcola la funzione di costo e verifica la convergenza
                cost = self.mean_squared_error(X, y_true, theta, m)
                if abs(cost - prev_cost) < self.epsilon:
                    break
                prev_cost = cost

                #print(theta)

        return theta

    def fit(self, X, y):
        '''
        Takes the X and Y parameters as input to perform gradient descent and calculate the theta parameters
        '''
        X_scaled = scale_features(X)
        self.theta = self.gradient_descent(X_scaled, y)
    
    def predict(self, X):
        '''
        Predicts the x-values based on the calculated values from gradient descent
        '''
        X_scaled = scale_features(X)
        return np.dot(X_scaled, self.theta)




def scale_features(X):
    '''
    Scale the features in values between 0 and 1 in order to make them more uniform
    '''
    mean = np.mean(X)
    std = np.std(X)
    scaled_X = (X - mean) / std
    
    # Gestisci valori NaN o infiniti dopo la standardizzazione
    scaled_X = np.nan_to_num(scaled_X, nan=0.0, posinf=1.0, neginf=-1.0)
    
    return scaled_X
    
def r_squared(y_true, y_pred):
    ss_total = np.sum((y_true - np.mean(y_true)) ** 2)
    ss_residual = np.sum((y_true - y_pred) ** 2)
    r2 = 1 - (ss_residual / ss_total)
    return r2
    
def r2_score(y_test, y_pred):
    r2 = r_squared(y_test, y_pred)
    print("R-squared: ", r2)