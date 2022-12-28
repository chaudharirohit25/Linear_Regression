class LR_FB_GD:

    def __init__(self,eta = 0.01, n_iterations = 1000 ):
        self.eta = eta
        self.n_iterations = n_iterations
    
    def fit(self, X,y):
        #full batch gradient descent method
        # adding X=1 for bias
        
        X_b = np.ones(X.shape[0]).reshape(-1,1)
        X_w = X
        X_wb = np.concatenate((X_b,X_w),axis = 1)
        if len(y.shape)==1:
            self.W_wb = np.random.random((X_wb.shape[1],))
        else:
            self.W_wb = np.random.random((X_wb.shape[1],y.shape[1]))

        for iteration in range(self.n_iterations):
            gradients = 2./X.shape[0] * np.matmul(X_wb.T,(np.matmul(X_wb, self.W_wb) - y))
          
            self.W_wb= self.W_wb - self.eta * gradients
         
                           
        self.intercept= self.W_wb[0]
        self.coeffs = self.W_wb[1:]
        self.gradients = gradients
        return self
        
    def predict(self,X):
        X_b = np.ones(X.shape[0]).reshape(-1,1)
        X_w = X
        X_wb = np.concatenate((X_b,X_w),axis = 1)
        y_pred = np.matmul(X_wb,self.W_wb)
        
        return y_pred
        
