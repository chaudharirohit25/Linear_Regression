class LR:

    def __init__(self ):
        pass

    
    def fit(self, X,y):
        #school book method
        # adding X=1 for bias
        X_b = np.ones(X.shape[0]).reshape(-1,1)
        X_w = X
        X_wb = np.concatenate((X_b,X_w),axis = 1)
        self.W_wb = np.matmul(np.matmul(np.linalg.inv(np.matmul(X_wb.T,X_wb)), X_wb.T),y)
        self.intercept= self.W_wb[0]
        self.coeffs = self.W_wb[1:]
    
        return self
        
    def predict(self,X):
        X_b = np.ones(X.shape[0]).reshape(-1,1)
        X_w = X
        X_wb = np.concatenate((X_b,X_w),axis = 1)
        y_pred = np.matmul(X_wb,self.W_wb)
        
        return y_pred
       
