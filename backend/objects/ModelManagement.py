import pickle

class ModelManagement:
    def __init__(self, X=None, y=None):
        self.X, self.y = X, y
        self.X_train, self.y_train, self.X_test, self.y_test = None, None, None, None
        self.model = None
        # self.preprocessor = None
        
    def load(self, filename):
        with open("../../" + filename, 'rb') as file:
            self.model = pickle.load(file)
                
    def predict(self, X):
        return self.model.predict(X)