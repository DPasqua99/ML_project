import numpy as np
from collections import Counter

class KNNClassifier:
    
    def __init__(self, k, p=2):
        self.k = k
        self.p = p
        self.X = None
        self.y = None

    def fit(self, X, y):
        self.X = np.array(X)
        self.y = np.array(y)
    
    def minkowski_distance(self, p1, p2):
        return np.sum(np.abs(p1 - p2)**self.p)**(1/self.p)
    
    def predict(self, X_test):
        y_pred = []
        for x in X_test:
            # Calcola la distanza tra il punto di test e tutti i punti di addestramento
            distances = [self.minkowski_distance(x, point) for point in self.X]
            
            # Ottieni gli indici dei k punti più vicini
            k_indices = np.argsort(distances)[:self.k]
            
            # Preleva le etichette dei k punti più vicini
            k_labels = self.y[k_indices]
            
            # Vota per l'etichetta più comune tra i k punti più vicini
            most_common = Counter(k_labels).most_common(1)
            y_pred.append(most_common[0][0])
        
        return y_pred
