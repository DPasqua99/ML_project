import random
import numpy as np

class KMeans:
    def __init__(self, n_clusters, max_iter, random_state=None, p=2):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.random_state = random_state
        self.p = p
        self.centroids = None  # Inizializza l'attributo centroids come None

    def minkowski_distance(self, p1, p2):
        return np.sum(np.abs(p1 - p2)**self.p)**(1/self.p)

    def fit(self, X):
        if self.random_state is not None:
            random.seed(self.random_state)

        # Inizializza i centroidi selezionando casualmente K punti dai dati
        initial_indices = random.sample(range(len(X)), self.n_clusters)
        self.centroids = [X[i] for i in initial_indices]

        for _ in range(self.max_iter):
            # Assegna ciascun punto al centroide più vicino
            labels = self.predict(X)
            
            # Calcola nuovi centroidi come media dei punti nei cluster
            new_centroids = [self.calculate_new_centroid(X, labels, i) for i in range(self.n_clusters)]
            
            # Verifica la convergenza
            if np.all(np.array(self.centroids) == np.array(new_centroids)):
                break
            
            self.centroids = new_centroids

        self.centroids = np.array(self.centroids)
        
        return self
    
    
    def calculate_new_centroid(self, X, labels, cluster):
        # creo un elenco che contiene tutti i punti del dataset X che sono assegnati a un cluster specificato
        cluster_points = [X[i] for i in range(len(X)) if labels[i] == cluster]
        
        # condizione che verifica se il cluster e vuoto: non puo calcolare nuovi centroidi
        if len(cluster_points) == 0:
            return self.centroids[cluster]
        
        # scompone la lista cluster_points in una serie di tuple, ognuna contenente un punto dalle coordinate corrispondenti
        # calcola la media delle coordinate corrispondenti dei punti in ciascuna dimensione, cioè la media di x e la media di y (nuovo centroide del cluster)
        return [sum(p) / len(cluster_points) for p in zip(*cluster_points)]
    
    
    def predict(self, X):
        labels = []
        for x in X:
            distances = [self.minkowski_distance(x, centroid) for centroid in self.centroids]
            labels.append(distances.index(min(distances)))
        return np.array(labels)