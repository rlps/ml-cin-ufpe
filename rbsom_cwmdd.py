import pandas as pd
import numpy as np
import gc
import matplotlib.pyplot as plt

from scipy.spatial import distance_matrix, distance
from tqdm.notebook import tqdm
from collections import defaultdict
from IPython.display import clear_output

class RBSOM_CWMdd():
    """
    The 2-D, rectangular grid self-organizing map class using Numpy.
    """
    def __init__(self, n=1.1, q=5,x=6, y=6, max_iter=50, random_state=None):
        """
        Parameters
        ----------
        m : int, default=3
            The shape along dimension 0 (vertical) of the SOM.
        n : int, default=3
            The shape along dimesnion 1 (horizontal) of the SOM.
        dim : int, default=3
            The dimensionality (number of features) of the input space.
        lr : float, default=1
            The initial step size for updating the SOM weights.
        sigma : float, optional
            Optional parameter for magnitude of change to each weight. Does not
            update over training (as does learning rate). Higher values mean
            more aggressive updates to weights.
        max_iter : int, optional
            Optional parameter to stop training if you reach this many
            interation.
        random_state : int, optional
            Optional integer seed to the random number generator for weight
            initialization. This will be used to create a new instance of Numpy's
            default random number generator (it will not call np.random.seed()).
            Specify an integer for deterministic results.
        """
        # Initialize features
        self.x = x
        self.y = y
        self.q = q
        self.n = n
        self.C = int(x*y)
        self.max_iter = max_iter
        self.random_state = random_state
        self.sigma_0 = np.sqrt(-(2*self.C)/2*np.log(0.1))
        self.sigma_f = np.sqrt(-1/2*np.log(0.01))
        self.delta = self.create_delta_matrix()

        #Initialize results log dict
        self.iteration_results = defaultdict()

    def create_delta_matrix(self):
        x = self.x
        y = self.y
        C = self.C
        nodes = []
        for i in range(1,x+1):
            for j in range(1,y+1):
                nodes.append([i,j])

        delta = np.empty((C,C))

        for i in range(delta.shape[0]):
            for j in range(delta.shape[1]):
                delta[i,j] = np.square(distance.euclidean(nodes[i],nodes[j]))
        
        return delta

    def calc_sigma(self,t):
        return self.sigma_0*pow((self.sigma_f/self.sigma_0),t/50)

    def calc_h(self,s,r,sigma):
        s=int(s)
        r=int(r)
        dist = delta[i,j]
        return np.exp(-dist/2*np.square(sigma))

    def build_h_matrix(self, sigma):
        C = self.C
        h_matrix = np.empty((C,C))

        for i in range(h_matrix.shape[0]):
            for j in range(h_matrix.shape[1]):
                h_matrix[i,j] = calc_h(i,j,sigma)
        
        return h_matrix
    
    def initialize_prototypes(self, X):
        C = self.C
        G = []
        for i in range(C):
            G.append(pd.DataFrame(X).sample(5).index)

        return np.array(G)

    def init_weight_matrix(self):
        C = self.C
        q = self.q

        v = pd.DataFrame(np.random.rand(C,q))
        v_row_sum = np.array(v.sum(axis=1))
        v = np.array(v)

        for i in range(v.shape[0]):
            for j in range(v.shape[1]):
                v[i,j] = v[i,j]/v_row_sum[i]
        
        return v
    
    def build_norm_dmatrix(self, df):
        dissimilarity = pd.DataFrame(distance_matrix(df.values,df.values),index=df.index,columns=df.index)
        norm_diss = np.array(dissimilarity/dissimilarity.iloc[np.argmin(dissimilarity.sum())].sum())
        return dissimilarity

    
    def D_vr(e,r):
        G_r = G[r]
        return sum(
            [np.power(v[r,i],n)*norm_diss[e,G_r[i]] for i in range(v.shape[1])]
        )

    def define_partitions(self, N):
        C = self.C
        h_matrix = self.h_matrix
        P = P = [[] for i in range(C)]
        clusters_array = np.zeros(N)

        for k in range(N):
            deltas = np.empty(C)
            for s in range(C):
                deltas[s] = sum([h_matrix[s,r]*D_vr(k,r) for r in range(C)])

            k_cluster = np.argmin(deltas)
            clusters_array[k] = k_cluster
            clusters_array = clusters_array.astype(int)
            P[k_cluster].append(k)

        return P,clusters_array   




    def fit(self, df):
        X = df.values
        N = X.shape[0]
        norm_diss = self.build_norm_dmatrix(df)
        # iter_counter = 0

        self.G = self.initialize_prototypes(X)
        self.v = self.init_weight_matrix(v)
        self.sigma = self.calc_sigma(0)
        self.h_matrix = self.build_h_matrix(sigma)

        P,clusters_array = self.define_partitions(N)

        for t in range(1,max_iter+1):

        
