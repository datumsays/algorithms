"""
Author: Daniel Lee
Date: 09-04-17
Objective: A script that trains a higher-order markov model and generates a sequence.
"""

from scipy.stats import itemfreq
import numpy as np

class MarkovModel:

    def __init__(self, series, states, lags):
        """

        """
        self.series = series
        self.states = states
        self.lags   = lags
        self.mu     = series.mean()
        self.var    = series.var()
        self.N      = len(self.series)
        return

    def standardize(self):
        """standardizes values into standard normal (mu=0 | var=1)"""
        sqrt = np.sqrt

        def zscore(x, mu, var, N):
            return (x - mu) / sqrt(var)

        self.series = [zscore(x=x, mu=self.mu, var=self.var, N=self.N) \
                      for x in self.series]
        return self.series

    def partition(self, series, bins):
        """Bins data based on user defined partitions"""
        inds = np.digitize(x=series, bins=bins)
        return inds

    def transition(self, series, bins, lags):
        """Creates k-th order transition matrix"""
        dim = len(bins)
        transition_matrix = np.zeros((dim,dim)) # Create a transition matrix with zeros
        transition = [transition_matrix.copy() for i in range(lags)] # Create a list of transition matrix
        length = len(series)
        for k in range(lags):
            k = k + 1
            for i, j in zip(series[k:], series[:length-k]): # j = state before, i = state after
                i -= 1
                j -= 1
                transition[k-1][i,j] += 1
            transition[k-1] = np.apply_along_axis(lambda x: x / sum(x), 0, transition[k-1])
        return transition

    def linear_programming(self, state, trans):
        """calculate the weight for each transition matrix"""
        combo = np.concatenate([[np.dot(mat, state.T)] for mat in trans], axis=0).T
        ones = -1 * np.ones((combo.shape[0],1))
        A = np.hstack((combo, ones))
        b = state.T

        A_stacked = np.column_stack((A, -1 * A)).flatten()
        print(A_stacked)

        objective = np.concatenate([np.repeat(0, self.lags),[1]])
        lambda_sum = np.concatenate([np.repeat(1, self.lags),[0]])
        lambda_b = [1]
        bnds = [(0, np.inf)] * self.lags
        return

    def state_prob(self, series):
        """Creates transition vector"""
        unique = itemfreq(series)[:,1] / len(series)
        return unique

    def train(self):
        return

    def generate(self, periods):
        return


X = np.array([-100,5,1,2,3,2,1,1,1,2,3,3,3,100])
mm = MarkovModel(series=X, states=3, lags=2)
std = mm.standardize()
partitions = mm.partition(std, bins=np.arange(-1,2,1))
partitions = [1,1,2,2,1,3,2,1,2,3,1,2,3,1,2,3,1,2,1,2]
trans = mm.transition(series=partitions, bins=np.unique(partitions), lags=2)
prob = mm.state_prob(series=partitions)
mm.linear_programming(state=prob, trans=trans)
