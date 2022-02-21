import numpy as np
import random, math
from scipy.optimize import minimize
import matplotlib.pyplot as plt

def main():
    N = 20
    C = .5
    svm = SVM(N, "poly")
    XC = {'type': 'eq', 'fun': svm.zerofun}
    B = [(0, C) for b in range(N)]
    start = np.zeros(N)
    ret = minimize(svm.objective, start, bounds=B, constraints=XC)
    alpha = ret['x']
    print(alpha)
    print(ret)
    plot_data(svm.data)
    # plot_decision_boundary()
    return 

class Kernel:
    def __init__(self, ker, ps = 0):
        self.ker = ker
        self.ps = ps
        return

    def __getitem__(self, input):
        if self.ker == 'linear': return np.dot(input[0],input[1])
        if self.ker == 'poly': return (np.dot(input[0],input[1])+1)**self.ps
        if self.ker == 'rbf': return math.exp(-(np.dot(np.subtract(input[0],input[1]) , np.subtract(input[0],input[1]))) / (2*self.ps)**2)

class SVM:
    def __init__(self, number_samples, ker='linear'):
        self.N = number_samples
        self.data = TestData(self.N)
        self.inputs = self.data.inputs
        self.targets = self.data.targets
        self.kernel = { 
            'linear': lambda x, y: np.dot(x.T,y), 
            'poly': lambda x, y: (np.dot(x,y) + 1) ** 2, 
            'rbf' : lambda x, y: math.exp(-(np.dot(np.subtract(x,y), np.subtract(x,y))) / (2*2)**2)
        }.get(ker)
        self.P = self.precompute_P()

    def precompute_P(self):
        size = self.N
        P = np.ndarray((size, size))
        for i in range(size):
            for j in range(size):
                P[i][j] = self.targets[i] * self.targets[j] * self.kernel(self.inputs[i], self.inputs[j])
        return P
    
    def objective(self, a_vec):
        return np.dot(np.dot(a_vec, self.P),a_vec)/2 - np.sum(a_vec)

    def zerofun(self, a_vec):
        return np.dot(a_vec, self.targets)

def plot_decision_boundary():
    return 

def plot_data(data, save_plt = False, filename = ""):
    plt.plot([p[0] for p in data.classA], [p[1] for p in data.classA], 'b.')
    plt.plot([p[0] for p in data.classB], [p[1] for p in data.classB], 'r.')
    if save_plt:
        plt.savefig(f"{filename}.pdf")
    plt.axis("equal")
    plt.show()

class TestData:
    def __init__(self, N):
        np.random.seed(100)
        classA = np.concatenate( 
            (np.random.randn(int(N/2), 2) * 0.2 + [1.5, 0.5],
            np.random.randn(int(N/2), 2) * 0.2 + [-1.5, 0.5]))
        classB = np.random.randn(N, 2) * 0.2 + [0.0 , 0.5]
        inputs = np.concatenate((classA , classB))
        targets = np.concatenate((np.ones(classA.shape[0]), 
                                    -np.ones(classB.shape[0])))
        permute = list(range(N)) 
        random.shuffle(permute) 
        self.inputs = inputs[permute, :]
        self.targets = targets[permute]
        self.classA = classA
        self.classB = classB


if __name__ == "__main__":
    main()