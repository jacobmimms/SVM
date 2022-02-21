import numpy as np, random
from scipy.optimize import minimize
import matplotlib.pyplot as plt

P = None

def SVM(number_samples, constraints, C = .1):
    N = number_samples
    data = TestData(N)
    inputs = data.inputs
    targets = data.targets
    precompute_P(inputs, targets, linear_kernal)
    start = np.zeros(N)
    test = np.random.rand(10)
    assert np.round(naive_objective_function(test),8) == np.round(objective_function(test),8)
    objective = objective_function
    B = [(0, C) for b in range(N)]
    XC = constraints #dict with 'type' and 'fun' fields 
    ret = minimize(objective, start, bounds=B, constraints=XC)
    alpha = ret['x']
    return 

def precompute_P(inputs, targets, kernel):
    """precomputes the matrix P_ij = (t_i)(t_j)K(x_i, x_j)

    Args:
        inputs (np.ndarry): a numpy array of the input datapoints 
        targets (np.ndarray): a numpy array of the datapoint classifications
        kernel (function): the kernel function to be used (takes two arguments which are type np.ndarray)
    """
    global P
    size = targets.size
    P = np.ndarray((size, size))
    for i in range(size):
        for j in range(size):
            P[i][j] = targets[i] * targets[j] * kernel(inputs[i], inputs[j])
    
    
def naive_objective_function(a_vec):
    ##TODO: implemement objective function
    """ 
    implements equation 4: 1/2∑∑(a_i)(a_j)(t_i)(t_j)K(x_i, x_j) - ∑a_i 
    """
    length = a_vec.size
    result = 0
    for i in range(length):
        for j in range(length):
            result += a_vec[i] * a_vec[j] * P[i][j]
    return .5 * result - np.sum(a_vec)

def objective_function(a_vec):
    """ 
    implements equation 4: 1/2∑∑(a_i)(a_j)(t_i)(t_j)K(x_i, x_j) - ∑a_i 
    """
    result = np.sum(np.dot(np.dot(a_vec,P),a_vec))
    return .5 * result - np.sum(a_vec)


def linear_kernal(x,y):
    """
    args:
        x,y: numpy arrays

    return value: 
        the scalar product K(x,y)=xT·y 
    """
    return np.inner(x.T, y)



def plot_decision_boundary():
    return 

def plot_data(data, save_plt, filename):
    plt.plot([p[0] for p in data.classA], [p[1] for p in data.classA], 'b.')
    plt.plot([p[0] for p in data.classB], [p[1] for p in data.classB], 'r.')
    if save_plt:
        plt.savefig(f"{filename}.pdf")
    plt.axis("equal")
    plt.show()

class TestData:
    inputs = None
    tarets = None
    classA = None
    classB = None

    def __init__(self, N) -> None:
        np.random.seed(100)
        classA = np.concatenate( 
            (np.random.randn(10, 2) * 0.2 + [1.5, 0.5],
            np.random.randn(10, 2) * 0.2 + [-1.5, 0.5]))
        classB = np.random.randn(20, 2) * 0.2 + [0.0 , -0.5]
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
    SVM(10, {})