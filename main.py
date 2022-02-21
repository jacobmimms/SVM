import numpy
import random
from scipy.optimize import minimize
import matplotlib.pyplot as plt

def SVM(number_samples, constraints, C = .1):
    N = number_samples
    data = TestData(N)
    plot_data(data, True, "test")
    return
    objective = None
    start = numpy.zeros(N)
    B = [(0, C) for b in range(N)]
    XC = constraints #dict with 'type' and 'fun' fields 
    ret = minimize(objective, start, bounds=B, constraints=XC)
    alpha = ret['x']
    return 

def objective_function():
    ##TODO: implemement objective function
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
        numpy.random.seed(100)
        classA = numpy.concatenate( 
            (numpy.random.randn(10, 2) * 0.2 + [1.5, 0.5],
            numpy.random.randn(10, 2) * 0.2 + [-1.5, 0.5]))
        classB = numpy.random.randn(20, 2) * 0.2 + [0.0 , -0.5]
        inputs = numpy.concatenate((classA , classB))
        targets = numpy.concatenate((numpy.ones(classA.shape[0]), 
                                    -numpy.ones(classB.shape[0])))
        permute = list(range(N)) 
        random.shuffle(permute) 
        self.inputs = inputs[permute, :]
        self.targets = targets[permute]
        self.classA = classA
        self.classB = classB


if __name__ == "__main__":
    SVM(10, {})