import numpy as np
import random, math
from scipy.optimize import minimize
import matplotlib.pyplot as plt

np.random.seed(100)

classA = np.concatenate( 
    (np.random.randn(10, 2) * 0.2 + [1.5, 0.5],
    np.random.randn(10, 2) * 0.2 + [-1.5, 0.5]))
classB = np.random.randn(20, 2) * 0.2 + [0.0 , -.5]
# classA = np.array([[0,0], [-1,-1]])
# classB = np.array([[1,1],[2,2]])

inputs = np.concatenate((classA , classB))
targets = np.concatenate((np.ones(classA.shape[0]), 
                            -np.ones(classB.shape[0])))
N = inputs.shape[0]

permute = list(range(N)) 
random.shuffle(permute) 
inputs = inputs[permute, :]
targets = targets [ permute ]

#TODO choose kernel type
kernel_type = 'linear'
p = 2
sigma =.1
kernel = { 
        'linear': lambda x, y: np.dot(x,y), 
        'poly': lambda x, y: (np.dot(x,y) + 1) ** p, 
        'rbf' : lambda x, y: math.exp(-(np.linalg.norm(x-y))/2*(sigma^2))
    }.get(kernel_type)

P = np.ndarray((N, N))
for i in range(N):
    for j in range(N):
        P[i][j] = targets[i] * targets[j] * kernel(inputs[i], inputs[j])

#TODO choose C
C = 10
B = [(0, C) for b in range(N)]
start = np.zeros(N)

def zerofun(a):
    return np.dot(a, targets)
XC = {'type': 'eq', 'fun': zerofun}

def objective(a):
    return .5 * np.dot(np.dot(a, P), a) - np.sum(a)

def get_b(a, sv):
    #sv[0] = inputs[i]
    #sv[1] = targets[i]
    #sv[2] = i
    b = 0 
    for v in sv:
        b += a[v[2]] * v[1] * kernel(v[0], sv[0][0]) 

    return b - sv[0][1]

def indicator(a, point, b):
    indy = 0
    for i in range(N):
        indy += a[i] * targets[i] * kernel(point, inputs[i]) 
    return indy - b

def main():
    ret = minimize(objective, start, bounds=B, constraints=XC)
    alpha = ret['x']
    alpha = [round(x,5) for x in alpha]
  
    support_vectors = [(inputs[i], targets[i], i) for i in range(N) if alpha[i] != 0]
    other_points = [(inputs[i], targets[i], i) for i in range(N) if alpha[i] == 0]
    
    plot(alpha, support_vectors, other_points)
    plt.show()

def plot(a, support_vectors, other_points, save_plt = False, filename = ""):
    
    plt.plot([p[0][0] for p in support_vectors if p[1] == 1], [p[0][1] for p in support_vectors if p[1] == 1], 'b+')
    plt.plot([p[0][0] for p in support_vectors if p[1] == -1], [p[0][1] for p in support_vectors if p[1] == -1], 'r+')

    plt.plot([p[0][0] for p in other_points if p[1] == 1], [p[0][1] for p in other_points if p[1] == 1], 'b.')
    plt.plot([p[0][0] for p in other_points if p[1] == -1], [p[0][1] for p in other_points if p[1] == -1], 'r.')
    

    if save_plt:
        plt.savefig(f"{filename}.pdf")
    plt.axis("equal")

    xgrid = np.linspace(-2, 5) 
    ygrid = np.linspace(-2, 5)
    grid = np.array([[indicator(a, np.array([x, y]), get_b(a, support_vectors)) for x in xgrid] for y in ygrid])
    plt.contour(xgrid, ygrid, grid , (-1, 0.0, 1.0),
    colors=('red', 'black', 'blue'), linewidths=(1, 3, 1))



if __name__ == "__main__":
    main()