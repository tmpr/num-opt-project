import numpy as np

def backtrack_line_search(x: np.array, direction: np.array, f, grad, alpha=1, c=0.5, r=0.8):
    if f(*(x + alpha*direction)) <=  f(*x) + c*alpha*  np.dot(grad, direction):
        print("Used alpha: ", alpha)
        return alpha
    else:
        return backtrack_line_search(x, direction, f, grad, r*alpha, c, r)