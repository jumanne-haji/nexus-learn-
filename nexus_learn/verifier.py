def exact(problem,pred,tol=0.25):
    return abs(float(pred)-float(problem.answer))<=tol
