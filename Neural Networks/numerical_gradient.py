import numpy as np

def compute_numerical_gradient(model,x,y,epi,reg):
  dParam = {}
  for param_n,param in model.parameters.items():
    if len(param.shape) == 1:
      param = param[None,:]
    dW = np.zeros_like(param)
    m,n = dW.shape
    for i in range(m):
      for j in range(n):
        param[i][j] += epi

        fin = model.loss(x,y,lamb = reg)[1]

        param[i][j] -= 2*epi

        init = model.loss(x,y,lamb = reg)[1]

        param[i][j] += epi

        dW[i][j] = (fin - init)/(2*epi)

    dParam[param_n] = dW
  return dParam