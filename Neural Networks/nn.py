from os import WIFSIGNALED
import numpy as np

class NeuralNetwork:
  '''
  Network must contain atleast one hidden layer
  i.e len(hidden_layer_sizes) >=1
  '''
  def __init__(self, input_size, hidden_layer_sizes,output_size,w_scale=1,verbose=False) -> None:
    self.input_size = input_size
    self.hidden_layer_sizes = hidden_layer_sizes
    self.output_size = output_size
    self.num_layers= len(hidden_layer_sizes)+1
    self.verbose = verbose
    self.parameters = {}
    self.w_scale = w_scale
    self.reset_network()
  
  def reset_network(self):
    self.parameters['w0'] = np.random.normal(0,self.w_scale,(self.input_size,self.hidden_layer_sizes[0]))
    self.parameters['b0'] = np.random.normal(0,self.w_scale,(self.hidden_layer_sizes[0],))

    for i,layer_size in enumerate(self.hidden_layer_sizes[1:]+[self.output_size]):
      self.parameters['w'+str(i+1)] = np.random.normal(0,self.w_scale,(self.hidden_layer_sizes[i],layer_size))
      self.parameters['b'+str(i+1)] = np.random.normal(0,self.w_scale,(layer_size,))

  def set_theta(self,theta,num):
    '''
    theta: the weights as numpy array (should be given in the same format in the example files)
    num: num = i for theta'i'
    For example:
    theta1 => num =1
    theta2 => num =2
    '''
    bias = theta[:,0]
    weights = theta[:,1:].T
    self.parameters["w"+str(num-1)] = weights
    self.parameters["b"+str(num-1)] = bias

  def get_theta(self,num):
    weights = self.parameters["w"+str(num-1)] 
    bias = self.parameters["b"+str(num-1)]
    return np.concatenate([bias[:,None],weights.T],axis = 1)

  def print_w_as_theta(self,params):
    ret_param = {}
    for i in range(self.num_layers):
      w = params['w'+str(i)]
      b = params['b'+str(i)]
      ret_param['theta'+str(i+1)] = np.concatenate([b[:,None],w.T],axis = 1)
      print('theta'+str(i+1),ret_param['theta'+str(i+1)],sep='\n')

  def print_loss_instance(self,instance,y):
    scores, loss, params = self.loss(instance[None,:],y[None,:],0)
    print("Gradients ")
    self.print_w_as_theta(params)    
    print()

  def update_params(self,grads,lr):
    for p in grads:
      self.parameters[p] -= lr * grads[p]

  def loss(self,X,y,lamb=0):

    # X = np.concatenate([np.ones(X.shape[0])[:,None],X],axis=1)
    out = X
    caches = []
    grads = {}
    weight_square_sum = []
    N = X.shape[0]
    if self.verbose:
      print("a1",np.concatenate([np.ones(out.shape[0])[:,None],out],axis = 1),sep="\n")
      print()

    for i in range(self.num_layers):
      out,cache = affine_sigmoid_forward(out,self.parameters["w"+str(i)],self.parameters["b"+str(i)],(self.verbose,i))
      caches.append(cache)
      weight_square_sum.append( np.sum(self.parameters["w"+str(i)]*self.parameters["w"+str(i)])) # +np.sum(self.parameters["b"+str(i)]**2)
    
    scores = out
    
    if self.verbose:
      print("f(x)",scores,sep="\n")
      print()
      print("y",y,sep="\n")
      print()
    # scores = scores[:,1:]
    loss,dout = logistic_loss(scores,y,self.verbose)

    loss += lamb * np.sum(weight_square_sum)/(2*(N))
    
    # dout = np.concatenate([np.zeros(dout.shape[0])[:,None],dout],axis=1)

    for i in reversed(range(self.num_layers)):
      cache = caches.pop()
      dout,dw,db = affine_sigmoid_backward(dout,cache,(self.verbose,i))
      if ("w"+str(i)) in grads:
        grads["w"+str(i)] += dw
      else:
        grads["w"+str(i)] = dw +lamb * self.parameters["w"+str(i)]/N

      if ("b"+str(i)) in grads:
        grads["b"+str(i)] += db
      else:
        grads["b"+str(i)] = db

    return scores,loss,grads

  def __str__(self) -> str:
    for parameter in self.parameters:
      print(parameter,self.parameters[parameter])

    return ""

  def sgd(self, X, y, reg = 0, lr=1e-3, epochs=10):
    losses = []
    self.reset_network()
    for epoch in range(epochs):
      scores,loss,grads = self.loss(X, y, reg)
      losses.append(loss)
      self.update_params(grads,lr)
    
    return losses

  def predict(self,X):
    out = X
    N = X.shape[0]

    for i in range(self.num_layers):
      out,cache = affine_sigmoid_forward(out,self.parameters["w"+str(i)],self.parameters["b"+str(i)],(False,i))
    
    scores = out
    
    return np.argmax(scores,axis=1)    


def sigmoid_forward(x):
  out = 1/(1+np.exp(-x))

  return out,(out,)

def sigmoid_backward(dout,cache):
  sig_out, = cache
  dx = dout * sig_out * (1-sig_out)
  return dx

def affine_forward(x,w,b):
  out = np.dot(x,w) + b[None,:]
  return out,(x,w)

def affine_backward(dout,cache,verbose_args):
  x,w = cache
  verbose,i = verbose_args

  dx = np.dot(dout,w.T)
  if verbose:
    print("delta"+str(i+2),dout,sep="\n")
    print()
  dw = np.dot(dout.T,x).T
  db = np.sum(dout.T,axis=1)

  return dx,dw,db

def affine_sigmoid_forward(x,w,b,verbose_args):
  verbose,i = verbose_args
  out1,cache1 = affine_forward(x,w,b)
  out2,cache2 = sigmoid_forward(out1)
  if verbose:
    print("z"+str(i+2),out1,sep="\n")
    print("a"+str(i+2),np.concatenate([np.ones(out2.shape[0])[:,None],out2],axis = 1),sep="\n")
    print()
  # out2 = np.concatenate([np.ones(out2.shape[0])[:,None],out2],axis=1)
  return out2,(cache1,cache2)

def affine_sigmoid_backward(dout,cache,verbose_args):
  cache1,cache2 = cache
  # dout = dout[:,1:]
  dout = sigmoid_backward(dout,cache2)
  dx,dw,db = affine_backward(dout,cache1,verbose_args)
  return dx,dw,db

def logistic_loss(x,y,verbose):
  loss_per_class = -y * np.log(x) -(1-y)* np.log(1-x)
  loss_per_class[y==0] = (-(1-y)* np.log(1-x))[y==0]
  loss_per_class[y==1] = (-y * np.log(x))[y==1]
  loss_per_instance = np.sum(loss_per_class,axis=1)
  if verbose:
    print("J ",loss_per_instance[:,None],sep="\n")
    print()
  loss = np.average(loss_per_instance)
  N, = loss_per_instance.shape
  dloss_per_instance = np.full_like(loss_per_instance,1/N)
  dloss_per_class = np.dot(dloss_per_instance[:,None],np.ones((1,loss_per_class.shape[1]))) 
  dx = dloss_per_class * (-y/x + (1-y)/(1-x))
  # dx[y==0] = ((1-y)/(1-x))[y==0]
  # dx[y==1] = (-y/x)[y==1]
  return loss,dx
