import torch
import numpy as np
import torch.nn as nn
from torch.autograd import Variable


def _concat(xs):
  return torch.cat([x.view(-1) for x in xs])


class Weights_Update(object):
  def __init__(self, model, args):
    self.arch_momentum = args.momentum
    self.arch_weight_decay = args.arch_weight_decay
    self.model = model
    self.optimizer = torch.optim.Adam(self.model.parameters(),
        lr=args.learning_rate, betas=(0.5, 0.999), weight_decay=args.weight_decay)
    self.args =args

  def _compute_unrolled_model(self, input, target, eta, arch_optimizer):
    # alpha = alpha - eta* Dtheta_{alpha}(train datasets)
    loss = self.model._loss(input, target)
    theta = _concat(self.model.arch_parameters()).data
    try:
      moment = _concat(arch_optimizer.state[v]['momentum_buffer'] for v in self.model.arch_parameters()).mul_(self.arch_momentum)
    except:
      moment = torch.zeros_like(theta)
    dtheta = _concat(torch.autograd.grad(loss, self.model.arch_parameters())).data + self.arch_weight_decay*theta
    unrolled_model = self._construct_model_from_theta(theta.sub(eta, moment+dtheta))
    return unrolled_model

  def step(self, input_train, target_train, input_valid, target_valid, eta, arch_optimizer, unrolled):
    self.optimizer.zero_grad()
    if unrolled:
        self._backward_step_unrolled(input_train, target_train, input_valid, target_valid, eta, arch_optimizer)
    else:
        self._backward_step(input_valid, target_valid)
    nn.utils.clip_grad_norm(self.model.parameters(), self.args.grad_clip)
    self.optimizer.step()

  def _backward_step(self, input_valid, target_valid):
    loss = self.model._loss(input_valid, target_valid)
    loss.backward()

  def _backward_step_unrolled(self, input_train, target_train, input_valid, target_valid, eta, arch_optimizer):
    unrolled_model = self._compute_unrolled_model(input_train, target_train, eta, arch_optimizer)
    #obtain the alpha_new(alpha') on train dataset;
    unrolled_loss = unrolled_model._loss(input_valid, target_valid)
    unrolled_loss.backward()
 
    # obtain the Dalpha_{val}(weights) first term
    dalpha = [v.grad for v in unrolled_model.parameters()] 
    # obtain the Dalpha_{val}{alpha'}, deta
    vector = [v.grad.data for v in unrolled_model.arch_parameters()]

    implicit_grads = self._hessian_vector_product(vector, input_train, target_train)

    for g, ig in zip(dalpha, implicit_grads):
      g.data.sub_(eta, ig.data) #first  term - second term

    for v, g in zip(self.model.parameters(), dalpha):
      if v.grad is None:
        v.grad = Variable(g.data)
      else:
        v.grad.data.copy_(g.data)

  def _construct_model_from_theta(self, theta):
    model_new = self.model.new()
    # model_dict = self.model.state_dict()

    params, offset = {}, 0
    # for k, v in self.model.named_parameters():
    #   v_length = np.prod(v.size())
    #   params[k] = theta[offset: offset+v_length].view(v.size())
    #   offset += v_length

     
    v_length = np.prod(self.model.alphas_normal.view(-1).size())
    model_new.alphas_normal.data = theta[offset:offset+v_length].view(v_length).copy()
    offset += v_length
    
    v_length = np.prod(self.model.alphas_reduce.view(-1).size())
    model_new.alphas_reduce.data = theta[offset:offset+v_length].view(v_length).copy()
    assert offset == len(theta)
    model_new._arch_parameters=[model_new.alphas_normal,model_new.alphas_reduce]
    # model_dict.update(params)
    # model_new.load_state_dict(model_dict)
    return model_new.cuda()

  def _hessian_vector_product(self, vector, input, target, r=1e-2):
    R = r / _concat(vector).norm()
    for p, v in zip(self.model.arch_parameters(), vector):
      p.data.add_(R, v)#  alpha'= alpha+r*deta 
    loss = self.model._loss(input, target)
    grads_p = torch.autograd.grad(loss, self.model.parameters())

    for p, v in zip(self.model.arch_parameters(), vector):
      p.data.sub_(2*R, v) #alpha''=alpha'-2*r*deta
    loss = self.model._loss(input, target)
    grads_n = torch.autograd.grad(loss, self.model.parameters())

    for p, v in zip(self.model.arch_parameters(), vector):
      p.data.add_(R, v) #alpha=alpha''+r*deta; reset to the original arch_parameters;

    return [(x-y).div_(2*R) for x, y in zip(grads_p, grads_n)]

