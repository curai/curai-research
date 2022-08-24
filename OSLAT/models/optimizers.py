import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_
import pdb

class Optimizer(object):
    """
    Controller class for optimization. Mostly a thin
    wrapper for `optim`, but also useful for implementing
    rate scheduling beyond what is currently available.
    Also implements necessary methods for training RNNs such
    as grad manipulations.

    Args:
      method (:obj:`str`): one of [sgd, adagrad, adadelta, adam]
      lr (float): learning rate
      lr_decay (float, optional): learning rate decay multiplier
      start_decay_steps (int, optional): step to start learning rate decay
      beta1, beta2 (float, optional): parameters for adam
      adagrad_accum (float, optional): initialization parameter for adagrad
      decay_method (str, option): custom decay options
      warmup_steps (int, option): parameter for `noam` decay

    We use the default parameters for Adam that are suggested by
    the original paper https://arxiv.org/pdf/1412.6980.pdf
    These values are also used by other established implementations,
    e.g. https://www.tensorflow.org/api_docs/python/tf/train/AdamOptimizer
    https://keras.io/optimizers/
    Recently there are slightly different values used in the paper
    "Attention is all you need"
    https://arxiv.org/pdf/1706.03762.pdf, particularly the value beta2=0.98
    was used there however, beta2=0.999 is still arguably the more
    established value, so we use that here as well
    """

    def __init__(self, method, learning_rate, max_grad_norm,
                 lr_decay=1, start_decay_steps=None, decay_steps=None,
                 beta1=0.9, beta2=0.999,
                 adagrad_accum=0.0,
                 decay_method=None,
                 warmup_steps=4000
                 ):
        self.last_ppl = None
        self.learning_rate = learning_rate
        self.original_lr = learning_rate
        self.max_grad_norm = max_grad_norm
        self.method = method
        self.lr_decay = lr_decay
        self.start_decay_steps = start_decay_steps
        self.decay_steps = decay_steps
        self.start_decay = False
        self._step = 0
        self.betas = [beta1, beta2]
        self.adagrad_accum = adagrad_accum
        self.decay_method = decay_method
        self.warmup_steps = warmup_steps

    def set_parameters(self, params, lr_lambda=None):
        """ ? """
        self.params = []
        for k, p in params:
            if p.requires_grad:
                self.params.append(p)

        if self.method == 'ADAM':
            self.optimizer = optim.Adam(self.params, lr=self.learning_rate,
                                        betas=self.betas, eps=1e-9)
        else:
            raise RuntimeError("Invalid optim method: " + self.method)
        
        if lr_lambda != None:
            self.lr_scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)
        elif self.decay_method == "noam":
            lr_lambda = lambda step: min((step + 1) ** (-0.5), (step + 1) * self.warmup_steps**(-1.5))
            self.lr_scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)


    def _set_rate(self, learning_rate):
        self.learning_rate = learning_rate
        if self.method != 'sparseadam':
            self.optimizer.param_groups[0]['lr'] = self.learning_rate
        else:
            for op in self.optimizer.optimizers:
                op.param_groups[0]['lr'] = self.learning_rate

    def step(self):
        """Update the model parameters based on current gradients.

        Optionally, will employ gradient modification or update learning
        rate.
        """
        self._step += 1

        if self.max_grad_norm:
            clip_grad_norm_(self.params, self.max_grad_norm)

        self.optimizer.step()
        self.lr_scheduler.step()


def build_optim(args, model, checkpoint=None):
    """ Build optimizer """
    optim = Optimizer(
        args.optim, args.lr, args.max_grad_norm,
        beta1=args.beta1, beta2=args.beta2,
        decay_method=args.decay_method,
        warmup_steps=args.warmup_steps)

    optim.set_parameters(list(model.named_parameters()))

    if checkpoint != None:
        optim.optimizer.load_state_dict(checkpoint['optim'])
        optim.lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        if args.device != 'cpu':
            for state in optim.optimizer.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = v.to(args.device)

        if (optim.method == 'adam') and (len(optim.optimizer.state) < 1):
            raise RuntimeError(
                "Error: loaded Adam optimizer from existing model" +
                " but optimizer state is empty")

    return optim
