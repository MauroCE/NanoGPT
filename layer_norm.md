# BatchNorm
Makes sure that across batch dimension, each neuron had unit gaussian distribution (mean 0, std 1). 
In the MakeMore series we wrote
```python
class BatchNorm1d:

    def __init__(self, dim, eps=1e-5, momentum=0.1):
        self.eps = eps
        self.momentum = momentum
        # the self.training attribute is actually present in Pytorch too because many layers have a different 
        # behavior based on whether you are during training or inference
        self.training = True
        # Parameters (trained with backprop)
        self.gamma = torch.ones(dim)
        self.beta = torch.zeros(dim)
        # Buffers (trained with a running momentum update)
        self.running_mean = torch.zeros(dim)
        self.running_var = torch.ones(dim)

    def __call__(self, x):
        # Calculate forward pass differently if we are in training or inference mode
        if self.training:
            # During training we estimate the from the batch
            xmean = x.mean(0, keepdim=True)
            xvar = x.var(0, keepdim=True)
        else:
            # During inference we use the running ones
            xmean = self.running_mean
            xvar = self.running_var
        xhat = (x - xmean) / torch.sqrt(xvar + self.eps)
        self.out = self.gamma*xhat + self.beta
        # Update buffers
        if self.training:
            with torch.no_grad():
                self.running_mean = (1 - self.momentum)*self.running_mean + self.momentum*xmean
                self.running_var = (1 - self.momentum)*self.running_var + self.momentum*xvar
        return self.out

    def parameters(self):
        return [self.gamma, self.beta]
```

# Layer norm
We simply do the same but this time we compute mean and std of each example, and standardise those. This means there is no interaction across batch and so we don't need a different behavior for training and test time, and we don't need to keep track of running mean and std.