import torch
import torch.nn as nn

class Perturbation(nn.Module):
    """ Network layer for performing perturbations. """
    def __init__(self, output_dim, dtype):
        super(Perturbation, self).__init__()

        #copy output from previous layer
        weights = torch.ones(output_dim, output_dim)
        self.weights = nn.Parameter(weights)
        bias = torch.zeros(output_dim)
        self.bias = nn.Parameter(bias)
        self.dtype = dtype

        for p in self.parameters():
            p.requires_grad = False
            # print(p)

    def forward(self, input, perturb):
        #perform perturbation through transformation
        
        out_t = torch.matmul(perturb, torch.unsqueeze(input, -1))
        out = torch.squeeze(out_t)

        return out
        
class RNN_single(nn.Module):
    def __init__(self, n_inputs, n_outputs, n_neurons_l1, alpha, 
                 dtype, nonlin, p_intermodule=1., noise=None, freeze_input = False, freeze_rec = False):
        super(RNN_single, self).__init__()

        self.n_neurons_l1 = n_neurons_l1
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.alpha = alpha
        self.nonlin = nonlin
        self.dtype = dtype
        self.freeze_input = freeze_input
        self.freeze_rec = freeze_rec

        if noise is None:
            self.noisy = False
            self.noise_amp = 0
        else:
            self.noisy = True
            self.noise_amp = noise

        self.rnn_l1 = nn.RNN(n_inputs, n_neurons_l1, num_layers=1,
                                nonlinearity=nonlin,bias=False) 

        self.noarm_output = nn.Linear(n_neurons_l1, n_outputs)

        #freeze output weights
        self.noarm_output.weight.requires_grad = False
        self.noarm_output.bias.requires_grad = False

        #freeze input weights
        if freeze_input:
            self.rnn_l1.weight_ih_l0.requires_grad = False
        if freeze_rec:
            self.rnn_l1.weight_hh_l0.requires_grad = False
        
        if self.n_outputs > 1:
            self.perturb_output = Perturbation(n_outputs, self.dtype)

    
    def save_parameters(self):
        wihl1 = self.rnn_l1.weight_ih_l0.cpu().detach().numpy().copy()
        whhl1 = self.rnn_l1.weight_hh_l0.cpu().detach().numpy().copy()
        wout = self.noarm_output.weight.cpu().detach().numpy().copy()
        bout = self.noarm_output.bias.cpu().detach().numpy().copy()

        dic = {'wihl1':wihl1,'whhl1':whhl1,
               'wout':wout,'bout':bout,
                }
        return dic
    
    def init_hidden(self, batch_size = None):
        if batch_size is None:
            batch_size = self.batch_size
        # needs to be small !! as rates are regularized during training -> so going small
        return ((torch.rand(1,batch_size, self.n_neurons_l1)-0.5)*0.2).type(self.dtype)

    def get_hidden(self, X):
        tsteps, self.batch_size, _ = X.shape

        # initial activity
        hidden1 = self.init_hidden()
        x1 = hidden1
        r1 = self.activate(x1)

        #only return hidden state before activation
        prehiddenl1 = torch.zeros(tsteps, self.batch_size, self.n_neurons_l1).type(self.dtype)
        
        for j in range(tsteps): #for each time step
            x1,r1 = self.f_step(X[j],x1,r1)
            prehiddenl1[j] = x1
        
        return prehiddenl1

    def forward(self, X, perturb, r1_input = None, hidden1 = None):
          
        tsteps, self.batch_size, _ = X.shape

        # initial activity
        if hidden1 is None:
            hidden1 = self.init_hidden()
        x1 = hidden1
        r1 = self.activate(x1)

        outv = torch.zeros(tsteps, self.batch_size, self.n_outputs).type(self.dtype) #initial output
        outp = torch.zeros(tsteps, self.batch_size, self.n_outputs).type(self.dtype) #perturbed output
        hiddenl1 = torch.zeros(tsteps, self.batch_size, self.n_neurons_l1).type(self.dtype)
        
        for j in range(tsteps): #for each time step
            x1,r1 = self.f_step(X[j],x1,r1)
            
            hiddenl1[j] = r1
            outv[j] = self.noarm_output(r1)

            if self.n_outputs > 1:
                outp[j] = self.perturb_output(outv[j],perturb[j])
            else:
                outp[j] = outv[j]
        return outv, outp, hiddenl1

    def force_training(self, X, perturb, P, output_weights_inv, target,
    stimt=None, perturbt=None, targett=None, criterion = None):
          
        tsteps, batch_size, _ = X.shape

        # initial activity
        hidden1 = self.init_hidden(batch_size)
        x1 = hidden1
        r1 = self.activate(x1)

        outv = torch.zeros(tsteps, batch_size, self.n_outputs).type(self.dtype) #initial output
        outp = torch.zeros(tsteps, batch_size, self.n_outputs).type(self.dtype) #perturbed output
        hiddenl1 = torch.zeros(tsteps, batch_size, self.n_neurons_l1).type(self.dtype)

        # update a copy of the weights
        weights = self.rnn_l1.weight_hh_l0.clone().detach()
        orig_weights = weights.clone()

        errors = []
        mean_dw = []
        for j in range(tsteps): #for each time step

            x1,r1 = self.f_step(X[j],x1,r1, batch_size = batch_size) 
   
            hiddenl1[j] = r1

            outv[j] = self.noarm_output(r1)
            if self.n_outputs > 1:
                outp[j] = self.perturb_output(outv[j],perturb[j])
        
            # Now for the RLS algorithm:
            # compute errors
            err = (outp[j] - target[j]).squeeze()
            err = output_weights_inv @ err
            r1_col = r1.reshape((r1.shape[2],1)) #n x p
            pr = (P @ r1_col).squeeze() #n x p
            norm = (1. + (r1_col.T @ pr.unsqueeze(1))).squeeze()
            P -= torch.outer(pr, pr) / norm

            #update weights
            for i in range(self.n_neurons_l1):
                weights[i] -= err[i] * pr / norm
            self.rnn_l1.weight_hh_l0.copy_(weights)

        total_weight_changes = weights - orig_weights
        mean_total_dw = torch.abs(total_weight_changes).mean().detach().item()

        return outv, outp, hiddenl1, P, mean_total_dw
    
    def activate(self, x1):
        if self.nonlin == 'relu':
            r1 = x1.relu()
        elif self.nonlin == 'tanh':
            r1 = x1.tanh()
        else:
            raise Exception('unknown nonlinearity')
        return r1

    def f_step(self,xin,x1,r1, single_step = False, batch_size = None):

        if single_step:
            self.batch_size = 1

        if batch_size is None:
            batch_size = self.batch_size

        if self.noisy:
            nx1 = self.noise_amp*torch.randn(1,batch_size, self.n_neurons_l1).type(self.dtype)
            x1 = x1 + self.alpha*(-x1 + r1 @ self.rnn_l1.weight_hh_l0.T 
                                      + xin @ self.rnn_l1.weight_ih_l0.T
                                      + nx1
                                 )                                                    
        else:
            x1 = x1 + self.alpha*(-x1 + r1 @ self.rnn_l1.weight_hh_l0.T 
                                      + xin @ self.rnn_l1.weight_ih_l0.T
                                 )

        r1 = self.activate(x1)

        return x1,r1