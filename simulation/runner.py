import torch
import numpy as np
import math
from simulation.networks import RNN_single
import torch.nn as nn
import torch.optim as optim
from simulation.repertoire_data import Repertoire_Dataset
from config_manager import base_configuration
from constants import Constants
from collections import OrderedDict

class Runner:
    """ Object to run simulations ."""
    
    def __init__(self, config: base_configuration.BaseConfiguration, proj_dir: str, training=True) -> None:
        """ 
        Class for running simulations.

        Parameters
        ----------
        config: configuration object specifying experiment setup.
        proj_dir: project directory
        training: if you are training the model

        """
        self.PROJ_DIR = proj_dir + '/'
        self._config = config
        
        if training:
            self._config.save_configuration(folder_path=self.PROJ_DIR +self._config.outdir)

        self._outdir = self.PROJ_DIR + self._config.outdir
        self._setup(training)

    def _setup(self, training:bool):
        """ 
        Method to set up device, data, and model.
        """
        if torch.cuda.is_available():
            self.dtype = torch.cuda.FloatTensor
            self.device = torch.device("cuda:{}".format(self._config.gpu_id))
            try:
                torch.cuda.set_device(self._config.gpu_id)
            except:
                pass
#             print("device", torch.cuda.current_device())
        else:
            self.dtype = torch.FloatTensor
            self.device = torch.device("cpu")
        self.data = Repertoire_Dataset(self.PROJ_DIR + self._config.datadir, training=training)
        self.model, self.criterion, self.optimizer = self._create_model()

    def run_train(self):
        """ 
        Set up and train a model. 
        """
        #set up perturbation information
        self.trial_pert_params, self.pert_transforms = self._setup_perturbation(self.data.ntrials)

        # set up params/weights for model and optimizer
        if self._config.type == Constants.INIT_TRAINING:
            self.model = self._create_and_initialize_weights(self.model)
        elif self._config.type == Constants.PERTURBATION:
            self.model = self._load_model(self.PROJ_DIR + self._config.training_dir, self.model)
            # self.optimizer = self._load_optimizer(self.PROJ_DIR + self._config.training_dir, self.optimizer)
        else:
            raise ValueError("Unknown simulation type")

        #get and save init parameters
        params0 = self.model.save_parameters()

        #train
        if self._config.optimizer == 'FORCE':
            lc = self.train_force()
        else:
            lc = self.train()

        # get and save final parameters
        params1 = self.model.save_parameters()

        # save parameters
        dic = {'params0':params0,'params1':params1,'lc':np.array(lc)}
        np.save(self._outdir+'training',dic)

        #check implementation
        import matplotlib.pyplot as plt
        plt.figure()
        plt.plot(np.array(lc)[:,0])
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.savefig(self._outdir + "loss.png")

    # def get_hidden_from_test(self, epoch=None):
    #     """ 
    #     Set up and test an existing model. 

    #     Parameters
    #     ----------
    #     epoch: int
    #         epoch that model was saved at during training

    #     Returns
    #     ---------- 
    #     hidden1: np array 
    #         hidden states before activation

    #     """
    #     model = self._load_model(self._outdir, self.model, epoch)
    #     model.eval()

    #     # get data and perturbation info
    #     stimulus = self.data.stimulus

    #     # set up in pytorch form
    #     stimt = torch.Tensor(stimulus.transpose(1,0,2)).type(self.dtype) #to tstep, trial, nstim
 
    #     with torch.no_grad():
    #         # run model
    #         hiddenl1 = model.get_hidden(stimt)

    #     hidden1 = hiddenl1.cpu().detach().numpy().transpose(1,0,2)
        
    #     return hidden1

    # def run_test_hidden(self, stimt, perturbt, hidden1):
    #     model = self._load_model(self._outdir, self.model, epoch)
    #     model.eval()

    #     with torch.no_grad():
    #         # run model
    #         if self._config.network == 'RNN_single':
    #             testout_pre, testout,testl1 = model(stimt, perturbt, hidden1 = hidden1)
    #         else:
    #             raise ValueError("network not implemented")

    #     # save it
    #     output_pre = testout_pre.cpu().detach().numpy().transpose(1,0,2)
    #     output = testout.cpu().detach().numpy().transpose(1,0,2)
    #     activity1 = testl1.cpu().detach().numpy().transpose(1,0,2)
        
    #     if self._config.network == 'RNN_arm':
    #         output[:,:,1] = output[:,:,1] + Constants.ARM_YOFFSET 

    #     return self._config.datadir, output_pre, output, activity1, activity2


    def run_test(self, epoch:int=None, r1_input:np.array=None):
        """ 
        Set up and test an existing model. 

        Parameters
        ----------
        epoch: epoch that model was saved at during training
        r1_input: pre-defined activity for the RNN (trials x tsteps x neurons)

        Returns
        ---------- 
        datadir: str
            directory where data is stored
        output_pre: np array (trials x tsteps x noutputs)
            model output before perturbation
        output: np array (trials x tsteps x noutputs)
            model output after perturbation   
        activity1: np array (trials x tsteps x neurons)
            activity for the RNN

        """
        model = self._load_model(self._outdir, self.model, epoch)
        model.eval()

        stimt, _, perturbt = self._setup_testdata(self.data)

        if r1_input is not None:
            r1t = torch.Tensor(r1_input.transpose(1,0,2)).type(self.dtype)
        else:
            r1t = None

        with torch.no_grad():
            # run model
            testout_pre, testout,testl1 = model(stimt, perturbt, r1_input = r1t)

        # save it
        output_pre = testout_pre.cpu().detach().numpy().transpose(1,0,2)
        output = testout.cpu().detach().numpy().transpose(1,0,2)
        activity1 = testl1.cpu().detach().numpy().transpose(1,0,2)

        return self._config.datadir, output_pre, output, activity1

    def _setup_testdata(self, test_data:Repertoire_Dataset):
        """ 
        Set up test data for the model.
        
        Parameters
        ----------
        test_data: dataset to test model on

        Returns
        ----------
        stimt: torch tensor (tsteps x trials x nstim)
            stimulus
        targett: torch tensor (tsteps x trials x noutputs)
            target output
        perturbt: torch tensor (tsteps x trials x 2 x 2)
            perturbation matrix (for rotation)
        """

        # get data and perturbation info
        stimulus = test_data.stimulus
        target = test_data.target_output

        _, pert_transforms = self._setup_perturbation(test_data.ntrials)
            
        # set up in pytorch form
        stimt = torch.Tensor(stimulus.transpose(1,0,2)).type(self.dtype) #to tstep, trial, nstim
        perturbt = torch.zeros(stimt.shape[0], stimt.shape[1],2,2).type(self.dtype)
        targett = torch.Tensor(target.transpose(1,0,2)).type(self.dtype)

        for i in range(test_data.ntrials):
            perturbt[:,i,] = torch.tensor(pert_transforms[i]) 
        
        return stimt, targett, perturbt


    def _test_current_model(self, stimt:torch.Tensor, perturbt:torch.Tensor, targett:torch.Tensor):
        """ 
        Test current model during training. 
        
        Parameters
        ----------
        stimt: stimulus (tsteps x trials x nstim)
        perturbt: perturbation matrix (tsteps x trials x 2 x 2)
        targett: target output (tsteps x trials x noutputs)

        Returns
        ----------
        error: torch tensor
            mean error between model output and target output
        """
        self.model.eval()

        # run model
        with torch.no_grad():
            _, testout,_ = self.model(stimt, perturbt)

        # calculate error
        error = self.criterion(testout[50:], targett[50:]) # only look at time points > 50dt  

        return error.mean()

    def _load_model(self, dir:str, model, epoch:str=None):
        """ 
        Load state parameters in model.

        Parameters
        ----------
        dir: directory where model (parameters) is saved
        model: initial model
        epoch: epoch that model was saved at during training

        Returns
        -------
        model: pytorch model
            model with trained parameters

        """
        epoch_str = ('_'+str(epoch)) if epoch is not None else ""
        try:
            temp = torch.load(dir+'model'+ epoch_str)['model_state_dict']
        except: # make sure model is loaded in available device
            temp = torch.load(dir+'model'+ epoch_str, map_location='cuda:0')['model_state_dict']

        model.load_state_dict(temp, strict = True)

        return model

    def _load_optimizer(self, dir:str, optimizer:optim, epoch:int = None):
        """ 
        Load state parameters in optimizer

        Parameters
        ----------
        dir: directory where model is saved
        optimizer: initial optimizer
        epoch: epoch that model was saved at during training

        Returns
        -------
        optimizer: Pytorch optimizer
            optimizer with trained parameters

        """
        if optimizer is not None:
            epoch_str = ('_'+str(epoch)) if epoch else ""   
            try:
                temp = torch.load(dir+'model'+ epoch_str)
            except:
                temp = torch.load(dir+'model'+ epoch_str, map_location='cuda:0')

            optimizer.load_state_dict(temp['optimizer_state_dict'])

            # use new specified parameters 
            for g in optimizer.param_groups:
                g['lr'] = self._config.lr

        return optimizer

    def _setup_perturbation(self, ntrials:int):
        """ 
        Set up perturbation parameters and transformations.
        
        Parameters
        ----------
        ntrials: number of trials in simulation
        target_param: direction of target (for rotation) for each trial
        
        Returns
        -------
        trial_pert_params: np array (ntrials)
            perturbation parameters for each trial
        pert_transforms: np array (ntrials, 2, 2)
            transformation matrices to apply to each trial to model perturbation
        """
        # for visuomotor rotation
        if self._config.perturbation == Constants.PERT_ROTATION:
            perturbation = self._config.perturbation
            pert_params = self._config.pert_params

            #TODO: simplify for only 1 pert param
            # transform pert parameters 
            pert_params = [math.radians(p) for p in pert_params]
            
            # set pert param for n trials
            npert_trials = math.ceil(ntrials/len(pert_params))
            trial_pert_params = torch.tensor(pert_params)
            trial_pert_params = trial_pert_params.repeat(npert_trials)[:ntrials]

            # calculate transformations needed for perturbations
            pert_transforms = np.array([self._calc_rotation_perturbation(angle) for angle in trial_pert_params])

        else:
            # no transformations needed for no perturbation
            pert_params = [0]
            trial_pert_params = np.zeros(ntrials)
            pert_transforms = np.zeros([ntrials,2,2])
            pert_transforms[:,] = [[1,0],[0,1]]

        return trial_pert_params, pert_transforms

    def _calc_rotation_perturbation(self, angle:float):
        """ 
        Calculate transformation for rotation perturbations.

        Parameters
        ----------
        angle: angle of rotation (in radians)

        Returns
        -------
        np array (2,2)
            transformation matrix for rotation perturbation
        """
        return [[math.cos(angle), -math.sin(angle)],[math.sin(angle), math.cos(angle)]]

    def _create_model(self):
        """ 
        Create new model, criterion, and optimizer. 

        Returns
        -------
        model: Pytorch model
            network model
        criterion: Pytorch criterion
            criterion for calculating the loss
        optimizer: Pytorch optimizer
            optimizer for training

        """
        # create model
        if self._config.network == 'RNN_single':
            model = RNN_single(self.data.input_dim, self.data.output_dim, 
                        self._config.n1, 
                        self.data.dt/self._config.tau, 
                        self.dtype, self._config.nonlin, 
                        noise= self._config.noise,
                        freeze_input = self._config.freeze_input,
                        freeze_rec = self._config.freeze_rec)
        else:
            raise ValueError('unknown network')
        
        if self.dtype == torch.cuda.FloatTensor:
            model = model.cuda()
        
        # define loss function
        criterion = nn.MSELoss(reduction='none')

        # create optimizer
        if self._config.optimizer == 'Adam':
            optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                            lr=self._config.lr)
        elif self._config.optimizer == 'SGD':
            optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()),
                            lr=self._config.lr)
        elif self._config.optimizer == 'FORCE':
            optimizer = None
        else:
            raise Exception('unknown optimizer')

        return model, criterion, optimizer

    def _create_and_initialize_weights(self, model):
        """ 
        Create and initialize new weights. 
        
        Parameters
        ----------
        model: Pytorch model
        
        Returns
        -------
        model: Pytorch model
            model with randomly initialized weights
        """

        # number of neurons
        n1 = self._config.n1

        # initialize weights
        state_dict = model.state_dict()

        ## input weights
        state_dict['rnn_l1.weight_ih_l0'] = torch.FloatTensor((np.random.rand(n1, self.data.input_dim)-0.5)*2.*self._config.gin) #Win

        ## recurrent weights
        state_dict['rnn_l1.weight_hh_l0'] = torch.FloatTensor(self._config.g1 / np.sqrt(n1) * np.random.randn(n1,n1)) #PMd
     
        ## output weights
        state_dict['noarm_output.weight'] = torch.FloatTensor((np.random.rand(self.data.output_dim, n1)-0.5)*2.*self._config.gout)
        state_dict['noarm_output.bias'] = torch.FloatTensor(np.zeros(self.data.output_dim))

        model.load_state_dict(state_dict, strict=True)

        return model

    def _setup_trainingdata(self, stimulus:np.array, target_output:np.array, pert_transforms:np.array):
        """ 
        Set up training data in pytorch form. 

        Parameters
        ----------
        stimulus: model input (trials, tsteps, input_dim)
            model input
        target_output: target output trajectories (trials, tsteps, output_dim)
        pert_transforms: matrix transformations for each trial to model perturbations (trials, 2, 2)
        
        Returns
        -------
        stim: torch tensor (training trials, tsteps, batchsize, input_dim)
            model input for each training trial
        target: torch tensor (training trials, tsteps, batchsize, output_dim)
            target output for each training trial
        perturb: torch tensor (training trials, tsteps, batchsize, 2, 2)
            perturbation matrices for each training trial
        """
            
        # convert stimulus and target to pytorch form
        stim = torch.zeros(self._config.training_trials, self.data.tsteps, self._config.batch_size, self.data.input_dim).type(self.dtype)
        target = torch.zeros(self._config.training_trials, self.data.tsteps, self._config.batch_size, self.data.output_dim).type(self.dtype)
        perturb = torch.zeros(self._config.training_trials, self.data.tsteps, self._config.batch_size, 2,2).type(self.dtype)

        for j in range(self._config.training_trials):
            idx = np.random.choice(range(self.data.ntrials), self._config.batch_size, replace=False)

            stim[j] = torch.Tensor(stimulus[idx].transpose(1,0,2)).type(self.dtype)
            target[j] = torch.Tensor(target_output[idx].transpose(1,0,2)).type(self.dtype)
            perturb[j] = torch.tensor([pert_transforms[idx]]).type(self.dtype)
         
        return stim, target, perturb
 
    def train_force(self):
        """ 
        Train a model using FORCE. 

        Returns
        -------
        lc: list
            loss during training
        """
        stim, target, perturb = self._setup_trainingdata(self.data.stimulus, self.data.target_output, self.pert_transforms)
        self.test_data = Repertoire_Dataset(self.PROJ_DIR + self._config.datadir, training=False)
        stimt, targett, perturbt = self._setup_testdata(self.test_data)
        lc = [] # save loss
        
        #no gradients needed in force training
        with torch.no_grad():

            # initialize the inverse correlation matrix
            P = torch.eye(self._config.n1)/self._config.lr
            P = P.type(self.dtype)

            # get inverse of output weights
            output_weights_inv = torch.linalg.pinv(self.model.noarm_output.weight.clone().detach())

            for epoch in range(self._config.training_trials):

                train_running_loss = 0.0

                # one training step
                if self._config.network == 'RNN_single':
                    _,output, _, P, mean_total_dw = self.model.force_training(
                        stim[epoch], perturb[epoch], 
                        P, output_weights_inv, target[epoch],
                        stimt = stimt, perturbt = perturbt, targett= targett, 
                        criterion = self.criterion)
                else:
                    raise ValueError('network not implemented yet')

                #calculate loss
                loss = self.criterion(output[50:], target[epoch,50:]) # only look at time points > 50dt  
                loss_train = loss.mean() 

                #calculate testing error
                test_error = self._test_current_model(stimt, perturbt, targett)
                #TODO: simplify this
                train_running_loss = [loss_train.detach().item(), test_error.detach().item(), [], [], mean_total_dw]

                toprint = OrderedDict()
                toprint['Loss'] = loss_train
                toprint['Error'] = test_error
                toprint['Weight change'] = mean_total_dw
    
                self._log(epoch, loss_train, toprint) 
                    
                lc.append(train_running_loss) 
        
        return lc

    
    def train(self):
        """ 
        Train a model using pytorch optimizers. 

        Returns
        -------
        lc: list
            loss during training
        """

        torch.save({'epoch': -1,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict() if self.optimizer is not None else None,
                    }, self._outdir+'model_'+str(-1))

        stim, target, perturb = self._setup_trainingdata(self.data.stimulus, self.data.target_output, self.pert_transforms)
        lc = [] # save loss

        #training mode
        self.model.train()

        for epoch in range(self._config.training_trials):

            train_running_loss = 0.0

            # one training step
            if self.optimizer is not None:
                self.optimizer.zero_grad()
            
            if self._config.network == 'RNN_single':
                _,output,rl1 = self.model(stim[epoch], perturb[epoch])
            else:
                raise ValueError('unknown network')

            #calculate loss
            loss = self.criterion(output[50:], target[epoch,50:]) # only look at time points > 50dt  
            loss_train = loss.mean() 

            #add regularization
            # term 1: parameters
            reg1in = self._config.alpha1*self.model.rnn_l1.weight_ih_l0.norm(2)
            reg1rec = self._config.gamma1*self.model.rnn_l1.weight_hh_l0.norm(2)          
            reg1out = self._config.alpha1*self.model.noarm_output.weight.norm(2)
            # term 2: rates
            reg1act = self._config.beta1*rl1.pow(2).mean()

            #calculate overall loss
            loss = loss_train+reg1in+reg1rec+reg1out+reg1act
            loss.backward()

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self._config.clipgrad)
            self.optimizer.step()

            train_running_loss = [loss_train.detach().item()]
            # print('train_running_loss', train_running_loss)

            toprint = OrderedDict()
            toprint['Loss'] = loss_train
            toprint['R_l1in'] = reg1in
            toprint['R_l1rec'] = reg1rec
            toprint['R_out'] = reg1out
            toprint['R_l1rate'] = reg1act
            self._log(epoch, loss, toprint) 
                
            lc.append(train_running_loss) 
        
        torch.save({'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict() if self.optimizer is not None else None,
                    }, self._outdir+'model')

        return lc

    def _log(self, epoch:int, loss:float, toprint:OrderedDict):
        """ 
        Log and save information during training. 

        Parameters
        ----------
        epoch: current epoch of training
        loss: loss after regularization for current epoch
        toprint: other information to print

        """
        if epoch % Constants.PRINT_EPOCH == 0:
            print(('Epoch=%d | '%(epoch)) +" | ".join("%s=%.4f"%(k, v) for k, v in toprint.items()))

        # save model if epoch falls on interval or was specified
        save_model = False
        if ((self._config.log_interval is not None) and (epoch % self._config.log_interval == 0)) \
            or ((self._config.log_epochs is not None) and (epoch in self._config.log_epochs)):
                save_model = True
        
        if save_model:
            torch.save({'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict() if self.optimizer is not None else None,
                    }, self._outdir+'model_'+str(epoch)) 

        # save the final model after training
        if (epoch + 1) == self._config.training_trials:
            torch.save({'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict() if self.optimizer is not None else None,
                    }, self._outdir+'model') 
