import torch
from architectures.diffusion.diffuser_trial import TSDiffuser

class losses():
    def __init__(self, weights=None):
        self.weights = weights

    def getlossweights(self, discount, action_weight=1.0, weights_dic={}):
        '''
        sets loss coefficients for trajectory

        action_weight   : float
            coefficient on first action loss
        discount   : float
            multiplies t^th timestep of trajectory loss by discount**t
        weights_dict    : dict
            { i: c } multiplies dimension i of observation loss by c

                                    WIP***
        '''
        self.action_weight = action_weight

        dim_weights = torch.ones(self.transition_dim, dtype=torch.float32)

        if weights_dict is None: weights_dict = {}
        for ind, w in weights_dict.items():
            dim_weights[self.action_dim + ind] *= w

        discounts = discount ** torch.arange(self.horizon, dtype=torch.float)
        discounts = discounts / discounts.mean()
        self.loss_weights = torch.einsum('h,t->ht', discounts, dim_weights)

        self.loss_weights[0, :self.action_dim] = action_weight
        return self.loss_weights

    def mse_loss(self, ysim, yact):
        """
        Implements torch.nn.Functional.mse_loss - mse loss:
        """
        if not self.weights:
            return torch.nn.functional.mse_loss(yact, ysim)
        else:
            return torch.nn.functional.mse_loss(yact, ysim)*self.loss_weights

    def mae_loss(self, ysim, yact):
        """
        Implements torch.nn.Functional.l1_loss - mae/l1 loss:
        """
        if not self.weights:
            return torch.nn.functional.l1_loss(ysim, yact)
        else:
            return torch.nn.functional.l1_loss(yact, ysim)*self.loss_weights

    def huber_loss(self, ysim, yact):
        """
        Implements torch.nn.Functional.huber_loss - huber loss:
        """
        if not self.weights:
            return torch.nn.functional.huber_loss(ysim, yact)
        else:
            return torch.nn.functional.huber_loss(yact, ysim)*self.loss_weights

    def rmse_loss(self, ysim, yact):
        """
        Implements l2 loss - l2 loss:
        """
        if not self.weights:
            n = len(yact)
            mse_error = torch.nn.functional.mse_loss(ysim, yact)
            return (mse_error*n)**0.5
        else:
            return (torch.nn.functional.mse_loss(yact, ysim)*self.loss_weights*n)**0.5

    def logcosh_loss(self, ysim, yact):
        """
        Implements logcosh loss - logcosh loss:
        """
        if not self.weights:
            return torch.sum(torch.log(torch.cosh(ysim - yact)))
        else:
            return torch.sum(torch.log(torch.cosh(ysim - yact)))*self.loss_weights

    def getloss(self, args, ysim, yact, architecture='transformer'):
        """
        Determines the loss type to be used in training and validation according
        to the user defined args
        """
        if architecture=='transformer':
            pass
        elif architecture=='diffuser':
            yact=TSDiffuser.sample_from_noisy_distribution()
            ysim=TSDiffuser.sample_from_noisy_distribution()

        if args.loss_function =='MSE':
            loss = self.mse_loss(ysim, yact)

        elif args.loss_function =='MAE':
            loss = self.mae_loss(ysim, yact)

        elif args.loss_function =='RMSE':
            loss = self.rmse_loss(ysim, yact)

        elif args.loss_function =='LC':
            loss = self.logcosh_loss(ysim, yact)

        elif args.loss_function =='Huber':
            loss = self.huber_loss(ysim, yact)
        
        return loss