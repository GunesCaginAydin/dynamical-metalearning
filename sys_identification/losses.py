import torch
try:
    from architectures.diffuser.diffuser_sim import *
except:
    from Sys_Identification.architectures.diffuser.diffuser_sim import * # type: ignore

class defaultloss():
    """
    Loss object for parsing and calculating losses incurred from different models,

    Parameters
    ---
        weights (dict) : multiplies dimension i of observation loss by c
        args (dict) : arguments parsed by the user

    """
    def __init__(self, args, weights=None):
        self.weights = weights
        self.args = args

    def mse_loss(self, ysim, yact, reduction='mean'):
        """
        Implements torch.nn.Functional.mse_loss - mse loss:

        Parameters
        ---
            ysim (torch.Tensor): predicted value
            yact (torch.Tensor): actual value

        Returns
        ---
            loss (torch.Tensor) mse loss
        """
        if self.weights is None:
            return torch.nn.functional.mse_loss(yact, ysim, reduction=reduction)
        else:
            return (torch.nn.functional.mse_loss(yact, ysim, reduction='none')*self.weights).mean()

    def mae_loss(self, ysim, yact, reduction='mean'):
        """
        Implements torch.nn.Functional.l1_loss - mae/l1 loss:

        Parameters
        ---
            ysim (torch.Tensor): predicted value
            yact (torch.Tensor): actual value

        Returns
        ---
            loss (torch.Tensor) mae loss
        """
        if self.weights is None:
            return torch.nn.functional.l1_loss(ysim, yact, reduction=reduction)
        else:
            return (torch.nn.functional.l1_loss(yact, ysim, reduction='none')*self.weights).mean()

    def huber_loss(self, ysim, yact, reduction='mean'):
        """
        Implements torch.nn.Functional.huber_loss - huber loss:

        Parameters
        ---
            ysim (torch.Tensor): predicted value
            yact (torch.Tensor): actual value

        Returns
        ---
            loss (torch.Tensor) huber loss
        """
        if self.weights is None:
            return torch.nn.functional.huber_loss(ysim, yact, reduction=reduction)
        else:
            return (torch.nn.functional.huber_loss(yact, ysim, reduction='none')*self.weights).mean()

    def rmse_loss(self, ysim, yact, reduction='mean'):
        """
        Implements l2 loss - l2 loss:

        Parameters
        ---
            ysim (torch.Tensor): predicted value
            yact (torch.Tensor): actual value

        Returns
        ---
            loss (torch.Tensor) rmse loss
        """
        if self.weights is None:
            mse_error = torch.nn.functional.mse_loss(ysim, yact, reduction=reduction)
            return torch.sqrt(mse_error)
        else:
            return (torch.sqrt(torch.nn.functional.mse_loss(yact, ysim), reduction='none')*self.weights).mean()

    def logcosh_loss(self, ysim, yact, reduction='mean'):
        """
        Implements logcosh loss - logcosh loss:

        Parameters
        ---
            ysim (torch.Tensor): predicted value
            yact (torch.Tensor): actual value

        Returns
        ---
            loss (torch.Tensor) logcosh loss
        """
        if self.weights is None:
            if reduction=='mean':
                return torch.mean(torch.mean(torch.log(torch.cosh(ysim - yact)),axis=0))
            else:
                return torch.log(torch.cosh(ysim - yact))
        else:
            return (torch.log(torch.cosh(ysim - yact))*self.weights).mean()

    def getloss(self, ysim, yact, reduction='mean'):
        """
        Determines the loss type to be used in training and validation according
        to the user defined args

        Available Loss Functions
        ---
        * mae (l1) loss

        * mse (l2) loss

        * huber loss

        * logcosh loss

        * rmse loss

        Parameters
        ---
            ysim (torch.Tensor): predicted value
            yact (torch.Tensor): actual value

        Returns
        ---
            loss (torch.Tensor) loss from passed arguments
        """
        if self.args.loss_function =='MSE':
            loss = self.mse_loss(ysim, yact, reduction=reduction)

        elif self.args.loss_function =='MAE':
            loss = self.mae_loss(ysim, yact, reduction=reduction)

        elif self.args.loss_function =='RMSE':
            loss = self.rmse_loss(ysim, yact, reduction=reduction)

        elif self.args.loss_function =='logcosh':
            loss = self.logcosh_loss(ysim, yact, reduction=reduction)

        elif self.args.loss_function =='Huber':
            loss = self.huber_loss(ysim, yact, reduction=reduction)
        
        return loss