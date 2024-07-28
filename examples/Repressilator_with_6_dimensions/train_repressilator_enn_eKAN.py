
"""# Transformer-PhysX Repressilator System

"""

import sys
import os
import logging
import h5py
import torch
import torch.nn as nn
import numpy as np
import argparse
import matplotlib.pyplot as plt


from typing import Dict, List, Tuple
# Torch imports
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from torch.optim.lr_scheduler import ExponentialLR
# Trphysx imports
from trphysx.embedding import EmbeddingModel
from trphysx.config.configuration_phys import PhysConfig
from trphysx.embedding import EmbeddingTrainingHead
from trphysx.embedding.training import EmbeddingParser, EmbeddingDataHandler, EmbeddingTrainer
# import KAN
#from fastkan import FastKAN as KAN
from kan import KAN, KANLinear




Tensor = torch.Tensor
TensorTuple = Tuple[torch.Tensor]
FloatTuple = Tuple[float]

logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser(description='make sure to add the training and validation data paths, and the experiment name')
parser.add_argument('--exp_name', type=str,required=True ,default='repressilator', help='Experiment name')
parser.add_argument('--train', type=str, required=True, help='Path to training data')
parser.add_argument('--eval', type=str, required=True, help='Path to validation data')

arg = parser.parse_args()
exp_name = arg.exp_name
train_path = arg.train
eval_path = arg.eval


argv = []
#argv = argv + ["--exp_name", "rossler"]
argv = argv + ["--exp_name", exp_name]
argv = argv + ["--exp_dir", exp_name+"/outputs"]
argv = argv + ["--training_h5_file", train_path]
argv = argv + ["--eval_h5_file", eval_path]
argv = argv + ["--lr","0.001"]
argv = argv + ["--stride", "16"]
argv = argv + ["--batch_size", "256"]
argv = argv + ["--block_size", "16"]
argv = argv + ["--n_train", "10240"]
argv = argv + ["--n_eval", "64"]
argv = argv + ["--epochs", "300"]

# make directory with the experiment name and change to it
if not os.path.exists(exp_name):
    os.makedirs(exp_name)
os.chdir(exp_name)



# Setup logging
logging.basicConfig(
    filename='logging.log',
    filemode='a',
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.DEBUG,
    force=True,)

args = EmbeddingParser().parse(argv)
if(torch.cuda.is_available()):
    use_cuda = "cuda"
args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
logger.info("Torch device: {}".format(args.device))

"""## Repressilator Config Class"""

class RepressilatorConfig(PhysConfig):
    """
    This is the configuration class for the modeling of the Repressilator system.
    """
    # same model as rossler
    #model_type = "rossler"
    model_type = "repressilator"

    def __init__(
        self,
        n_ctx=32,
        n_embd=32,
        n_layer=4,
        n_head=4, # n_head must be a factor of n_embd
        state_dims=[6],
        activation_function="gelu_new",
        initializer_range=0.02,
        **kwargs
    ):
        super().__init__(
            n_ctx=n_ctx,
            n_embd=n_embd,
            n_layer=n_layer,
            n_head=n_head,
            state_dims=state_dims,
            activation_function=activation_function,
            initializer_range=initializer_range,
            **kwargs
        )

"""## Embedding Neural Network Class"""

class RepressilatorEmbedding(EmbeddingModel):
    """Embedding model for the Repressilator ODE system

    Args:
        config (PhysConfig) Configuration class with transformer/embedding parameters
    """
    #model_name = "embedding_rossler"
    model_name = "embedding_repressilator"

    def __init__(self, config: PhysConfig) -> None:
        """Constructor method
        """
        super().__init__(config)

        #hidden_states = int(abs(config.state_dims[0] - config.n_embd)/2) + 1
        #hidden_states = 500
        self.observableNet = nn.Sequential( 
            #nn.Linear(config.state_dims[0],config.state_dims[0]),
            KAN([config.state_dims[0],50,config.n_embd]),
            nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon),
            nn.Dropout(config.embd_pdrop)
        )
        
        self.recoveryNet = nn.Sequential(
            #nn.Linear(config.n_embd,config.n_embd),
            nn.Tanh(),
            KAN([config.n_embd,50,config.state_dims[0]])
        )
        # Learned koopman operator
        # Learns skew-symmetric matrix with a diagonal
        self.obsdim = config.n_embd
        self.kMatrixDiag = nn.Parameter(torch.linspace(1, 0, config.n_embd))

        xidx = []
        yidx = []
        for i in range(1, 3):
            yidx.append(np.arange(i, config.n_embd))
            xidx.append(np.arange(0, config.n_embd-i))

        self.xidx = torch.LongTensor(np.concatenate(xidx))
        self.yidx = torch.LongTensor(np.concatenate(yidx))
        self.kMatrixUT = nn.Parameter(0.1*torch.rand(self.xidx.size(0)))
        # Normalization occurs inside the model
        self.register_buffer('mu', torch.tensor([0., 0., 0., 0., 0., 0.]))
        self.register_buffer('std', torch.tensor([1., 1., 1., 1., 1., 1.]))
        print('Number of embedding parameters: {}'.format( super().num_parameters ))
        logger.info(f'Number of embedding parameters: {super().num_parameters}')

    def forward(self, x: Tensor) -> TensorTuple:
        """Forward pass

        Args:
            x (torch.Tensor): [B, 6] Input feature tensor

        Returns:
            (tuple): tuple containing:

                | (torch.Tensor): [B, config.n_embd] Koopman observables
                | (torch.Tensor): [B, 6] Recovered feature tensor
        """
        # Encode
        x = self._normalize(x)
        g = self.observableNet(x)
        # Decode
        out = self.recoveryNet(g)
        xhat = self._unnormalize(out)
        return g, xhat

    def embed(self, x: Tensor) -> Tensor:
        """Embeds tensor of state variables to Koopman observables

        Args:
            x (Tensor): [B, 6] input feature tensor

        Returns:
            (Tensor): [B, config.n_embd] Koopman observables
        """
        x = self._normalize(x)
        g = self.observableNet(x)
        return g

    def recover(self, g: Tensor) -> Tensor:
        """Recovers feature tensor from Koopman observables

        Args:
            g (Tensor): [B, config.n_embd] Koopman observables

        Returns:
            (Tensor): [B, 6] Physical feature tensor
        """

        out = self.recoveryNet(g)
        x = self._unnormalize(out)
        return x

    def koopmanOperation(self, g: Tensor) -> Tensor:
        """Applies the learned koopman operator on the given observables.

        Args:
            (Tensor): [B, config.n_embd] Koopman observables

        Returns:
            (Tensor): [B, config.n_embd] Koopman observables at the next time-step
        """
        # Koopman operator
        kMatrix = Variable(torch.zeros(self.obsdim, self.obsdim)).to(self.kMatrixUT.device)
        # Populate the off diagonal terms
        kMatrix[self.xidx, self.yidx] = self.kMatrixUT
        kMatrix[self.yidx, self.xidx] = -self.kMatrixUT

        # Populate the diagonal
        ind = np.diag_indices(kMatrix.shape[0])
        kMatrix[ind[0], ind[1]] = self.kMatrixDiag

        # Apply Koopman operation
        gnext = torch.bmm(kMatrix.expand(g.size(0), kMatrix.size(0), kMatrix.size(0)), g.unsqueeze(-1))
        self.kMatrix = kMatrix
        return gnext.squeeze(-1) # Squeeze empty dim from bmm

    @property
    def koopmanOperator(self, requires_grad: bool = True) -> Tensor:
        """Current Koopman operator

        Args:
            requires_grad (bool, optional): if to return with gradient storage, defaults to True
        """
        if not requires_grad:
            return self.kMatrix.detach()
        else:
            return self.kMatrix

    def _normalize(self, x: Tensor) -> Tensor:
        return (x - self.mu.unsqueeze(0))/self.std.unsqueeze(0)

    def _unnormalize(self, x: Tensor) -> Tensor:
        return self.std.unsqueeze(0)*x + self.mu.unsqueeze(0)

    @property
    def koopmanDiag(self):
        return self.kMatrixDiag

"""## Embedding Network Trainer Class"""

class RepressilatorEmbeddingTrainer(EmbeddingTrainingHead):
    """Training head for the Repressilator embedding model for parallel training

    Args:
        config (PhysConfig) Configuration class with transformer/embedding parameters
    """
    def __init__(self, config: PhysConfig) -> None:
        """Constructor method
        """
        super().__init__()
        self.embedding_model = RepressilatorEmbedding(config)

    def forward(self, states: Tensor) -> FloatTuple:
        """Trains model for a single epoch

        Args:
            states (Tensor): [B, T, 6] Time-series feature tensor

        Returns:
            FloatTuple: Tuple containing:

                | (float): Koopman based loss of current epoch
                | (float): Reconstruction loss
        """
        self.embedding_model.train()
        device = self.embedding_model.devices[0]

        loss_reconstruct = 0
        mseLoss = nn.MSELoss()

        xin0 = states[:,0].to(device) # Time-step
        

        # Model forward for both time-steps
        g0, xRec0 = self.embedding_model(xin0)
        loss = (1e3)*mseLoss(xin0, xRec0)
        loss_reconstruct = loss_reconstruct + mseLoss(xin0, xRec0).detach()

        g1_old = g0

        # Koopman transform
        for t0 in range(1, states.shape[1]):
            xin0 = states[:,t0,:].to(device) # Next time-step
            _, xRec1 = self.embedding_model(xin0)

            g1Pred = self.embedding_model.koopmanOperation(g1_old)
            xgRec1 = self.embedding_model.recover(g1Pred)

            loss = loss + mseLoss(xgRec1, xin0) + (1e3)*mseLoss(xRec1, xin0) \
                + (1e-1)*torch.sum(torch.pow(self.embedding_model.koopmanOperator, 2))

            loss_reconstruct = loss_reconstruct + mseLoss(xRec1, xin0).detach()
            g1_old = g1Pred

        return loss, loss_reconstruct

    def evaluate(self, states: Tensor) -> Tuple[float, Tensor, Tensor]:
        """Evaluates the embedding models reconstruction error and returns its
        predictions.

        Args:
            states (Tensor): [B, T, 3] Time-series feature tensor

        Returns:
            Tuple[Float, Tensor, Tensor]: Test error, Predicted states, Target states
        """
        self.embedding_model.eval()
        device = self.embedding_model.devices[0]

        mseLoss = nn.MSELoss()

        # Pull out targets from prediction dataset
        #yTarget = states[:,1:].to(device)
        yTarget = states[:,:-1].to(device)
        xInput = states[:,:-1].to(device)
        yPred = torch.zeros(yTarget.size()).to(device)

 
        #xInput0 = xInput[:,0].to(device)
        #g0 = self.embedding_model.embed(xInput0)
        #yPred0 = self.embedding_model.recover(g0).detach()
        #logger.info(f'x0 {xInput0}')
        #logger.info(f'g0 {g0}')
        #logger.info(f'xRec0 {yPred0}')

        
        # Test accuracy of one time-step
        for i in range(xInput.size(1)):
            xInput0 = xInput[:,i].to(device)
            g0 = self.embedding_model.embed(xInput0)
            yPred0 = self.embedding_model.recover(g0).detach()
            yPred[:,i] = yPred0.squeeze()

        test_loss = mseLoss(yTarget, yPred)

        return test_loss, yPred, yTarget

"""## Repressilator Embedding Data-Handler"""

class RepressilatorDataHandler(EmbeddingDataHandler):
    """Embedding data handler for repressilator system.
    Contains methods for creating training and testing loaders,
    dataset class and data collator.
    """
    class RepressilatorDataset(Dataset):
        def __init__(self, examples):
            self.examples = examples

        def __len__(self):
            return len(self.examples)

        def __getitem__(self, i) -> Dict[str, torch.Tensor]:
            return {"states": self.examples[i]}

    class RepressilatorDataCollator:
        """
        Data collator for Repressilator embedding problem
        """
        # Default collator
        def __call__(self, examples:List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:

            x_data_tensor =  torch.stack([example['states'] for example in examples])
            return {"states": x_data_tensor}

    def createTrainingLoader(
        self,
        file_path: str,
        block_size: int,
        stride:int = 1,
        ndata:int = -1,
        batch_size:int = 32,
        shuffle=True,
    ) -> DataLoader:
        """Creating embedding training data loader for Repressilator system.
        For a single training simulation, the total time-series is sub-chunked into
        smaller blocks for training.

        Args:
            file_path (str): Path to HDF5 file with training data
            block_size (int): The length of time-series blocks
            stride (int): Stride of each time-series block
            ndata (int, optional): Number of training time-series. If negative, all of the provided
            data will be used. Defaults to -1.
            batch_size (int, optional): Training batch size. Defaults to 32.
            shuffle (bool, optional): Turn on mini-batch shuffling in dataloader. Defaults to True.

        Returns:
            (DataLoader): Training loader
        """
        logger.info('Creating training loader')
        assert os.path.isfile(file_path), "Training HDF5 file {} not found".format(file_path)

        examples = []
        with h5py.File(file_path, "r") as f:
            # Iterate through stored time-series
            samples = 0
            for key in f.keys():
                data_series = torch.Tensor(f[key])
                # Stride over time-series by specified block size
                for i in range(0,  data_series.size(0) - block_size + 1, stride):
                    examples.append(data_series[i : i + block_size].unsqueeze(0))

                samples = samples + 1
                if(ndata > 0 and samples > ndata): #If we have enough time-series samples break loop
                    break

        data = torch.cat(examples, dim=0)
        logger.info("Training data-set size: {}".format(data.size()))

        # Normalize training data
        # Normalize x and y with Gaussian, normalize z with max/min
        self.mu = torch.tensor([torch.min(data[:,:,0]), torch.min(data[:,:,1]), torch.min(data[:,:,2]),torch.min(data[:,:,3]),torch.min(data[:,:,4]),torch.min(data[:,:,5])])
        self.std = torch.tensor([torch.max(data[:,:,0])-torch.min(data[:,:,0]), torch.max(data[:,:,1])-torch.min(data[:,:,1]), torch.max(data[:,:,2])-torch.min(data[:,:,2]),torch.max(data[:,:,3])-torch.min(data[:,:,3]),torch.max(data[:,:,4])-torch.min(data[:,:,4]),torch.max(data[:,:,5])-torch.min(data[:,:,5])])

        # Needs to min-max normalization due to the reservoir matrix, needing to have a spectral density below 1
        if(data.size(0) < batch_size):
            logger.warn('Lower batch-size to {:d}'.format(data.size(0)))
            batch_size = data.size(0)

        dataset = self.RepressilatorDataset(data)
        data_collator = self.RepressilatorDataCollator()
        training_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=data_collator,num_workers=4, drop_last=True)
        return training_loader

    def createTestingLoader(self,
        file_path: str,
        block_size: int,
        ndata:int = -1,
        batch_size:int=32,
        shuffle=False
    ) -> DataLoader:
        """Creating testing/validation data loader for Repressilator system.
        For a data case with time-steps [0,T], this method extract a smaller
        time-series to be used for testing [0, S], s.t. S < T.

        Args:
            file_path (str): Path to HDF5 file with testing data
            block_size (int): The length of testing time-series
            ndata (int, optional): Number of testing time-series. If negative, all of the provided
            data will be used. Defaults to -1.
            batch_size (int, optional): Testing batch size. Defaults to 32.
            shuffle (bool, optional): Turn on mini-batch shuffling in dataloader. Defaults to False.

        Returns:
            (DataLoader): Testing/validation data loader
        """
        logger.info('Creating testing loader')
        assert os.path.isfile(file_path), "Testing HDF5 file {} not found".format(file_path)

        examples = []
        with h5py.File(file_path, "r") as f:
            # Iterate through stored time-series
            samples = 0
            for key in f.keys():
                data_series = torch.Tensor(f[key])
                # Stride over time-series
                for i in range(0,  data_series.size(0) - block_size + 1, block_size):  # Truncate in block of block_size
                    examples.append(data_series[i : i + block_size].unsqueeze(0))
                    break

                samples = samples + 1
                if(ndata > 0 and samples >= ndata): #If we have enough time-series samples break loop
                    break

        # Combine data-series
        data = torch.cat(examples, dim=0)
        logger.info("Testing data-set size: {}".format(data.size()))

        if(data.size(0) < batch_size):
            logger.warn('Lower batch-size to {:d}'.format(data.size(0)))
            batch_size = data.size(0)

        #data = (data - self.mu.unsqueeze(0).unsqueeze(0)) / self.std.unsqueeze(0).unsqueeze(0)
        dataset = self.RepressilatorDataset(data)
        data_collator = self.RepressilatorDataCollator()
        testing_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=data_collator, num_workers=4,drop_last=False)

        return testing_loader

"""## Initializing Datasets and Models

Now we can use the auto classes to initialized the predefined configs, dataloaders and models. This may take a bit!
"""

data_handler = RepressilatorDataHandler()
# Set up data-loaders
training_loader = data_handler.createTrainingLoader(
    args.training_h5_file,
    block_size=args.block_size,
    stride=args.stride,
    ndata=args.n_train,
    batch_size=args.batch_size)

testing_loader = data_handler.createTestingLoader(
    args.eval_h5_file,
    block_size=32,
    ndata=args.n_eval,
    batch_size=8)

# Load configuration file then init model
config = RepressilatorConfig()
model = RepressilatorEmbeddingTrainer(config)
mu, std = data_handler.norm_params
model.embedding_model.mu = mu.to(args.device)
model.embedding_model.std = std.to(args.device)

if args.epoch_start > 1:
    model.load_model(args.ckpt_dir, args.epoch_start)

"""Initialize optimizer and scheduler. Feel free to change if you want to experiment."""

optimizer = torch.optim.Adam(model.parameters(), lr=args.lr*0.995**(args.epoch_start), weight_decay=1e-8)
#optimizer = torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=1e-4)
scheduler = ExponentialLR(optimizer, gamma=0.995)

"""## Training the Embedding Model

Train the model. No visualization here, just boring numbers. This notebook only trains for 100 epochs for brevity, feel free to train longer. The test loss here is only the recovery loss MSE(x - decode(encode(x))) and does not reflect the quality of the Koopman dynamics.
"""
## model architecture
## printing model architecture
logger.info(model)

trainer = EmbeddingTrainer(model, args, (optimizer, scheduler))
trainer.train(training_loader, testing_loader)

"""Check your Google drive for checkpoints."""

""" plot the test loss"""
with h5py.File(os.path.join(args.run_dir,'embedding_metrics.h5'), 'r') as f:
    # change the keys values to list
    test_loss = list(f['test_loss'])
    train_loss = list(f['train_loss'])
    epoch = list(f['epoch'])
    # plot the graph
    plt.plot(epoch, test_loss, label='test_loss')
    #plt.plot(epoch, train_loss, label='train_loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend()
    plt.savefig(os.path.join(args.run_dir,'embedding_loss.png'))