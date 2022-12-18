import torch

from training_config import MLP_MNIST_DEFAULT
from training import setup_train, train_model

training_config = MLP_MNIST_DEFAULT
training_config.epochs = 10
training_config.seed = 7

training_config.batch_size = 1

# train model a
model_a = train_model(*setup_train(training_config), verbose = 2)
torch.save(model_a.state_dict(), 'model_a.pt')

# change seed to get different model
experiment_config.seed = 42

# train model b
model_a = train_model(*setup_train(training_config), verbose = 2)
torch.save(model_b.state_dict(), 'model_b.pt')
