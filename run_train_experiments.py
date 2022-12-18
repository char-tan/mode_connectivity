from train_experiments import *

experiment_config = MLP_MNIST_DEFAULT
experiment_config.epochs = 10
experiment_config.seed = 7

model_a,device,train_loader,test_loader,optimizer,epochs,scheduler,log_interval = setup_train(experiment_config)
run_training(model_a,device,train_loader,test_loader,optimizer,epochs,scheduler,log_interval, verbose=1)

# Change seed to get different model
experiment_config.seed = 42
model_b,device,train_loader,test_loader,optimizer,epochs,scheduler,log_interval = setup_train(experiment_config)
run_training(model_b,device,train_loader,test_loader,optimizer,epochs,scheduler,log_interval, verbose=1)
