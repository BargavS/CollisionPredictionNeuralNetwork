import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from Networks import Action_Conditioned_FF
from Data_Loaders import Nav_Dataset, Data_Loaders

def test_init():
    network = Action_Conditioned_FF()
    assert True

def test_forward():
    network = Action_Conditioned_FF()
    data = Data_Loaders(16)
    sample = next(iter(data.train_loader))
    # verify sample is tensor([[ ]]) of floats
    inputs = sample['input']
    assert inputs.dtype is torch.float32, "Input values are not torch.float32s"
    labels = sample['label']
    assert labels.dtype is torch.float32, "Labels are not torch.float32s"
    output = network.forward(inputs)
    assert output.dtype is torch.float32, "Output values are not torch.float32"
    assert output.ndim == 2, "Wrong number of dimensions"
    assert output.shape == torch.Size([16, 1]), "Output shape is not Size([16, 1])"

def test_evaluate():
    network = Action_Conditioned_FF()
    loss_function =  nn.MSELoss() 
    data = Data_Loaders(16)
    # sample = next(iter(data.train_loader))
    average_loss = network.evaluate(network, data.test_loader, loss_function)
    assert type(average_loss) is float, "Average loss was not a float32"
    assert average_loss > 0.0, "Value probably should be > 0.0."
    assert average_loss < 10.0, "Value probably should have been less than 10 using normalized data."
