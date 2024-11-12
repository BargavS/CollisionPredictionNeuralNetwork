from Data_Loaders import Data_Loaders
from Networks import Action_Conditioned_FF

import torch
import torch.nn as nn
import matplotlib.pyplot as plt


def train_model(no_epochs):

    batch_size = 16
    data_loaders = Data_Loaders(batch_size)
    model = Action_Conditioned_FF()
    # Me Added
    learning_rate = 0.0004
    # Loss function
    loss_function = nn.BCELoss()  # May change as appropriate
    optimizer = torch.optim.Adam(model.parameters(), learning_rate)

    losses = []
    min_loss = model.evaluate(model, data_loaders.test_loader, loss_function)
    losses.append(min_loss)
    evals = []
    for epoch_i in range(no_epochs):
        model.train()
        m_loss = 0
        count = 0
        for idx, sample in enumerate(data_loaders.train_loader):  # sample['input'] and sample['label']
            x = sample['input']
            y = sample['label']
            z = model.forward(x)
            # print(z)
            loss = loss_function(z, y)
            m_loss += loss.item()
            count += 1
            # backword
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        losses.append(m_loss / count)
        evals.append(model.evaluate(model, data_loaders.test_loader, loss_function))

        torch.save(model.state_dict(), 'saved/saved_model.pkl', _use_new_zipfile_serialization=False)



if __name__ == '__main__':
    no_epochs = 1000
    train_model(no_epochs)
