import torch
import torch.nn as nn

class Action_Conditioned_FF(nn.Module):
    def __init__(self):
# STUDENTS: __init__() must initiatize nn.Module and define your network's
        super(Action_Conditioned_FF, self).__init__()
        # parameters
        self.fc1 = nn.Linear(6, 16)
        self.fc2 = nn.Linear(16, 32)
        self.fc3 = nn.Linear(32, 1)


    def forward(self, input):
        # STUDENTS: forward() must complete a single forward pass through your network
        # and return the output which should be a tensor
        self.z = torch.relu(self.fc1(input))
        self.z1 = torch.relu(self.fc2(self.z))
        output = torch.sigmoid(self.fc3(self.z1))
        return output


    def evaluate(self, model, test_loader, loss_function):
# STUDENTS: evaluate() must return the loss (a value, not a tensor) over your testing dataset. Keep in
# mind that we do not need to keep track of any gradients while evaluating the
# model. loss_function will be a PyTorch loss function which takes as argument the model's
# output and the desired output.
        print(test_loader)
        loss = 0
        for i in range(0,len(test_loader['input'])):
          output = model(test_loader['input'][i])
          loss = loss + loss_function(output, test_loader['label'][i])
        loss = loss / len(test_loader['input'])
        return loss

def main():
    model = Action_Conditioned_FF()

if __name__ == '__main__':
    main()
