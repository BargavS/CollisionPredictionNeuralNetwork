import torch
import torch.utils.data as data
import torch.utils.data.dataset as dataset
import numpy as np
import pickle
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split


class Nav_Dataset(dataset.Dataset):
    def __init__(self):
        self.label0 = []
        self.label1 = []
        self.data = np.genfromtxt('saved/training_data.csv', delimiter=',')
# STUDENTS: it may be helpful for the final part to balance the distribution of your collected data

        # normalize data and save scaler for inference
        self.scaler = MinMaxScaler()
        self.normalized_data = self.scaler.fit_transform(self.data) #fits and transforms
        for data_i in self.normalized_data:
            if data_i[6] == 0:

                label0_dict = {'input': [], 'label': 0}
                label0_dict['input'].append(data_i[0])
                label0_dict['input'].append(data_i[1])
                label0_dict['input'].append(data_i[2])
                label0_dict['input'].append(data_i[3])
                label0_dict['input'].append(data_i[4])
                label0_dict['input'].append(data_i[5])
                label0_dict['label'] = (data_i[6])
                self.label0.append(label0_dict)
            elif data_i[6] == 1:
                label1_dict = {'input': [], 'label': 0}
                label1_dict['input'].append(data_i[0])
                label1_dict['input'].append(data_i[1])
                label1_dict['input'].append(data_i[2])
                label1_dict['input'].append(data_i[3])
                label1_dict['input'].append(data_i[4])
                label1_dict['input'].append(data_i[5])
                label1_dict['label'] = (data_i[6])

                self.label1.append(label1_dict)

        pickle.dump(self.scaler, open("saved/scaler.pkl", "wb")) #save to normalize at inference

    def __len__(self):
        # STUDENTS: __len__() returns the length of the dataset
        return len(self.normalized_data)

    def __getitem__(self, idx):
        if not isinstance(idx, int):
            idx = idx.item()
        x_data = torch.tensor(self.normalized_data[idx,0], self.normalized_data[idx,1],self.normalized_data[idx,2],self.normalized_data[idx,3],self.normalized_data[idx,4],self.normalized_data[idx,5])
        y_data = torch.tensor(self.normalized_data[idx, 6])
        return {'input': x_data, 'label': y_data}

# STUDENTS: for this example, __getitem__() must return a dict with entries {'input': x, 'label': y}
# x and y should both be of type float32. There are many other ways to do this, but to work with autograding
# please do not deviate from these specifications.


class Data_Loaders():
    def __init__(self, batch_size):
        self.nav_dataset = Nav_Dataset()
        self.X = []
        self.Y = []
        for data_i in self.nav_dataset.normalized_data:
            self.X.append([(data_i[0],data_i[1],data_i[2],data_i[3],data_i[4],data_i[5])])
            self.Y .append(data_i[6])

        [train_X, test_X, train_Y, test_Y] = train_test_split(np.array(self.X), np.array(self.Y), test_size=0.2, shuffle=True, stratify=self.Y)
        DatasetTrain = []
        DatasetTest = []
        for d_train,d_train_l in zip( train_X,train_Y):
            DatasetTrain.append({'input': d_train, 'label': d_train_l})
        for d_test,d_test_l in zip(test_X,test_Y):
            DatasetTest.append({'input': d_test, 'label': d_test_l})

        #DatasetTrain = dataset.TensorDataset(torch.from_numpy(train_X),torch.from_numpy(train_Y))
        #DatasetTest = dataset.TensorDataset(torch.from_numpy(test_X),torch.from_numpy(test_Y))
        #print(DatasetTest)
        self.train_loader = data.DataLoader(DatasetTrain,  shuffle=True, drop_last=True,num_workers=0)

        self.test_loader = data.DataLoader(DatasetTest, drop_last=True, num_workers=0)



# STUDENTS: randomly split dataset into two data.DataLoaders, self.train_loader and self.test_loader
# make sure your split can handle an arbitrary number of samples in the dataset as this may vary

def main():
    batch_size = 16
    data_loaders = Data_Loaders(batch_size)
    # STUDENTS : note this is how the dataloaders will be iterated over, and cannot be deviated from
    for idx, sample in enumerate(data_loaders.train_loader):
        _, _ = sample['input'], sample['label']
    for idx, sample in enumerate(data_loaders.test_loader):
        _, _ = sample['input'], sample['label']

if __name__ == '__main__':
    main()
