from Data_Loaders import Nav_Dataset, Data_Loaders
import numpy
import torch
import torch.utils.data
import torch.utils.data as data
import torch.utils.data.dataset as dataset

def test_init_test():
    nav_dataset = Nav_Dataset()
    assert len(nav_dataset.data) != 0, "Dataset length is 0; no data loaded."

def test_len_not_zero():
    nav_dataset = Nav_Dataset()
    length = nav_dataset.__len__()
    assert type(length) == int, "Dataset length did not return an integer"
    assert length > 100, "Dataset length was not more than 100 rows. Did you load all the data?"

def test_getitem():
    nav_dataset = Nav_Dataset()
    item = nav_dataset.__getitem__(0)
    assert type(item) == dict, "Return from __getitem__ is not a dictionary"
    has_input_key = 'input' in item
    has_label_key = 'label' in item
    assert has_input_key == True, "Return dictionary did not have key 'input'"
    assert has_label_key == True, "Return dictionary did not have key 'label'"
    assert type(item['input']) is numpy.ndarray
    assert type(item['label']) is numpy.float32
    
def test_data_loaders_dataloader_types_correct():
    data_loaders = Data_Loaders()
    assert type(data_loaders.train_loader) is data.DataLoader, "train_loader is not of type DataLoader"
    assert type(data_loaders.test_loader) is data.DataLoader, "test_loader is not of type DataLoader"

def test_data_loaders_batch_size_correct():
    data_loaders = Data_Loaders()
    assert data_loaders.train_loader.batch_size == 16, "Batch size for train_loader not set to 16"
    assert data_loaders.test_loader.batch_size == 16, "Bach size for test_loader not set to 16"
