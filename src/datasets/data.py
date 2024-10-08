import csv

from torch.utils.data import DataLoader, random_split
from src.datasets.subsets import FlickrSubset, FlickrSubsetWithPath, AugmentedFlickrSubset, Nips17Subset, CustomCIFAR10, CustomCIFAR100
from src.datasets.data_transforms.img_transform import IMGTransforms
from torch.utils.data import ConcatDataset
from torchvision.datasets import MNIST, CIFAR10, CIFAR100

class Data:
    
    def __init__(self, dataset_name, *args, **kwargs):
        self.dataset = self.loader(dataset_name, *args, **kwargs)
    
    def loader(self, dataset_name, *args, **kwargs):
        if dataset_name == 'nips17':
            dataset = Nips17ImgNetData(*args, **kwargs)
        elif dataset_name == 'mnist':
            dataset = MNISTDataset(*args, **kwargs)
        elif dataset_name == 'cifar10':
            dataset = CIFAR10Dataset(*args, **kwargs)
        else:
            raise ValueError('Dataset not recognized')
        return dataset

class BaseDataset:
    
    def __init__(self,
                dataset_name,
                model,
                device,
                batch_size,
                transform,
                adversarial_opt,
                adversarial_training_opt,
                greyscale_opt,
                jpeg_compression,
                jpeg_compression_rate,
                target_transform=None,
                input_size=224):


        self.transform_type = transform
        self.transforms = IMGTransforms(transform,
                                   device=device,
                                   target_transform=target_transform, 
                                   input_size=input_size, 
                                   adversarial_opt=adversarial_opt,
                                   greyscale_opt=greyscale_opt,
                                   dataset_type=dataset_name,
                                   model=model,
                                   jpeg_compression=jpeg_compression,
                                   jpeg_compression_rate=jpeg_compression_rate)
        self.greyscale_opt = greyscale_opt
        self.adversarial_opt = adversarial_opt
        self.adversarial_training_opt = adversarial_training_opt
        self.device = device
        self.batch_size = batch_size
        self.x, self.y = input_size, input_size
        
class Nips17ImgNetData(BaseDataset):

    def __init__(self, n_datapoints, *args,**kwargs):
        super().__init__('nips17', *args, **kwargs)
        
        self.categories = self.get_categories()
        self.dataset_type = 'nips17'

        self.test_data = self.get_data(transform_val=self.transforms.transform_val, 
                                    target_transform=self.transforms.target_transform)
        if n_datapoints == -1:
            self.test = self.train = self.validation =  DataLoader(self.test_data, batch_size=self.batch_size)
        else:
            self.test_data, _ = random_split(self.test_data, [n_datapoints, len(self.test_data)-n_datapoints])
            self.test = self.train = self.validation =  DataLoader(self.test_data, batch_size=self.batch_size)

    def get_data(self, transform_val, target_transform,):
        path_test = './data/nips17/'
        path_labels = path_test + 'images.csv'
        path_images = path_test + 'images/'
        test = Nips17Subset(label_path=path_labels, 
                            img_path=path_images, 
                            transform=transform_val, 
                            target_transform=target_transform, 
                            adversarial=self.adversarial_opt.adversarial, 
                            is_test_data=True)
        return test
        
    def get_categories(self):
        categories = {}
        path = './data/nips17/categories.csv'
        with open(path, 'r') as cats:
            filereader = csv.reader(cats)
            next(filereader)
            for ind, cat in filereader:
                categories[int(ind) - 1] = cat
        return categories

class MNISTDataset(BaseDataset):
    
    def __init__(self,
                n_datapoints,
                *args,
                **kwargs):
        super().__init__('mnist', *args, **kwargs)
        
        self.train_val_data, self.test_data =  self.get_data()
        self.dataset_type = 'mnist'
        
        if self.adversarial_training_opt.adversarial_training:
            self.train = DataLoader(self.train_val_data, batch_size=self.batch_size, shuffle=True)
        else:
            self.train_data, self.val_data = self.train_val_data.split_random(self.train_val_data, lengths=[0.8, 0.2])
            self.train_data = DataLoader(self.train_data, batch_size=self.batch_size, shuffle=True)
            self.validation = DataLoader(self.val_data, batch_size=self.batch_size, shuffle=False)
        if n_datapoints == -1:
            self.test = DataLoader(self.test_data, batch_size=self.batch_size, shuffle=False)
        else:
            self.test_data, _ = random_split(self.test_data, [n_datapoints, len(self.test_data)-n_datapoints])
            self.test = DataLoader(self.test_data, batch_size=self.batch_size, shuffle=False)
    
    def get_data(self):
        train_val_data = MNIST(root='./data', train=True, download=True, transform=self.transforms.transform_train)
        test_data = MNIST(root='./data', train=False, download=True, transform=self.transforms.transform_val)
        return train_val_data, test_data
    

class CIFAR10Dataset(BaseDataset):
    
    def __init__(self,
                 n_datapoints,
                *args,
                **kwargs):
        super().__init__('cifar10', *args, **kwargs)
        
        self.train_val_data, self.test_data =  self.get_data()
        self.dataset_type = 'cifar10'

        if self.adversarial_training_opt.adversarial_training:
            self.train = DataLoader(self.train_val_data, batch_size=self.batch_size, shuffle=True)
        else:
            train_size = int(len(self.train_val_data) * 0.8) # 80% training data
            valid_size = len(self.train_val_data) - train_size # 20% validation data
            self.train_data, self.val_data = random_split(self.train_val_data, [train_size, valid_size])
            
            self.train_data = DataLoader(self.train_data, batch_size=self.batch_size, shuffle=True)
            self.validation = DataLoader(self.val_data, batch_size=self.batch_size, shuffle=False)
        if n_datapoints == -1:
            self.test = DataLoader(self.test_data, batch_size=self.batch_size, shuffle=False)
        else:
            self.test_data, _ = random_split(self.test_data, [n_datapoints, len(self.test_data)-n_datapoints])
            self.test = DataLoader(self.test_data, batch_size=self.batch_size, shuffle=False)

    def get_data(self):
        train_val_data = CIFAR10(root='./data', train=True, download=True, transform=self.transforms.transform_train)
        test_data = CustomCIFAR10(root='./data', train=False, download=True, transform=self.transforms.transform_val, adversarial=self.adversarial_opt.adversarial, is_test_data=True)
        return train_val_data, test_data

class CIFAR100Dataset(BaseDataset):
    
    def __init__(self,
                *args,
                **kwargs):
        super().__init__('cifar100', *args, **kwargs)
        
        self.train_val_data, self.test_data =  self.get_data()
        self.dataset_type = 'cifar100'

        if self.adversarial_training_opt.adversarial_training:
            self.train = DataLoader(self.train_val_data, batch_size=self.batch_size, shuffle=True)
        else:
            self.train_data, self.val_data = random_split(self.train_val_data, [0.8, 0.2])
            # self.train_data, self.val_data = self.train_val_data.split_random(self.train_val_data, lengths=[0.8, 0.2])
            self.train = DataLoader(self.train_data, batch_size=self.batch_size, shuffle=True)
            self.validation = DataLoader(self.val_data, batch_size=self.batch_size, shuffle=False)
        self.test = DataLoader(self.test_data, batch_size=self.batch_size, shuffle=False)

    def get_data(self):
        train_val_data = CIFAR100(root='./data', train=True, download=True, transform=self.transforms.transform_train)
        test_data = CustomCIFAR100(adversarial=self.adversarial_opt.adversarial, is_test_data=True, root='./data', train=False, download=True, transform=self.transforms.transform_val)
        return train_val_data, test_data
        
        
if __name__ == '__main__':
    pass
