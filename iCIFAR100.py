from torchvision.datasets import CIFAR100
import numpy as np
from PIL import Image
import torchvision.transforms as transforms

mean = (0.4914, 0.4822, 0.4465)
std = (0.2023, 0.1994, 0.2010)
normalize = transforms.Normalize(mean=mean, std=std)
train_transform = transforms.Compose([
        transforms.RandomResizedCrop(size=32, scale=(0.2, 1.)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(),
        normalize,
    ])

class TwoCropTransform:
    """Create two crops of the same image"""
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, x):
        return [self.transform(x), self.transform(x)]


class iCIFAR100(CIFAR100):
    def __init__(self,root,
                 train=True,
                 train_transform=None,
                 target_transform=None,
                 test_transform=None,
                 target_test_transform=None,
                 download=False):
        super(iCIFAR100,self).__init__(root,
                                       train=train,
                                       transform=TwoCropTransform(train_transform),
                                       #transform=transform,
                                       target_transform=target_transform,
                                       download=download)

        self.target_test_transform=target_test_transform
        self.test_transform=test_transform
        self.TrainData = []
        self.TrainLabels = []
        self.TestData = []
        self.TestLabels = []

    def concatenate(self,datas):
        con_data=datas[0]
        # con_label=labels[0]
        for i in range(1,len(datas)):
            con_data=np.concatenate((con_data,datas[i]),axis=0)
            #con_label=np.concatenate((con_label,labels[i]),axis=0)
        return con_data

    def getTestData(self, classes):
        datas,labels=[],[]
        for label in range(classes[0], classes[1]):
            data = self.data[np.array(self.targets) == label]
            datas.append(data)
            for _ in range(data.shape[0]):
                labels.append(label)
            #labels.append(np.full((data.shape[0]), label))
        datas=self.concatenate(datas)
        self.TestData=datas if len(self.TestData) == 0 else np.concatenate((self.TestData,datas),axis=0)
        self.TestLabels=labels if len(self.TestLabels) == 0 else (self.TestLabels + labels)
        print("the size of test set is %s"%(str(self.TestData.shape)))
        print("the size of test label is %s"%str(len(self.TestLabels)))


    def getTrainData(self,classes,exemplar_set,exemplar_set_label):

        datas,labels=[],[]
        if len(exemplar_set)!=0:
            exemplars=[exemplar for exemplar in exemplar_set]
            datas.append(np.stack(exemplars))
            labels = [label for label in exemplar_set_label]
            #length=len(datas[0])
            #labels=[np.full((length),label) for label in range(len(exemplar_set))]

        for label in range(classes[0],classes[1]):
            data=self.data[np.array(self.targets)==label]
            datas.append(data)
            for _ in range(data.shape[0]):
                labels.append(label)
            #labels.append(np.full((data.shape[0]),label))
        self.TrainData=self.concatenate(datas)
        self.TrainLabels = labels
        print("the size of train set is %s"%(str(self.TrainData.shape)))
        print("the size of train label is %s"%str(len(self.TrainLabels)))

    def getTrainItem(self,index):
        img, target = Image.fromarray(self.TrainData[index]), self.TrainLabels[index]

        if self.transform:
            img=self.transform(img)

        if self.target_transform:
            target=self.target_transform(target)

        return index,img,target

    def get_item(self, index):
        img, target = self.TrainData[index], self.TrainLabels[index]

        return img,target

    def getTestItem(self,index):
        img, target = Image.fromarray(self.TestData[index]), self.TestLabels[index]

        if self.test_transform:
            img=self.test_transform(img)

        if self.target_test_transform:
            target=self.target_test_transform(target)

        return index, img, target

    def __getitem__(self, index):
        if len(self.TrainData)!=0:
            return self.getTrainItem(index)
        elif len(self.TestData)!=0:
            return self.getTestItem(index)


    def __len__(self):
        if len(self.TrainData)!=0:
        # if self.TrainData!=[]:
            return len(self.TrainData)
        elif len(self.TestData)!=0:
        # elif self.TestData!=[]:
            return len(self.TestData)

    def get_image_class(self,label):
        return self.data[np.array(self.targets)==label]