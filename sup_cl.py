import torch.nn as nn
import torch
from torchvision import transforms
import numpy as np
from torch.nn import functional as F
from PIL import Image
import torch.optim as optim
from myNetwork import network
from iCIFAR100 import iCIFAR100
from torch.utils.data import DataLoader
from resnet_big import SupConResNet
from loss import SupConLoss
from utils import AverageMeter
from dpp import get_dpp
from tqdm import tqdm

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def get_one_hot(target, num_class):
    one_hot = torch.zeros(target.shape[0], num_class).to(device)
    one_hot = one_hot.scatter(dim=1, index=target.long().view(-1, 1), value=1.)
    return one_hot


class SupCL:

    def __init__(self, numclass, feature_extractor, batch_size, task_size, memory_size, epochs, learning_rate, temperature):

        super(SupCL, self).__init__()
        self.epochs = epochs
        self.temperature = temperature
        self.learning_rate = learning_rate
        self.model = feature_extractor
        self.exemplar_set = []
        self.exemplar_set_label = []
        self.class_mean_set = []
        self.numclass = numclass
        self.transform = transforms.Compose([  # transforms.Resize(img_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))])
        self.old_model = None

        self.train_transform = transforms.Compose([  # transforms.Resize(img_size),
            transforms.RandomCrop((32, 32), padding=4),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.24705882352941178),
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))])

        self.test_transform = transforms.Compose([  # transforms.Resize(img_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))])

        self.classify_transform = transforms.Compose([transforms.RandomHorizontalFlip(p=1.),
                                                      # transforms.Resize(img_size),
                                                      transforms.ToTensor(),
                                                      transforms.Normalize((0.5071, 0.4867, 0.4408),
                                                                           (0.2675, 0.2565, 0.2761))])

        self.train_dataset = iCIFAR100('dataset', train_transform=self.train_transform, train=True, download=True)
        self.test_dataset = iCIFAR100('dataset', test_transform=self.test_transform, train=False, download=True)

        self.batchsize = batch_size
        self.memory_size = memory_size
        self.task_size = task_size

        self.train_loader = None
        self.test_loader = None

    # get incremental train data
    # incremental
    def beforeTrain(self):
        self.model.eval()
        classes = [self.numclass - self.task_size, self.numclass]
        self.train_loader, self.test_loader = self._get_train_and_test_dataloader(classes)
        # if self.numclass > self.task_size:
        #     self.model.Incremental_learning(self.numclass)
        self.model.train()
        self.model.to(device)

    def _get_train_and_test_dataloader(self, classes):
        self.train_dataset.getTrainData(classes, self.exemplar_set, self.exemplar_set_label)
        self.test_dataset.getTestData(classes)
        train_loader = DataLoader(dataset=self.train_dataset,
                                  shuffle=True,
                                  batch_size=self.batchsize)

        test_loader = DataLoader(dataset=self.test_dataset,
                                 shuffle=False,
                                 batch_size=self.batchsize)

        return train_loader, test_loader

    '''
    def _get_old_model_output(self, dataloader):
        x = {}
        for step, (indexs, imgs, labels) in enumerate(dataloader):
            imgs, labels = imgs.to(device), labels.to(device)
            with torch.no_grad():
                old_model_output = torch.sigmoid(self.old_model(imgs))
            for i in range(len(indexs)):
                x[indexs[i].item()] = old_model_output[i].cpu().numpy()
        return x
    '''

    def _compute_loss_kd(self, images, target, temperature):
        n_total = len(self.train_dataset)
        criterion = SupConLoss(n_total=n_total, n_buffer=self.memory_size, temperature=temperature)
        bsz = target.shape[0]
        features = self.model(images)
        f1, f2 = torch.split(features, [bsz, bsz], dim=0)
        features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
        if self.old_model is None:
            return criterion(features, target)
        else:
            loss_cls = criterion(features, target, target_labels=list(range(self.memory_size)))
            #feature = self.model.feature(imgs)
            feature_old = self.old_model(images)
            f1_old, f2_old = torch.split(feature_old, [bsz, bsz], dim=0)
            features_old = torch.cat([f1_old.unsqueeze(1), f2_old.unsqueeze(1)], dim=1)
            loss_kd = torch.dist(features, features_old, 2)

            return loss_cls + 0.3 * loss_kd

    # train model
    # compute loss
    # evaluate model
    def train(self):
        accuracy = 0
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=0.00005)
        losses = AverageMeter()
        #criterion = SupConLoss(temperature=0.07)
        for epoch in range(self.epochs):
            if epoch == 48:
                if self.numclass == self.task_size:
                    print(1)
                    #opt = optim.SGD(self.model.parameters(), lr=0.01 / 5, weight_decay=0.00001)
                    opt = optim.Adam(self.model.parameters(), lr=0.01/5, weight_decay=0.00005)
                else:
                    for p in opt.param_groups:
                        p['lr'] = self.learning_rate / 5
                    # opt = optim.SGD(self.model.parameters(), lr=self.learning_rate/ 5,weight_decay=0.00001,momentum=0.9,nesterov=True,)
                print("change learning rate:%.3f" % (self.learning_rate / 5))
            elif epoch == 62:
                if self.numclass > self.task_size:
                    for p in opt.param_groups:
                        p['lr'] = self.learning_rate / 25
                    # opt = optim.SGD(self.model.parameters(), lr=self.learning_rate/ 25,weight_decay=0.00001,momentum=0.9,nesterov=True,)
                else:
                    #opt = optim.SGD(self.model.parameters(), lr=0.01 / 25, weight_decay=0.00001)
                    opt = optim.Adam(self.model.parameters(), lr=0.01/25, weight_decay=0.00005)
                print("change learning rate:%.3f" % (self.learning_rate / 25))
            elif epoch == 80:
                if self.numclass == self.task_size:
                    #pt = optim.SGD(self.model.parameters(), lr=0.01 / 125, weight_decay=0.00001)
                    opt = optim.Adam(self.model.parameters(), lr=0.01/125, weight_decay=0.00005)
                else:
                    for p in opt.param_groups:
                        p['lr'] = self.learning_rate / 125
                    # opt = optim.SGD(self.model.parameters(), lr=self.learning_rate / 125,weight_decay=0.00001,momentum=0.9,nesterov=True,)
                print("change learning rate:%.3f" % (self.learning_rate / 100))
            for step, (indexs, images, target) in enumerate(self.train_loader):
                images = torch.cat([images[0], images[1]], dim=0)
                images, target = images.to(device), target.to(device)
                bsz = target.shape[0]
                #print(features.size()) [256,2,128]
                #loss = criterion(features, target)
                loss = self._compute_loss_kd(images, target, self.temperature)
                #loss = _compute_loss(features, target)
                losses.update(loss.item(), bsz)

                # ADAM
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                print('epoch:%d,step:%d,loss:%.3f' % (epoch, step, loss.item()))

                # output = self.model(images)
            #     loss_value = self._compute_loss(indexs, images, target)
            #     opt.zero_grad()
            #     loss_value.backward()
            #     opt.step()
            #     print('epoch:%d,step:%d,loss:%.3f' % (epoch, step, loss_value.item()))
            # accuracy = self._test(self.test_loader, 1)
            self.model.eval()
            self._test(self.train_loader, self.test_loader, mlp_epochs=1, num_class=self.numclass, epochs=self.epochs,
                       epoch=epoch)
            self.model.train()
            # print('epoch:%d,accuracy:%.3f' % (epoch, accuracy))
        return losses.avg

    def train_net(self, model, dataloader, test_loader, epochs):
        # device = torch.device('cuda:1')
        net = ShareNet(feature_ex=model)
        net = net.to(device)
        total_step = len(dataloader)
        optimizer = torch.optim.Adam(
            [{'params': net.share.parameters(), 'lr': 0}, {'params': net.previous.parameters()}], lr=0.01)
        # net.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
        # optimizer = torch.optim.Adam(net.parameters(),lr=0.01)
        criterion = nn.CrossEntropyLoss()
        net.train()
        for epoch in range(epochs):
            for i, (_, image, label) in enumerate(dataloader):
                images = image[0]
                images = images.to(device)
                label = label.to(device)
                #bsz = label.shape[0]

                outputs = net(images)
                loss = criterion(outputs, label)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if (i + 1) % 3 == 0:
                    print(f'LOSS: {loss.item()}  STEP: [{i + 1}/{total_step}] EPOCH [{epoch + 1}/{epochs}]')
        net.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for _, image, label in test_loader:
                image = image.to(device)
                label = label.to(device)
                outputs = net(image)
                correct += (torch.argmax(outputs, dim=1) == label).sum().item()
                total += label.size(0)
                acc = correct / total
            print(f'accurancy: {acc}')
        return acc


    def _test(self, train_loader, testloader, mlp_epochs,epoch, epochs, num_class, mode=0):
        # if mode == 0:
        #     print("compute NMS")
        if mode == 0:
            print('compute knn')
            acc = test_knn(self.model, train_loader, testloader, epoch, epochs, num_class)
        else:
            acc = self.train_net(self.model, train_loader, testloader, mlp_epochs)
        # self.model.eval()
        # correct, total = 0, 0
        # for setp, (indexs, imgs, labels) in enumerate(testloader):
        #     imgs, labels = imgs.to(device), labels.to(device)
        #     with torch.no_grad():
        #         outputs = self.model(imgs) if mode == 1 else self.classify(imgs)
        #     predicts = torch.max(outputs, dim=1)[1] if mode == 1 else outputs
        #     correct += (predicts.cpu() == labels.cpu()).sum()
        #     total += len(labels)
        # accuracy = 100 * correct / total
        # self.model.train()
        return acc

    def _compute_loss(self, indexs, imgs, target):
        output = self.model(imgs)
        target = get_one_hot(target, self.numclass)
        output, target = output.to(device), target.to(device)
        if self.old_model == None:
            return F.binary_cross_entropy_with_logits(output, target)
        else:
            # old_target = torch.tensor(np.array([self.old_model_output[index.item()] for index in indexs]))
            old_target = torch.sigmoid(self.old_model(imgs))
            old_task_size = old_target.shape[1]
            target[..., :old_task_size] = old_target
            return F.binary_cross_entropy_with_logits(output, target)



    # change the size of examplar
    def afterTrain(self, accuracy):
        self.exemplar_set = []
        self.exemplar_set_label = []
        features_list = []
        labels_list = []
        self.model.eval()
        with torch.no_grad():
            for (_, data, data_label) in self.train_loader:
                images, labels = data, data_label
                images = torch.cat([images[0], images[1]], dim=0)
                images, labels = images.to(device), labels.to(device)
                bsz = labels.shape[0]
                features = self.model(images)
                f1, _ = torch.split(features, [bsz, bsz], dim=0)
                features_list.append(f1.cpu().numpy())
                labels_list.append(labels.cpu().numpy())

        features = np.concatenate(features_list)
        labels = np.concatenate(labels_list)
        #print(features.shape)

        sampled_indices = get_dpp(features, self.memory_size, labels, self.numclass)
        #print(len(sampled_indices))
        # m = int(self.memory_size / self.numclass)
        # self._reduce_exemplar_sets(m)
        # for i in range(0, self.numclass):
        #     #print('construct class %s examplar:' % (i), end='')
        #     images = self.train_dataset.get_image_class(i)
        #     self._construct_exemplar_set(images, m)

        for i in sampled_indices:
            data, label = self.train_dataset.get_item(i)
            self.exemplar_set.append(data)
            self.exemplar_set_label.append(label)
        #print(len(self.exemplar_set), len(self.exemplar_set_label))

        #self.compute_exemplar_class_mean()
        self.model.eval()
        KNN_accuracy = self._test(self.train_loader, self.test_loader, mlp_epochs=100, num_class=self.numclass, epochs=self.epochs, epoch=self.epochs)
        print("NMS accuracy：" + str(KNN_accuracy))
        filename = 'model/accuracy:%.3f_KNN_accuracy:%.3f_increment:%d_net.pkl' % (accuracy, KNN_accuracy, self.numclass)
        self.numclass += self.task_size
        torch.save(self.model, filename)
        self.old_model = torch.load(filename)
        self.old_model.to(device)
        self.old_model.eval()
        self.model.train()

    

    def Image_transform(self, images, transform):
        data = transform(Image.fromarray(images[0])).unsqueeze(0)
        for index in range(1, len(images)):
            data = torch.cat((data, self.transform(Image.fromarray(images[index])).unsqueeze(0)), dim=0)
        return data

    


class ShareNet(nn.Module):
    def __init__(self, feature_ex):
        super(ShareNet, self).__init__()
        self.share = feature_ex
        self.previous = nn.Sequential(
            nn.ReLU(),
            nn.Linear(128, 100),
            # nn.ReLU(),
            # nn.Linear(128, 100)
            )

    def forward(self, x):
        out = self.share(x)
        out = self.previous(out)
        return out


def knn_predict(feature, feature_bank, feature_labels, classes, knn_k, knn_t):
    # compute cos similarity between each feature vector and feature bank ---> [B, N]
    sim_matrix = torch.mm(feature, feature_bank)
    # [B, K]
    sim_weight, sim_indices = sim_matrix.topk(k=knn_k, dim=-1)
    # [B, K]
    sim_labels = torch.gather(feature_labels.expand(feature.size(0), -1), dim=-1, index=sim_indices)
    sim_weight = (sim_weight / knn_t).exp()

    # counts for each class
    one_hot_label = torch.zeros(feature.size(0) * knn_k, classes, device=sim_labels.device)
    # [B*K, C]
    one_hot_label = one_hot_label.scatter(dim=-1, index=sim_labels.view(-1, 1), value=1.0)
    # weighted score ---> [B, C]
    pred_scores = torch.sum(one_hot_label.view(feature.size(0), -1, classes) * sim_weight.unsqueeze(dim=-1), dim=1)

    pred_labels = pred_scores.argsort(dim=-1, descending=True)
    return pred_labels


def test_knn(net, memory_data_loader, test_data_loader, epoch, epochs, num_class, knn_k=200, knn_t=0.1):
    net.eval()
    #classes = len(memory_data_loader.dataset.classes)
    classes = num_class
    total_top1, total_top5, total_num, feature_bank, feature_labels = 0.0, 0.0, 0, [], []
    with torch.no_grad():
        # generate feature bank
        for _, data, target in tqdm(memory_data_loader, desc='Feature extracting'):
            data = data[0]
            feature = net(data.to(device, non_blocking=True))
            feature = F.normalize(feature, dim=1)
            feature_bank.append(feature)
            feature_labels.append(target)
        # [D, N]
        feature_bank = torch.cat(feature_bank, dim=0).t().contiguous()
        # [N]
        #feature_labels = torch.tensor(memory_data_loader.dataset.label, device=feature_bank.device)
        #print(len(feature_labels))
        #feature_labels = torch.tensor(feature_labels, device=feature_bank.device)
        feature_labels = torch.cat(feature_labels)
        feature_labels = feature_labels.to(device)
        # loop test data to predict the label by weighted knn search
        test_bar = tqdm(test_data_loader)
        for _, data, target in test_bar:
            data, target = data.to(device, non_blocking=True), target.to(device, non_blocking=True)
            feature = net(data)
            feature = F.normalize(feature, dim=1)

            pred_labels = knn_predict(feature, feature_bank, feature_labels, classes, knn_k, knn_t)

            total_num += data.size(0)
            total_top1 += (pred_labels[:, 0] == target).float().sum().item()
            test_bar.set_description(
                'Test Epoch: [{}/{}] Acc@1:{:.2f}%'.format(epoch, epochs, total_top1 / total_num * 100))

    return total_top1 / total_num * 100
