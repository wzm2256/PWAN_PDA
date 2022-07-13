import pdb

is_torchvision_installed = True
try:
    import torchvision
except:
    is_torchvision_installed = False

import torch
import torch.nn.functional as F
import torch.utils.data
import random
import numpy as np

import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import tqdm
from scipy.spatial.distance import cdist

from torch.utils.data.sampler import BatchSampler
import torch
import matplotlib

matplotlib.use('Agg')
from sklearn.manifold import TSNE
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as col
import os

norm = plt.Normalize(vmin=0., vmax=1.)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


#--------SAMPLER-------

class BalancedBatchSampler(torch.utils.data.sampler.BatchSampler):
    """
    BatchSampler - from a MNIST-like dataset, samples n_samples for each of the n_classes.
    Returns batches of size n_classes * (batch_size // n_classes)
    adapted from https://github.com/adambielski/siamese-triplet/blob/master/datasets.py
    """

    def __init__(self, labels, batch_size):
        classes = sorted(set(labels.numpy()))
        print(classes)

        n_classes = len(classes)
        self._n_samples = batch_size // n_classes
        if self._n_samples == 0:
            raise ValueError(
                f"batch_size should be bigger than the number of classes, got {batch_size}"
            )

        self._class_iters = [
            InfiniteSliceIterator(np.where(labels == class_)[0], class_=class_)
            for class_ in classes
        ]

        batch_size = self._n_samples * n_classes
        self.n_dataset = len(labels)
        self._n_batches = self.n_dataset // batch_size
        if self._n_batches == 0:
            raise ValueError(
                f"Dataset is not big enough to generate batches with size {batch_size}"
            )
        print("K=", n_classes, "nk=", self._n_samples)
        print("Batch size = ", batch_size)

    def __iter__(self):
        for _ in range(self._n_batches):
            indices = []
            for class_iter in self._class_iters:
                indices.extend(class_iter.get(self._n_samples))
            np.random.shuffle(indices)
            yield indices

        for class_iter in self._class_iters:
            class_iter.reset()

    def __len__(self):
        return self._n_batches
    

class CrossEntropyLabelSmooth(nn.Module):
    """Cross entropy loss with label smoothing regularizer.
    Reference:
    Szegedy et al. Rethinking the Inception Architecture for Computer Vision. CVPR 2016.
    Equation: y = (1 - epsilon) * y + epsilon / K.
    Args:
        num_classes (int): number of classes.
        epsilon (float): weight.
    """

    def __init__(self, num_classes, epsilon=0.1, use_gpu=True, reduction=True):
        super(CrossEntropyLabelSmooth, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.use_gpu = use_gpu
        self.reduction = reduction
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def smooth(self, targets):
        targets = torch.zeros((targets.shape[0], self.num_classes)).scatter_(1, targets.unsqueeze(1).cpu(), 1).cuda()
        smoothed_targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
        return smoothed_targets

    def forward(self, inputs, targets):
        """
        Args:
            inputs: prediction matrix (before softmax) with shape (batch_size, num_classes)
            targets: ground truth labels with shape (batch_size)
        """

        log_probs = self.logsoftmax(inputs)
        smoothed_targets = self.smooth(targets)
        # targets = torch.zeros(log_probs.size()).scatter_(1, targets.unsqueeze(1).cpu(), 1)
        # if self.use_gpu: targets = targets.cuda()
        # targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
        # pdb.set_trace()
        loss = (- smoothed_targets * log_probs).sum(dim=1)
        if self.reduction:
            return loss.mean()
        else:
            return loss

class InfiniteSliceIterator:
    def __init__(self, array, class_):
        assert type(array) is np.ndarray
        self.array = array
        self.i = 0
        self.class_ = class_

    def reset(self):
        self.i = 0

    def get(self, n):
        len_ = len(self.array)
        # not enough element in 'array'
        if len_ < n:
            print(f"there are really few items in class {self.class_}")
            self.reset()
            np.random.shuffle(self.array)
            mul = n // len_
            rest = n - mul * len_
            return np.concatenate((np.tile(self.array, mul), self.array[:rest]))

        # not enough element in array's tail
        if len_ - self.i < n:
            self.reset()

        if self.i == 0:
            np.random.shuffle(self.array)
        i = self.i
        self.i += n
        return self.array[i : self.i]

def accuracy(output, target, topk=(1,)):
    r"""
    Computes the accuracy over the k top predictions for the specified values of k

    Args:
        output (tensor): Classification outputs, :math:`(N, C)` where `C = number of classes`
        target (tensor): :math:`(N)` where each value is :math:`0 \leq \text{targets}[i] \leq C-1`
        topk (sequence[int]): A list of top-N number.

    Returns:
        Top-N accuracies (N :math:`\in` topK).
    """
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target[None])

        res = []
        for k in topk:
            correct_k = correct[:k].flatten().sum(dtype=torch.float32)
            res.append(correct_k * (100.0 / batch_size))
        return res



def collect_feature(data_loader: DataLoader, feature_extractor: nn.Module,
                                   device: torch.device, max_num_features=None):
    """
    Fetch data from `data_loader`, and then use `feature_extractor` to collect features

    Args:
        data_loader (torch.utils.data.DataLoader): Data loader.
        feature_extractor (torch.nn.Module): A feature extractor.
        device (torch.device)
        max_num_features (int): The max number of features to return

    Returns:
        Features in shape (min(len(data_loader), max_num_features), :math:`|\mathcal{F}|`).
    """
    feature_extractor.eval()
    all_features = []
    all_target = []
    with torch.no_grad():
        for i, (images, target) in enumerate(tqdm.tqdm(data_loader)):
            images = images.to(device)
            feature = feature_extractor(images).cpu()
            # Fea = feature_extractor(images)
            # F = Fea.view(Fea.shape[0], -1)
            # feature = bottleneck(F).cpu()
            # pdb.set_trace()
            all_features.append(feature)
            all_target.append(target)
            if max_num_features is not None and i * feature.shape[0] >= max_num_features:
                break
    return torch.cat(all_features, dim=0), torch.cat(all_target)


def visualize(source_feature: torch.Tensor, target_feature: torch.Tensor,
              source_label, target_label, color_label=False, source_color='r', target_color='b',
              logpath=None, name=1):
    """
    Visualize features from different domains using t-SNE.

    Args:
        source_feature (tensor): features from source domain in shape :math:`(minibatch, F)`
        target_feature (tensor): features from target domain in shape :math:`(minibatch, F)`
        filename (str): the file name to save t-SNE
        source_color (str): the color of the source features. Default: 'r'
        target_color (str): the color of the target features. Default: 'b'

    """
    source_feature = source_feature.numpy()
    target_feature = target_feature.numpy()
    features = np.concatenate([source_feature, target_feature], axis=0)


    # pdb.set_trace()
    # map features to 2-d using TSNE
    X_tsne = TSNE(n_components=2, random_state=33).fit_transform(features)

    ############################################

    target_label_set = set(list(target_label.numpy()))
    source_label_new = [(i.item() not in target_label_set) for i in source_label]

    discard_index_source = source_label_new + ([False] * len(target_feature) )
    keep_index_source = [(not i) for i in source_label_new] + ([False] * len(target_feature) )
    index_target = ([False] * len(source_feature)) + ([True] * len(target_feature))

    source_label_cut = source_label[[(not i) for i in source_label_new]]

    # pdb.set_trace()
    plt.figure(figsize=(10, 10))

    if color_label == True:
        plt.scatter(X_tsne[discard_index_source, 0], X_tsne[discard_index_source, 1], c='gray', s=10, alpha=0.8, marker='s')
        plt.scatter(X_tsne[keep_index_source, 0], X_tsne[keep_index_source, 1], c=source_label_cut / 10, cmap=plt.cm.tab20, norm=norm, s=10, alpha=0.8, marker='s')
        plt.scatter(X_tsne[index_target, 0], X_tsne[index_target, 1], c=target_label / 10 + 0.051, cmap=plt.cm.tab20, norm=norm, s=10, alpha=0.8, marker='o')
    else:
        plt.scatter(X_tsne[discard_index_source, 0], X_tsne[discard_index_source, 1], c='gray', s=10, alpha=0.8, marker='s')
        plt.scatter(X_tsne[keep_index_source, 0], X_tsne[keep_index_source, 1], c=source_color, s=10, alpha=0.8, marker='s')
        plt.scatter(X_tsne[index_target, 0], X_tsne[index_target, 1], c=target_color, s=10, alpha=0.8, marker='o')



    #######################################
    # domain labels, 1 represents source while 0 represents target
    # domains = np.concatenate((np.ones(len(source_feature)), np.zeros(len(target_feature))))
    # # visualize using matplotlib
    # plt.figure(figsize=(10, 10))
    # plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=domains, cmap=col.ListedColormap([target_color, source_color]), s=3, alpha=0.1)
    # plt.savefig(filename)
    ###########################################

    tSNE_filename = os.path.join(logpath, '{}_TSNE.png'.format(name))
    vis_matrix_filename = os.path.join(logpath, '{}_tsne.npy'.format(name))
    keep_index_source_filename = os.path.join(logpath, '{}_keep_index_source.npy'.format(name))
    discard_index_source_filename = os.path.join(logpath, '{}_discard_index_source.npy'.format(name))
    index_target_filename = os.path.join(logpath, '{}_index_target.npy'.format(name))
    source_label_cut_filename = os.path.join(logpath, '{}_source_label_cut.npy'.format(name))
    target_label_filename = os.path.join(logpath, '{}_target_label.npy'.format(name))


    plt.savefig(tSNE_filename)

    # if vis_matrix_filename is not None:
    np.save(vis_matrix_filename, X_tsne)
    np.save(keep_index_source_filename, keep_index_source)
    np.save(discard_index_source_filename, discard_index_source)
    np.save(index_target_filename, index_target)
    np.save(source_label_cut_filename, source_label_cut)
    np.save(target_label_filename, target_label)


def obtain_label(loader, feat_ext, fc, distance='cosine', threshold=0.3):
    start_test = True
    with torch.no_grad():
        iter_test = iter(loader)
        for _ in range(len(loader)):
            data = iter_test.next()
            inputs = data[0]
            labels = data[1]
            inputs = inputs.cuda()
            feas = feat_ext(inputs)
            outputs = fc(feas)
            if start_test:
                all_fea = feas.float().cpu()
                all_output = outputs.float().cpu()
                all_label = labels.float()
                start_test = False
            else:
                all_fea = torch.cat((all_fea, feas.float().cpu()), 0)
                all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                all_label = torch.cat((all_label, labels.float()), 0)

    all_output = nn.Softmax(dim=1)(all_output)
    _, predict = torch.max(all_output, 1)

    accuracy = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size()[0])
    if distance == 'cosine':
        all_fea = torch.cat((all_fea, torch.ones(all_fea.size(0), 1)), 1)
        all_fea = (all_fea.t() / torch.norm(all_fea, p=2, dim=1)).t()

    all_fea = all_fea.float().cpu().numpy()
    K = all_output.size(1)
    aff = all_output.float().cpu().numpy()

    for _ in range(2):
        initc = aff.transpose().dot(all_fea)
        initc = initc / (1e-8 + aff.sum(axis=0)[:,None])
        cls_count = np.eye(K)[predict].sum(axis=0)
        labelset = np.where(cls_count>threshold)
        labelset = labelset[0]

        dd = cdist(all_fea, initc[labelset], distance)
        pred_label = dd.argmin(axis=1)
        predict = labelset[pred_label]

        aff = np.eye(K)[predict]

    acc = np.sum(predict == all_label.float().numpy()) / len(all_fea)
    log_str = 'Accuracy = {:.2f}% -> {:.2f}%'.format(accuracy * 100, acc * 100)

    # out_file.write(log_str + '\n')
    # out_file.flush()
    print(log_str+'\n')

    return predict.astype('int')

