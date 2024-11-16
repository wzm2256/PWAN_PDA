import pdb
from torch.autograd import grad
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
from GAN_model.util import cal_dloss, Concate_w, Entropy, cal_dloss_inc, Entropy_whole

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

    def __init__(self, num_classes, epsilon=0.1, use_gpu=True):
        super(CrossEntropyLabelSmooth, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.use_gpu = use_gpu
        # self.reduction = reduction
        self.logsoftmax = nn.LogSoftmax(dim=1)
        # self.weight = weight

    def smooth(self, targets):
        targets = torch.zeros((targets.shape[0], self.num_classes)).scatter_(1, targets.unsqueeze(1).cpu(), 1).cuda()
        smoothed_targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
        return smoothed_targets

    def forward(self, inputs, targets, weight=None, norm_type=2):
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

        if weight is None:
            return loss.mean()

        if norm_type == 1:
            weight_ = weight
        elif norm_type == 2:
            # pdb.set_trace()
            weight_ = weight/(torch.sum(weight)+1e-5)
        elif norm_type == 3:
            weight_ = weight * self.num_classes / inputs.shape[0]
        elif norm_type == 4:
            weight_ = weight / torch.max(weight) / inputs.shape[0]
        elif norm_type == 5:
            weight_ = weight / inputs.shape[0]
        return torch.sum(weight_*loss)
        # elif norm_type ==1:
        #     weight_ = torch.nn.functional.relu(weight - 1e-5)
        #     return torch.sum(weight * loss) / (torch.sum(weight) + 1e-5)


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
    all_logit = []
    with torch.no_grad():
        for i, (images, target, index) in enumerate(tqdm.tqdm(data_loader)):
            images = images.to(device)
            feature, logit = feature_extractor(images)
            feature = feature.cpu()
            logit = logit.cpu()
            all_features.append(feature)
            all_target.append(target)
            all_logit.append(logit)
            if max_num_features is not None and i * feature.shape[0] >= max_num_features:
                break
    return torch.cat(all_features, dim=0), torch.cat(all_target), torch.cat(all_logit)


# norm_extract(source_feature, target_feature, domain_D, args.d_weight_label, my_CrossEntropy,
#              args.label_smooth == 1 and args.cat_smooth == 1)


def norm_extract(source_feature, target_feature, source_label, target_label, train_bs,
                 source_logit, target_logit, domain_D, d_weight_label, my_CrossEntropy, smooth):

    if d_weight_label < 10.0:
        if smooth == 1:
            ys_onehot = my_CrossEntropy.smooth(source_label)
        else:
            ys_onehot = F.one_hot(source_label, num_classes=my_CrossEntropy.num_classes).float()

        yt_predict = F.softmax(target_logit, -1)
        # pdb.set_trace()
        cor_s_d = Concate_w(source_feature.detach(), ys_onehot.to('cpu'), weight=d_weight_label)
        cor_t_d = Concate_w(target_feature.detach(), yt_predict.detach(), weight=d_weight_label)
    elif d_weight_label > 10.0 and d_weight_label < 20.0:
        yt_predict = F.softmax(target_logit, -1)
        ys_predict = F.softmax(source_logit, -1)
        cor_s_d = Concate_w(source_feature.detach(), ys_predict.detach(), weight=d_weight_label - 10.0)
        cor_t_d = Concate_w(target_feature.detach(), yt_predict.detach(), weight=d_weight_label - 10.0)
    elif d_weight_label > 20.0:
        cor_s_d = Concate_w(source_feature.detach(), source_logit.detach(), weight=d_weight_label - 20.0)
        cor_t_d = Concate_w(target_feature.detach(), target_logit.detach(), weight=d_weight_label - 20.0)

    # potential_r = domain_D(cor_s_d)
    # potential_f = domain_D(cor_t_d)

    b_r = cor_s_d.shape[0] // train_bs
    r_norm = []
    domain_D.to('cpu')
    for i in range(b_r + 1):
        if (i + 1) * train_bs <= cor_s_d.shape[0]:
            batch = cor_s_d[i * train_bs : (i + 1) * train_bs]
        else:
            batch = cor_s_d[i * train_bs: ]

        # batch = batch.to('cuda')
        batch.requires_grad_(True)

        potential_r = domain_D(batch)
        gradients = grad(outputs=potential_r, inputs=batch,
                         grad_outputs=torch.ones(potential_r.size()).contiguous())[0]
        # pdb.set_trace()
        r_norm.append(gradients.norm(2, dim=1).detach().cpu())
    source_norm = torch.cat(r_norm)

    b_f = cor_t_d.shape[0] // train_bs
    f_norm = []
    for i in range(b_f + 1):
        if (i + 1) * train_bs <= cor_t_d.shape[0]:
            batch = cor_t_d[i * train_bs : (i + 1) * train_bs]
        else:
            batch = cor_t_d[i * train_bs: ]

        # batch = batch.to('cuda')
        batch.requires_grad_(True)

        potential_f = domain_D(batch)
        gradients = grad(outputs=potential_f, inputs=batch,
                         grad_outputs=torch.ones(potential_f.size()).contiguous())[0]
        # pdb.set_trace()
        f_norm.append(gradients.norm(2, dim=1).detach().cpu())
    target_norm = torch.cat(f_norm)

    domain_D.to('cuda')
    # pdb.set_trace()
    return source_norm, target_norm, cor_s_d, cor_t_d

def visualize(source_feature: torch.Tensor, target_feature: torch.Tensor,
              source_label, target_label, source_norm, target_norm, source_logit, target_logit,
              color_label=False, source_color='r', target_color='b',
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
        plt.scatter(X_tsne[discard_index_source, 0], X_tsne[discard_index_source, 1], c='gray', s=10, alpha=0.3, marker='s')
        plt.scatter(X_tsne[keep_index_source, 0], X_tsne[keep_index_source, 1], c=source_color, s=10, alpha=0.3, marker='s')
        plt.scatter(X_tsne[index_target, 0], X_tsne[index_target, 1], c=target_color, s=10, alpha=0.3, marker='o')



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
    source_label_full_filename = os.path.join(logpath, '{}_source_label_full.npy'.format(name))
    source_label_predict_filename = os.path.join(logpath, '{}_source_label_predict.npy'.format(name))
    target_label_predict_filename = os.path.join(logpath, '{}_target_label_predict.npy'.format(name))
    source_norm_filename = os.path.join(logpath, '{}_source_norm.npy'.format(name))
    target_norm_filename = os.path.join(logpath, '{}_target_norm.npy'.format(name))


    plt.savefig(tSNE_filename)

    # if vis_matrix_filename is not None:
    np.save(vis_matrix_filename, X_tsne)
    np.save(keep_index_source_filename, keep_index_source)
    np.save(discard_index_source_filename, discard_index_source)
    np.save(index_target_filename, index_target)
    np.save(source_label_cut_filename, source_label_cut)
    np.save(target_label_filename, target_label)
    np.save(target_label_predict_filename, target_logit)
    np.save(source_label_full_filename, source_label)
    np.save(source_label_predict_filename, source_logit)

    np.save(source_norm_filename, source_norm.numpy())
    np.save(target_norm_filename, target_norm.numpy())


    plt.figure(figsize=(10, 10))
    plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=torch.cat([source_norm, target_norm]).numpy(), cmap=plt.cm.viridis, s=3, alpha=0.1)
    norm_filename = os.path.join(logpath, '{}_norm.png'.format(name))

    plt.savefig(norm_filename)

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

def marginloss(yHat, y, classes=65, alpha=1, weight=None):
    batch_size = len(y)
    classes = classes
    yHat = F.softmax(yHat, dim=1)
    Yg = torch.gather(yHat, 1, torch.unsqueeze(y, 1))#.detach()
    Yg_ = (1 - Yg) + 1e-7  # avoiding numerical issues (first)
    Px = yHat / Yg_.view(len(yHat), 1)
    Px_log = torch.log(Px + 1e-10)  # avoiding numerical issues (second)
    y_zerohot = torch.ones(batch_size, classes).scatter_(
        1, y.view(batch_size, 1).data.cpu(), 0)

    output = Px * Px_log * y_zerohot.cuda()
    loss = torch.sum(output, dim=1)/ np.log(classes - 1)
    Yg_ = Yg_ ** alpha
    if weight is not None:
        # pdb.set_trace()
        weight *= (Yg_.view(len(yHat), )/ Yg_.sum())
    else:
        weight = (Yg_.view(len(yHat), )/ Yg_.sum())

    weight = weight.detach()
    loss = torch.sum(weight * loss) / torch.sum(weight)

    return loss

def source_only(network, step, optimizer, lr_scheduler, schedule_param, dset_loaders, loss, cot_weight):
    for i in range(step):
        optimizer_bak = lr_scheduler(optimizer, i, **schedule_param)
        # pdb.set_trace()
        # train one iter
        if i % len(dset_loaders["source"]) == 0:
            iter_source = iter(dset_loaders["source"])


        xs, ys, ind_s = iter_source.next()
        xs, ys = xs.cuda(), ys.cuda()

        _, f = network(xs)
        # pdb.set_trace()
        classifier_loss = loss(f, ys)
        #############
        # cot_loss = marginloss(f, ys, classes=network.class_num, alpha=1, weight=None)
        #############
        cot_loss = torch.tensor(0.).cuda()
        #############
        Total_loss = classifier_loss + cot_weight * cot_loss

        optimizer_bak.zero_grad()
        Total_loss.backward()
        optimizer_bak.step()

        cls_acc = accuracy(f, ys)[0]

        if i % 100 == 0:
            print('PreTrain:{}\t---cls_loss:{}\ts_acc:{}'.format(i, np.round(classifier_loss.cpu().item(), 3),
                                                                 np.round(cls_acc.cpu().item(), 3)))
