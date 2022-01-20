import argparse
import os, random, pdb, math, sys
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import network, my_loss
import lr_schedule, data_list
from utils import *
# import ot 
from GAN_model.D import D2
from GAN_model.Grad import Grad_Penalty
import torch.optim as optim
from GAN_model.util import cal_dloss, Concate_w, Entropy
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn


def image_train(pre_process=0, resize_size=256, crop_size=224):
    if pre_process == 0:
        return transforms.Compose([
            transforms.Resize((resize_size, resize_size)),
            transforms.RandomCrop(crop_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    elif pre_process == 1:
        return transforms.Compose([
            transforms.Resize((resize_size, resize_size)),
            transforms.RandomResizedCrop(crop_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    elif pre_process == 2:
        raise ValueError('Not implement yet')


def image_test(resize_size=256, crop_size=224):
    return  transforms.Compose([
        transforms.Resize((resize_size, resize_size)),
        transforms.CenterCrop(crop_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

def image_classification(loader, model):
    start_test = True
    with torch.no_grad():
        iter_test = iter(loader["test"])
        for i in range(len(loader['test'])):
            data = iter_test.next()
            inputs = data[0]
            labels = data[1]
            inputs = inputs.cuda()
            _, outputs = model(inputs)
            if start_test:
                all_output = outputs.float().cpu()
                all_label = labels.float()
                start_test = False
            else:
                all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                all_label = torch.cat((all_label, labels.float()), 0)
    _, predict = torch.max(all_output, 1)
    accuracy = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size()[0])
    #mean_ent = torch.mean(my_loss.Entropy(torch.nn.Softmax(dim=1)(all_output))).cpu().data.item()

    #hist_tar = torch.nn.Softmax(dim=1)(all_output).sum(dim=0)
    #hist_tar = hist_tar / hist_tar.sum()
    return accuracy#, hist_tar, mean_ent

def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ## prepare data
    train_bs, test_bs = args.batch_size, args.batch_size * 2

    dsets = {}
    dsets["source"] = data_list.ImageList(open(args.s_dset_path).readlines(), transform=image_train(pre_process=args.pre_process))
    dsets["target"] = data_list.ImageList(open(args.t_dset_path).readlines(), transform=image_train(pre_process=args.pre_process))
    dsets["test"] = data_list.ImageList(open(args.t_dset_path).readlines(), transform=image_test())

    dset_loaders = {}
    if args.balance == 0:
        dset_loaders["source"] = DataLoader(dsets["source"], batch_size=train_bs, shuffle=True, num_workers=args.worker,
                                            prefetch_factor=8,
                                            drop_last=True)
    else:
        source_labels = torch.tensor([i[1] for i in dsets["source"].imgs])
        train_batch_sampler = BalancedBatchSampler(source_labels, batch_size=train_bs)
        dset_loaders["source"] = DataLoader(dsets["source"], batch_sampler=train_batch_sampler, num_workers=args.worker,
                                            prefetch_factor=8
                                            )
    #########
    dset_loaders["target"] = DataLoader(dsets["target"], batch_size=train_bs, shuffle=True,
                                        num_workers=args.worker, drop_last=True,
                                        prefetch_factor=8
                                        )
    dset_loaders["test"]   = DataLoader(dsets["test"], batch_size=test_bs, shuffle=False,
                                        num_workers=args.worker,
                                        prefetch_factor=8
                                        )

    if "ResNet" in args.net:
        params = {"resnet_name":args.net, "use_bottleneck":True, "bottleneck_dim":256, "new_cls":True, 'class_num': args.class_num, 'init_fc':args.init_fc}
        base_network = network.ResNetFc(**params)
    
    if "VGG" in args.net:
        params = {"vgg_name":args.net, "use_bottleneck":True, "bottleneck_dim":256, "new_cls":True, 'class_num': args.class_num}
        base_network = network.VGGFc(**params)

    # pdb.set_trace()
    base_network = base_network.cuda()

    parameter_list = base_network.get_parameters()
    base_network = torch.nn.DataParallel(base_network).cuda() 

    domain_D = D2(in_feature=256 + args.class_num, hidden_size=args.d_hidden, norm_I=args.d_norm,
                  leaky=args.d_leaky).to(device)

    G_P = Grad_Penalty(100, gamma=1, device='cuda')

    # pdb.set_trace()

    D_Net_parames = [j for (i, j) in domain_D.named_parameters() if i != 'h']
    optimizer_D = optim.RMSprop(D_Net_parames, lr=args.lr_D)

    ## set optimizer
    if args.opt_G == 0:
        optimizer_config = {"type":torch.optim.SGD, "optim_params":
                            {'lr':args.lr, "momentum":0.9, "weight_decay":5e-4, "nesterov":True}, 
                            "lr_type":"inv", "lr_param":{"lr":args.lr, "gamma":0.001, "power":0.75}
                        }
    elif args.opt_G == 1:
        optimizer_config = {"type":torch.optim.RMSprop, "optim_params":
                            {'lr':args.lr, "weight_decay":5e-4,}, 
                            "lr_type":"inv", "lr_param":{"lr":args.lr, "gamma":0.001, "power":0.75}
                        }
    elif args.opt_G == 2:
        optimizer_config = {"type":torch.optim.Adam, "optim_params":
                            {'lr':args.lr, "weight_decay":5e-4,}, 
                            "lr_type":"inv", "lr_param":{"lr":args.lr, "gamma":0.001, "power":0.75}
                        }
    elif args.opt_G == 3:
        optimizer_config = {"type": torch.optim.RMSprop, "optim_params":
                            {'lr': args.lr},
                            "lr_type": "inv", "lr_param": {"lr": args.lr, "gamma": 0.001, "power": 0.75}
                            }
    optimizer = optimizer_config["type"](parameter_list,**(optimizer_config["optim_params"]))

    param_lr = []
    for param_group in optimizer.param_groups:
        param_lr.append(param_group["lr"])
    schedule_param = optimizer_config["lr_param"]
    lr_scheduler = lr_schedule.schedule_dict[optimizer_config["lr_type"]]

    feature_extractor = nn.Sequential(base_network.module.feature_layers, nn.Flatten(), base_network.module.bottleneck).to('cuda')
    for i in range(args.max_iterations + 1):

        if (i % args.test_interval == 0 and i > 0) or (i == args.max_iterations):
        # if (i % args.test_interval == 0) or (i == args.max_iterations):
            # obtain the class-level weight and evalute the current model
            base_network.train(False)
            temp_acc = image_classification(dset_loaders, base_network)
            args.writer.add_scalar('Test/t_acc', temp_acc, i // args.test_interval)


            source_feature = collect_feature(dset_loaders["source"], feature_extractor, device)
            target_feature = collect_feature(dset_loaders["target"], feature_extractor, device)
            # plot t-SNE
            tsne_name = '{}_TSNE.png'.format(i)
            tSNE_filename = os.path.join(args.Log_path, tsne_name)
            visualize(source_feature, target_feature, tSNE_filename)
            print("Saving t-SNE to", tSNE_filename)


        base_network.train(True)
        optimizer = lr_scheduler(optimizer, i, **schedule_param)
        
        # train one iter
        if i % len(dset_loaders["source"]) == 0:
            iter_source = iter(dset_loaders["source"])
        if i % len(dset_loaders["target"]) == 0:
            iter_target = iter(dset_loaders["target"])

        xs, ys = iter_source.next()
        xt, yt = iter_target.next()
        xs, xt, ys = xs.cuda(), xt.cuda(), ys.cuda()


        ######
        # g_xs, f_g_xs = base_network(xs)
        # g_xt, f_g_xt = base_network(xt)
        ######
        x = torch.cat((xs, xt), dim=0)
        g, f = base_network(x)
        g_xs, g_xt = g.chunk(2, dim=0)
        f_g_xs, f_g_xt = f.chunk(2, dim=0)
        ######

        # pdb.set_trace()
        ###############################################################
        ys_onehot = F.one_hot(ys, num_classes=args.class_num).float()

        yt_predict = F.softmax(f_g_xt, -1)
        cor_s_d = Concate_w(g_xs.detach(), ys_onehot.to('cuda'), weight=args.d_weight_label)
        cor_t_d = Concate_w(g_xt.detach(), yt_predict.detach(), weight=args.d_weight_label)

        cor_s_d.requires_grad_(True)
        cor_t_d.requires_grad_(True)
        for d in range(args.d_iter):
            potential_r = domain_D(cor_s_d)
            potential_f = domain_D(cor_t_d)
            d_loss = cal_dloss(potential_r, potential_f, args.point_mass * args.q ** i)

            if d == 0:
                gp_loss, M = G_P(d_loss, [cor_s_d, cor_t_d])
                args.writer.add_scalar('Train/M_grad', M, i)
            else:
                gp_loss = torch.tensor(0.)
                M = torch.tensor(0.)

            d_loss_all = d_loss + gp_loss
            print(
                ' d_iter ' + str(d) + ' d_loss: ' + str(np.array(d_loss.item()).round(6)) + '\t' +
                'gp_loss: ' + str(np.array(gp_loss.item()).round(6)) + '\t' + ' M_grad:' + str(np.array(M.item()).round(6)))

            optimizer_D.zero_grad()
            d_loss_all.backward()
            optimizer_D.step()

        cor_s_g = Concate_w(g_xs, ys_onehot.to('cuda'), weight=args.d_weight_label)
        cor_t_g = Concate_w(g_xt, yt_predict, weight=args.d_weight_label)

        potential_r_g = domain_D(cor_s_g)
        potential_f_g = domain_D(cor_t_g)
        transfer_loss = -cal_dloss(potential_r_g, potential_f_g, args.point_mass * args.q ** i)

        #########################################################################
        # transfer_loss = torch.tensor(0.)
        # d_loss = torch.tensor(0.)
        #########################################################################

        classifier_loss = torch.nn.CrossEntropyLoss()(f_g_xs, ys)
        total_loss = classifier_loss * args.cls_weight + transfer_loss * args.trade_off + args.entropy * Entropy(f_g_xt).mean() + args.entropy_s * Entropy(f_g_xs).mean()
            
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        cls_acc = accuracy(f_g_xs, ys)[0]
        tgt_acc = accuracy(f_g_xt, yt.to('cuda'))[0]

        args.writer.add_scalar('Train/cls_loss', classifier_loss, i)
        args.writer.add_scalar('Train/d_loss', transfer_loss, i)
        args.writer.add_scalar('Train/s_acc', cls_acc, i)
        args.writer.add_scalar('Train/t_acc', tgt_acc, i)



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='BA3US for Partial Domain Adaptation')
    parser.add_argument('--s', type=int, default=0, help="source")
    parser.add_argument('--t', type=int, default=1, help="target")
    parser.add_argument('--seed', type=int, default=0, help="random seed")
    parser.add_argument('--max_iterations', type=int, default=5000, help="max iterations")
    parser.add_argument('--batch_size', type=int, default=65, help="batch_size")
    parser.add_argument('--worker', type=int, default=4, help="number of workers") 
    parser.add_argument('--net', type=str, default='ResNet50', choices=["ResNet50", "VGG16"])
    
    parser.add_argument('--dset', type=str, default='office_home', choices=["office", "office_home", "imagenet_caltech"])
    parser.add_argument('--test_interval', type=int, default=500, help="interval of two continuous test phase")
    
    ########
    parser.add_argument('--lr', type=float, default=1e-4, help="learning rate")
    parser.add_argument('--d_hidden', default=256, type=int)
    parser.add_argument('--d_norm', default=0, type=int)
    parser.add_argument('--d_leaky', default=0.2, type=float)
    parser.add_argument('--trade_off', default=1., type=float,
                        help='the trade-off hyper-parameter for transfer loss')
    parser.add_argument('--lr_D', type=float, default=1e-4)
    parser.add_argument('--d_weight_label', type=float, default=0.01)
    parser.add_argument('--d_iter', type=int, default=2)
    parser.add_argument('--point_mass', type=float, default=1.0)
    parser.add_argument('--cls_weight', type=float, default=1.)
    parser.add_argument('--opt_G', type=int, default=0, help='0: SGD momentum 1:RMSprop 2: SGD 3: RMSProp no schedule')
    parser.add_argument('--entropy', type=float, default=0., help='entropy weight')
    parser.add_argument('--entropy_s', type=float, default=0., help='entropy weight')
    parser.add_argument('--pre_process', type=int, default=0, help='')
    parser.add_argument('--init_fc', type=int, default=0, help='')

    parser.add_argument('--pm_ratio', type=float, default=1., help='point mass decrease ratio at the end of the training.')

    args = parser.parse_args()

    if args.dset == 'office_home':
        names = ['Art', 'Clipart', 'Product', 'RealWorld']
        k = 25
        args.class_num = 65
        args.test_interval = 500
        if args.batch_size == 65:
            args.balance = 1
        else:
            args.balance = 0

    if args.dset == 'office':
        names = ['amazon', 'dslr', 'webcam']
        k = 10
        args.class_num = 31
        args.max_iterations = 2000
        args.test_interval = 200
        args.lr=1e-4

    if args.dset == 'imagenet_caltech':
        names = ['imagenet', 'caltech']
        k = 84
        args.class_num = 1000
        if args.s == 1:
            args.class_num = 256

        args.max_iterations = 40000
        args.test_interval = 4000
        args.lr=1e-3

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    data_folder = './data/'
    args.s_dset_path = data_folder + args.dset + '/' + names[args.s] + '_list.txt'
    args.t_dset_path = data_folder + args.dset + '/' + names[args.t] + '_' + str(k) + '_list.txt'

    args.name = names[args.s][0].upper() + names[args.t][0].upper()


    setting_name = 'h_{}_N_{}_ly_{}_WO_{}_lrD_{}_lrG_{}_WL_{}_ItD_{}_PM_{}_Wcls_{}_opt_G{}_ent_{}_ent_s_{}_bs_{}_pre_{}' \
                   '_init{}_r_{}'.format(
                    args.d_hidden, args.d_norm, args.d_leaky, args.trade_off, args.lr_D, args.lr, args.d_weight_label,
                    args.d_iter, args.point_mass, args.cls_weight, args.opt_G, args.entropy, args.entropy_s,
                    args.batch_size, args.pre_process, args.init_fc, args.pm_ratio)

    task_name = args.name
    args.Log_path = os.path.join('LOG', setting_name, task_name)
    if not os.path.isdir(args.Log_path):
        os.makedirs(args.Log_path)

    config_path = os.path.join(args.Log_path, 'config.txt')
    f = open(config_path, 'w')
    config_Dict = vars(args)
    for key, value in config_Dict.items():
        f.write(str(key))
        f.write('\t')
        f.write(str(value))
        f.write('\n')
    f.close()

    tf_log = os.path.join(args.Log_path, '0', 'LOG')
    args.writer = SummaryWriter(tf_log)

    args.q = np.exp(np.log(args.pm_ratio) / args.max_iterations)

    train(args)
