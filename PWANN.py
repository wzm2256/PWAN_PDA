import copy
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
from GAN_model.Grad import Grad_Penalty, Grad_Penalty_w
import torch.optim as optim
from GAN_model.util import cal_dloss, Concate_w, Entropy, cal_dloss_inc, Entropy_whole
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
import datasets.partial as datasets
from datasets.partial import default_partial as partial_dataset
from torch.optim.lr_scheduler import LambdaLR
import torch.backends.cudnn as cudnn
import network_adv as network_adv
import pdb

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
        return transforms.Compose([
            transforms.Resize((resize_size, resize_size)),
            transforms.RandomCrop(crop_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    elif pre_process == 3:
        return transforms.Compose([
            transforms.Resize((resize_size, resize_size)),
            transforms.CenterCrop(crop_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        raise ValueError('Not implement yet')


def image_test(resize_size=256, crop_size=224):
    return  transforms.Compose([
        transforms.Resize((resize_size, resize_size)),
        transforms.CenterCrop(crop_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

def image_classification(loader, model, threshold=10, feature=1):
    start_test = True
    with torch.no_grad():
        iter_test = iter(loader["test"])
        for i in range(len(loader['test'])):
            data = iter_test.next()
            inputs = data[0]
            labels = data[1]
            inputs = inputs.cuda()
            # pdb.set_trace()
            ####
            if feature == 1:
                _, outputs = model(inputs)
            else:
                outputs = model(inputs)
            if start_test:
                # all_fea = feas.float().cpu()
                all_output = outputs.float().cpu()
                all_label = labels.float()
                start_test = False
            else:
                # all_fea = torch.cat((all_fea, feas.float().cpu()), 0)
                all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                all_label = torch.cat((all_label, labels.float()), 0)
    _, predict = torch.max(all_output, 1)
    prob_output = nn.Softmax(dim=1)(all_output)
    accuracy = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size()[0])
    # mean_ent = torch.mean(my_loss.Entropy(prob_output)).cpu().data.item()

    hist_tar = prob_output.sum(dim=0)
    hist_tar = hist_tar / hist_tar.sum()

    return accuracy, hist_tar, predict, None

def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ## prepare data
    train_bs, test_bs = args.batch_size, args.batch_size * 2

    dsets = {}
    dataset = datasets.__dict__[args.dset]
    p_dataset = partial_dataset(dataset, args.tcls)
    dsets["source"] = dataset(root=args.dset_path, task=args.s_name, download=True, transform=image_train(pre_process=args.pre_process))
    dsets["target"] = p_dataset(root=args.dset_path, task=args.t_name, download=True, transform=image_train(pre_process=args.pre_process))
    dsets["test"] = p_dataset(root=args.dset_path, task=args.t_name, download=True, transform=image_test())

    # pdb.set_trace()
    # print('length source {}'.format(len(dsets["source"])))
    # print('length target {}'.format(len(dsets["target"])))

    dset_loaders = {}
    source_labels = torch.tensor(list(zip(*(dsets["source"].samples)))[1])
    source_label_count = np.array([np.sum(source_labels.numpy() == i) for i in range(args.class_num)])
    source_label_dis = source_label_count / np.sum(source_label_count)

    if args.balance == 0:
        dset_loaders["source"] = DataLoader(dsets["source"], batch_size=train_bs, shuffle=True, num_workers=args.worker,
                                            prefetch_factor=8,
                                            drop_last=True)
    else:
        train_batch_sampler = BalancedBatchSampler(source_labels, batch_size=train_bs)
        dset_loaders["source"] = DataLoader(dsets["source"], batch_sampler=train_batch_sampler, num_workers=args.worker,
                                            prefetch_factor=8
                                            )

    dset_loaders["target"] = DataLoader(dsets["target"], batch_size=train_bs, shuffle=True,
                                        num_workers=args.worker, drop_last=True,
                                        prefetch_factor=8,
                                        )
    dset_loaders["test"]   = DataLoader(dsets["test"], batch_size=test_bs, shuffle=False,
                                        num_workers=args.worker,
                                        prefetch_factor=8
                                        )

    dset_loaders["warm_source"] = DataLoader(dsets["source"], batch_size=32, shuffle=True, num_workers=args.worker,
                                        prefetch_factor=8, drop_last=True)
    dset_loaders["warm_target"] = DataLoader(dsets["target"], batch_size=32, shuffle=True,
                                        num_workers=args.worker, drop_last=True, prefetch_factor=8)


    target_label_set = list(set(list(zip(*(dsets["test"].samples)))[1]))
    True_class = np.zeros(args.class_num)
    for i in target_label_set:
        True_class[i] = 1

    # pdb.set_trace()
    # if args.auto_ratio == 1:
        # print('Dataset size -- source: {} \t target: {} \t ratio: {}'.format(len(dsets["source"]),
        #         len(dsets["target"]), len(dsets["target"]) / len(dsets["source"])))
        # set_ratio = len(dsets["target"]) / len(dsets["source"])
        # args.point_mass = set_ratio
        # print('Setting point mass to: {}'.format(set_ratio))


    if "ResNet" in args.net:
        if args.dset == 'ImageNetCaltech__':
            params = {"resnet_name": args.net, "use_bottleneck": False, "new_cls": False, 'class_num': args.class_num}
        else:
            params = {"resnet_name":args.net, "use_bottleneck":True, "bottleneck_dim":args.bottle_dim, "new_cls":True,
                      'class_num': args.class_num, 'init_fc':args.init_fc, "NoRelu":args.NoRelu, "normalize":args.normalize,
                      'leaky':args.leaky}
        #########
        base_network = network.ResNetFc(**params)
        #########
        # params_adv = {"resnet_name": args.net, "bottleneck_dim": 256,
        #           'class_num': 126, "radius": 20, "normalize_classifier": False}
        # base_network = network_adv.ResNetFc(**params_adv)
        #########

        base_network_tmp = network.ResNetFc(**params)
        # base_network_tmp = copy.deepcopy(base_network)


    base_network_tmp = base_network_tmp.cuda()
    parameter_list_tmp = base_network_tmp.get_parameters()

    base_network = base_network.cuda()
    parameter_list = base_network.get_parameters()

    base_network = torch.nn.DataParallel(base_network).cuda() 

    if args.label_smooth == 1:
        my_CrossEntropy = CrossEntropyLabelSmooth(args.class_num)
    else:
        my_CrossEntropy = CrossEntropyLabelSmooth(args.class_num, epsilon=0.0)
        # torch.nn.CrossEntropyLoss()

    # pdb.set_trace()
    domain_D = D2(in_feature=base_network.module.output_num() + args.class_num, hidden_size=args.d_hidden, norm_I=args.d_norm,
                  leaky=args.d_leaky).to(device)

    if args.WG == 0:
        G_P = Grad_Penalty(1000, gamma=1, device='cuda')
    else:
        G_P = Grad_Penalty_w(1000, gamma=1, device='cuda')
    # pdb.set_trace()

    D_Net_parames = [j for (i, j) in domain_D.named_parameters() if i != 'h']

    if args.opt_D == 0:
        optimizer_D = optim.RMSprop(D_Net_parames, lr=args.lr_D)
    elif args.opt_D == 1:
        optimizer_D = optim.Adam(D_Net_parames, lr=args.lr_D)
    elif args.opt_D == 2:
        optimizer_D = optim.Adam(D_Net_parames, lr=args.lr_D, betas=(0, 0.99))
    elif args.opt_D == 3:
        optimizer_D = optim.Adam(D_Net_parames, lr=args.lr_D, betas=(0, 0.5))
    elif args.opt_D == 4:
        optimizer_config_G = {"type":torch.optim.Adam, "optim_params":
                            {'lr':args.lr_D, "weight_decay":5e-4}
                            }
        optimizer_D = optim.Adam(D_Net_parames,**(optimizer_config_G["optim_params"]))
    if args.opt_D == 5:
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
    elif args.opt_G == 4:
        optimizer_config = {"type": torch.optim.RMSprop, "optim_params":
                            {'lr': args.lr, "weight_decay":5e-4,},
                            "lr_type": "null", "lr_param": {"lr": args.lr}
                            }
    elif args.opt_G == 5:
        optimizer_config = {"type":torch.optim.Adam, "optim_params":
                            {'lr':args.lr, "weight_decay":5e-4, 'betas':(0, 0.5)},
                            "lr_type":"inv", "lr_param":{"lr":args.lr, "gamma":0.001, "power":0.75}
                        }

    warm_optimizer_config = {"type": torch.optim.SGD, "optim_params":
        {'lr': 1e-2, "momentum": 0.9, "weight_decay": 5e-4, "nesterov": True},
                        "lr_type": "inv", "lr_param": {"lr": 1e-2, "gamma": 0.001, "power": 0.75}
                        }
    warm_optimizer = warm_optimizer_config["type"](parameter_list, **(warm_optimizer_config["optim_params"]))
    warm_schedule_param = warm_optimizer_config["lr_param"]

    optimizer = optimizer_config["type"](parameter_list,**(optimizer_config["optim_params"]))
    optimizer_bak = optimizer_config["type"](parameter_list, **(optimizer_config["optim_params"]))
    optimizer_tmp = optimizer_config["type"](parameter_list_tmp, **(optimizer_config["optim_params"]))

    schedule_param = optimizer_config["lr_param"]
    lr_scheduler = lr_schedule.schedule_dict[optimizer_config["lr_type"]]

    if args.opt_D == 5:
        lr_scheduler_D = LambdaLR(optimizer_D, lambda x: args.lr_D * (1. + 0.001 * float(x)) ** (-0.75))

    # feature_extractor = nn.Sequential(base_network.module.feature_layers, nn.Flatten(), base_network.module.bottleneck).to('cuda')

    source_only(base_network.module, args.pre_train, optimizer_bak, lr_scheduler, schedule_param, dset_loaders,
                my_CrossEntropy, args.cot_weight)

    if args.dset == 'ImageNetCaltech':
        # We use the pre-trained model directly. Do not need to prepare for a pre-tained model.
        from torchvision import models
        evaluate_net = models.resnet50(pretrained=True).to('cuda')
    elif args.auto_ratio > 0 and args.pre_train < args.auto_ratio:
        source_only(base_network_tmp, args.auto_ratio, optimizer_tmp, lr_scheduler, schedule_param, dset_loaders,
                my_CrossEntropy, args.cot_weight)
        evaluate_net = base_network_tmp
    else:
        evaluate_net = base_network.module


    if args.auto_ratio >= 0:
        ################################################
        # source_only(base_network_tmp, args.auto_ratio, optimizer_tmp, lr_scheduler, schedule_param, dset_loaders,
        #             my_CrossEntropy, args.cot_weight)
        ################################
        # base_network_tmp.train(False)
        # temp_acc, class_weight, predict_label, acc = image_classification(dset_loaders, base_network_tmp)
        ################################asdfa
        evaluate_net.train(False)
        if args.dset == 'ImageNetCaltech':
            feature_cls = 0
        else:
            feature_cls = 1
        temp_acc, class_weight, predict_label, acc = image_classification(dset_loaders, evaluate_net, feature=feature_cls)

        predict_label_count = np.array([np.sum(predict_label.numpy() == i) for i in range(args.class_num)])
        predict_label_dis = predict_label_count / np.sum(predict_label_count)
        print('True ratio:{}'.format(np.sum(source_label_dis[:args.tcls])))
        print('Predict label ratio {}-----{}'.format(np.sum(predict_label_dis > 1 / args.class_num),
                                                     np.sum(predict_label_dis > 1 / args.class_num) / args.class_num))
        if args.balance == 1:
            weight_ratio = np.sum((predict_label_dis > 1 / args.class_num)) / args.class_num
        else:
            weight_ratio = np.sum((predict_label_dis > 1 / args.class_num) * source_label_dis)
        # pdb.set_trace()
        print('Predict label weighted ratio {}. Set point mass to {}.'.format(weight_ratio, weight_ratio))

        args.point_mass = weight_ratio
        f = open(os.path.join(args.Log_path, 'mass.txt'), 'w')

        f.write('True ratio:{}\n'.format(np.sum(source_label_dis[:25])))
        f.write(args.s_name + '\t---------------------\t' + args.t_name + ' acc: ' + str(temp_acc) + '\n')
        f.write('Predict label ratio {}-----{}\n'.format(np.sum(predict_label_dis > 1 / args.class_num),
                                                     np.sum(predict_label_dis > 1 / args.class_num) / args.class_num))
        f.write('Predict weight ratio {}-----{}\n'.format(np.sum(class_weight.numpy() > 1 / args.class_num), np.sum(
            class_weight.numpy() > 1 / args.class_num) / args.class_num))
        f.write('Predict label weighted ratio {}\n'.format(
            np.sum((predict_label_dis > 1 / args.class_num) * source_label_dis)))
        f.write('Predict weight weighted ratio {}\n'.format(
            np.sum((class_weight.numpy() > 1 / args.class_num) * source_label_dis)))


        f.write('Setting point mass to: ' + str(weight_ratio))
        f.close()
        evaluate_net.train(True)
    # del base_network_tmp

    for i in range(args.max_iterations + 1):

        if (i % args.test_interval == 0) or (i == args.max_iterations):
        # if i == 1:
            # obtain the class-level weight and evalute the current model
            base_network.train(False)
            print('Start testing.....')
            if i == 0:
                if args.skip_first == 0:
                    temp_acc, class_weight, predict_label, acc = image_classification(dset_loaders, base_network)
                elif args.skip_first == 1:
                    # use pretrained estimated class weight
                    temp_acc, _, _, acc = image_classification(dset_loaders, base_network)
                elif args.skip_first == 2:
                    # use uniformed class weight and predict label
                    temp_acc, _, predict_label_debug, acc = image_classification(dset_loaders, base_network)
                    class_weight = torch.ones(args.class_num) / args.class_num

            else:
                temp_acc, class_weight, predict_label, acc = image_classification(dset_loaders, base_network)
            # source_label_count
            predict_label_count = np.array([np.sum(predict_label.numpy() == i) for i in range(args.class_num)])
            predict_label_dis = torch.from_numpy(predict_label_count / np.sum(predict_label_count)).to('cuda')

            if i == 0 and args.skip_first == 2:
                predict_label_dis = class_weight
            # print(class_weight)
            print('Finish testing.....')
            args.writer.add_scalar('Test/t_acc', temp_acc, i // args.test_interval)

            # if i % 5000 == 0:
            # if i > -1:
            # if i == args.max_iterations or i == args.warm:
            # if i == 1:
            if (i % args.test_interval == 0) or (i == args.max_iterations):

            #     source_feature, source_label = collect_feature(dset_loaders["source"], feature_extractor, device, 10000)
            #     target_feature, target_label = collect_feature(dset_loaders["target"], feature_extractor, device, 10000)
                source_feature, source_label, source_logit = collect_feature(dset_loaders["source"], base_network, device, 10000)
                # target_feature, target_label, target_logit = collect_feature(dset_loaders["target"], base_network, device, 10000)
                target_feature, target_label, target_logit = collect_feature(dset_loaders["test"], base_network, device, 10000)

                # pdb.set_trace()
                source_norm, target_norm, cor_s, cor_t = norm_extract(source_feature, target_feature, source_label, target_label, train_bs, source_logit, target_logit,
                             domain_D, args.d_weight_label, my_CrossEntropy, args.label_smooth == 1 and args.cat_smooth == 1)
                visualize(cor_s, cor_t, source_label, target_label, source_norm, target_norm, source_logit, target_logit,
                          color_label=args.color_label, logpath=args.Log_path, name=i)
                # visualize(source_feature, target_feature, source_label, target_label, source_norm, target_norm,
                #                       color_label=args.color_label, logpath=args.Log_path, name=i)
                    # print("Saving t-SNE to", tSNE_filename)    torch.sum(torch.max(target_logit,1)[1] == target_label) / 1675

            if i == args.max_iterations:
                exit(0)
            base_network.train(True)

        # if args.opt_G != 4:
        # pdb.set_trace()
        if i < args.warm:
            optimizer = lr_scheduler(warm_optimizer, i, **warm_schedule_param)
        else:
            optimizer = lr_scheduler(optimizer, i, **schedule_param)

        if i < args.warm:
            if i % len(dset_loaders["warm_source"]) == 0:
                iter_source = iter(dset_loaders["warm_source"])
            if i % len(dset_loaders["warm_target"]) == 0:
                iter_target = iter(dset_loaders["warm_target"])
        else:
            if i % len(dset_loaders["source"]) == 0:
                iter_source = iter(dset_loaders["source"])
            if i % len(dset_loaders["target"]) == 0:
                iter_target = iter(dset_loaders["target"])

        # train one iter

        xs, ys, ind_s = iter_source.next()
        xt, yt, ind_t = iter_target.next()
        # print(yt)
        xs, xt, ys = xs.cuda(), xt.cuda(), ys.cuda()


        if args.sf == 1:
            g_xs, f_g_xs = base_network(xs)
            g_xt, f_g_xt = base_network(xt)
        else:
            x = torch.cat((xs, xt), dim=0)
            g, f = base_network(x)
            g_xs, g_xt = g.chunk(2, dim=0)
            f_g_xs, f_g_xt = f.chunk(2, dim=0)


        # pdb.set_trace()
        ###############################################################
        if args.d_weight_label < 10.0:
            if args.label_smooth == 1 and args.cat_smooth == 1:
                ys_onehot = my_CrossEntropy.smooth(ys)
            else:
                ys_onehot = F.one_hot(ys, num_classes=args.class_num).float()

            yt_predict = F.softmax(f_g_xt, -1)
            cor_s_d = Concate_w(g_xs.detach(), ys_onehot.to('cuda'), weight=args.d_weight_label)
            cor_t_d = Concate_w(g_xt.detach(), yt_predict.detach(), weight=args.d_weight_label)
        elif args.d_weight_label > 10.0 and args.d_weight_label < 20.0:
            yt_predict = F.softmax(f_g_xt, -1)
            ys_predict = F.softmax(f_g_xs, -1)
            cor_s_d = Concate_w(g_xs.detach(), ys_predict.detach(), weight=args.d_weight_label - 10.0)
            cor_t_d = Concate_w(g_xt.detach(), yt_predict.detach(), weight=args.d_weight_label - 10.0)
        elif args.d_weight_label > 20.0:
            cor_s_d = Concate_w(g_xs.detach(), f_g_xs.detach(), weight=args.d_weight_label - 20.0)
            cor_t_d = Concate_w(g_xt.detach(), f_g_xt.detach(), weight=args.d_weight_label - 20.0)

        cor_s_d.requires_grad_(True)
        cor_t_d.requires_grad_(True)

        if i >= args.warm:
            for d in range(args.d_iter):
                # pdb.set_trace()
                potential_r = domain_D(cor_s_d)
                potential_f = domain_D(cor_t_d)

                if args.mass_inc == 1:
                    d_loss = cal_dloss_inc(potential_r, potential_f, args.point_mass * args.q ** i)
                else:
                    d_loss = cal_dloss(potential_r, potential_f, args.point_mass * args.q ** i)

                if d == 0:
                    if args.WG == 0:
                        gp_loss, M, source_norm, all_norm, grad_all = G_P(d_loss, [cor_s_d, cor_t_d])
                    else:
                        gp_loss, M, source_norm, all_norm, grad_all = G_P(d_loss, [cor_s_d, cor_t_d], args.point_mass * args.q ** i, type=args.WG)
                    args.writer.add_scalar('Train/M_grad', M, i)
                else:
                    gp_loss = torch.tensor(0.)
                    M = torch.tensor(0.)

                d_loss_all = d_loss + gp_loss
                # print(
                #     '\t d_iter ' + str(d) + ' d_loss: ' + str(np.array(d_loss.item()).round(6)) + '\t' +
                #     'gp_loss: ' + str(np.array(gp_loss.item()).round(6)) + '\t' + ' M_grad:' + str(np.array(M.item()).round(6)))

                optimizer_D.zero_grad()
                d_loss_all.backward()
                optimizer_D.step()

        if i >= args.warm:
            if args.opt_D == 5:
                lr_scheduler_D.step()

            if args.d_weight_label < 10.0:
                cor_s_g = Concate_w(g_xs, ys_onehot.to('cuda'), weight=args.d_weight_label)
                cor_t_g = Concate_w(g_xt, yt_predict, weight=args.d_weight_label)
            elif args.d_weight_label > 10.0 and args.d_weight_label < 20.0:
                cor_s_g = Concate_w(g_xs, ys_predict, weight=args.d_weight_label - 10.0)
                cor_t_g = Concate_w(g_xt, yt_predict, weight=args.d_weight_label - 10.0)
            elif args.d_weight_label > 20.0:
                cor_s_g = Concate_w(g_xs, f_g_xs, weight=args.d_weight_label - 20.0)
                cor_t_g = Concate_w(g_xt, f_g_xt, weight=args.d_weight_label - 20.0)

            if args.detach_s == 1:
                potential_r_g = domain_D(cor_s_g).detach()
            else:
                potential_r_g = domain_D(cor_s_g)
            potential_f_g = domain_D(cor_t_g)

            if args.mass_inc == 1:
                transfer_loss = -cal_dloss_inc(potential_r_g, potential_f_g, args.point_mass * args.q ** (i - args.warm))
            else:
                transfer_loss = -cal_dloss(potential_r_g, potential_f_g, args.point_mass * args.q ** (i - args.warm))
        else:
            transfer_loss = torch.tensor(0.)
        #########################################################################
        # transfer_loss = torch.tensor(0.)
        # d_loss = torch.tensor(0.)
        #########################################################################

        if args.detach_ent == 1:
            # pdb.set_trace()
            fc_copy = copy.deepcopy(base_network.module.fc)
            for param in fc_copy.parameters():
                param.requires_grad = False
            target_ent = Entropy(fc_copy(g_xt)).mean()
        else:
            target_ent = Entropy(f_g_xt).mean()

        target_ent_whole = Entropy_whole(f_g_xt)

        if args.clsw > 0 and args.clsw < 10:
            classifier_loss = my_CrossEntropy(f_g_xs, ys, weight=class_weight.to('cuda')[ys], norm_type=args.clsw)
        elif args.clsw > 10 and args.clsw < 20:
            classifier_loss = my_CrossEntropy(f_g_xs, ys, weight=predict_label_dis[ys], norm_type=args.clsw-10)
        elif args.clsw > 20:
            if args.clsw == 25:
                # pdb.set_trace()
                input_weight = (all_norm / torch.max(all_norm))[:ys.shape[0]]
                classifier_loss = my_CrossEntropy(f_g_xs, ys, weight=input_weight, norm_type=5)
            elif args.clsw == 24:
                input_weight = source_norm / torch.max(source_norm)
                classifier_loss = my_CrossEntropy(f_g_xs, ys, weight=input_weight, norm_type=5)
            elif args.clsw == 22:
                input_weight = source_norm / (torch.sum(source_norm) + 1e-5)
                classifier_loss = my_CrossEntropy(f_g_xs, ys, weight=input_weight, norm_type=1)
        else:
            classifier_loss = my_CrossEntropy(f_g_xs, ys, weight=None)

        if args.cot == 0:
            cot_loss = torch.tensor(0.).cuda()
        elif args.cot == 1:
            cot_loss = marginloss(f_g_xs, ys, classes=args.class_num, alpha=1, weight=None)
            # pdb.set_trace()
        else:
            cot_loss = marginloss(f_g_xs, ys, classes=args.class_num, alpha=1, weight=class_weight.cuda())

        if args.t_cls == 1:
            predict_label = torch.from_numpy(predict_label).cuda()
            pred = predict_label[ind_t]
            t_cls_loss = nn.CrossEntropyLoss()(f_g_xt, pred)
        else:
            t_cls_loss = torch.tensor(0.).cuda()

        print('Training:\t step:{}\tclassifier:{}\ttransfer_loss:{}\t'.format(i, classifier_loss.item(), transfer_loss.item()))
        # pdb.set_trace()
        # if args.clsw % 10 == 2:
        #     if np.sum(True_class[ys.cpu().numpy()]) == 0:
        #         Target = np.zeros_like(True_class[ys.cpu().numpy()])
        #     else:
        #         Target = True_class[ys.cpu().numpy()] /  np.sum(True_class[ys.cpu().numpy()])
        # else:
        #     Target = True_class[ys.cpu().numpy()]
        # norm_error = np.linalg.norm(input_weight.cpu().numpy() - Target, 1)
        # print('Norm error: {}'.format(norm_error))
        # args.writer.add_scalar('Train/norm error', norm_error, i)
        if i > args.warm:
            args.writer.add_scalar('Train/norm on label', torch.max(torch.norm(grad_all[:, -args.class_num:], -1)), i)

        total_loss = classifier_loss * args.cls_weight + transfer_loss * args.trade_off + \
                     args.entropy * target_ent + args.entropy_s * Entropy(f_g_xs).mean() + \
                     args.entropy_w * target_ent_whole + args.cot_weight * cot_loss + \
                     args.t_cls_weight * t_cls_loss

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
    
    parser.add_argument('--dset', type=str, default='OfficeHome', choices=["VisDA2017", "OfficeHome", "ImageNetCaltech", "DomainNet"])
    parser.add_argument('--test_interval', type=int, default=500, help="interval of two continuous test phase")
    parser.add_argument('--auto_ratio', type=int, default=0, help="")
    parser.add_argument('--detach_ent', type=int, default=0, help="")
    parser.add_argument('--label_smooth', type=int, default=0, help="")
    parser.add_argument('--cat_smooth', type=int, default=0, help="")
    parser.add_argument('--NoRelu', type=int, default=0, help="")
    parser.add_argument('--normalize', type=int, default=0, help="")
    parser.add_argument('--entropy_w', type=float, default=0, help="")
    parser.add_argument('--detach_s', type=int, default=0, help="")
    parser.add_argument('--skip_first', type=int, default=0, help="")


    ########
    parser.add_argument('--mass_inc', default=0, type=int)
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
    parser.add_argument('--opt_D', type=int, default=0, help='0: SGD momentum 1:RMSprop 2: SGD 3: RMSProp no schedule')
    parser.add_argument('--entropy', type=float, default=0., help='entropy weight')
    parser.add_argument('--entropy_s', type=float, default=0., help='entropy weight')
    parser.add_argument('--pre_process', type=int, default=0, help='')
    parser.add_argument('--init_fc', type=int, default=0, help='')
    parser.add_argument('--pre_train', type=int, default=0, help='')
    parser.add_argument('--cot', type=int, default=0, help='0: no, 1:uniform weight, 2: use weight')
    parser.add_argument('--cot_weight', type=float, default=0., help='')
    parser.add_argument('--t_cls', type=int, default=0, help='')
    parser.add_argument('--t_cls_weight', type=float, default=0., help='')
    parser.add_argument('--clsw', type=int, default=0, help='')
    parser.add_argument('--bottle_dim', type=int, default=256, help='')
    parser.add_argument('--tcls', type=int, default=0, help='')
    parser.add_argument('--sf', type=int, default=0, help='')
    parser.add_argument('--warm', type=int, default=0, help='')
    parser.add_argument('--WG', type=int, default=0, help='')
    parser.add_argument('--leaky', type=float, default=0., help='')

    parser.add_argument('--pm_ratio', type=float, default=1., help='point mass decrease ratio at the end of the training.')

    args = parser.parse_args()

    if args.dset =='OfficeHome':
        names = ['Ar', 'Cl', 'Pr', 'Rw']
        if args.tcls == 0:
            k = 25
        else:
            k = args.tcls

        args.class_num = 65
        # args.test_interval = 500
        if args.batch_size == 65:
            args.balance = 1
        else:
            args.balance = 0
        args.s_name = names[args.s]
        args.t_name = names[args.t]
        args.color_label = False
        # args.pre_train = 500

    if args.dset == 'VisDA2017':
        names = ['Synthetic', 'Real']
        k = 6
        args.class_num = 12
        # args.test_interval = 500
        if args.batch_size % 12 == 0:
            args.balance = 1
        else:
            args.balance = 0
        args.s = 0
        args.t = 1
        args.s_name = names[args.s]
        args.t_name = names[args.t]
        # args.color_label = True
        args.color_label = False
        # args.pre_train = 0

    if args.dset == 'ImageNetCaltech':
        names = ['I', 'C']
        k = 84
        if args.s == 1:
            args.class_num = 256
        else:
            args.class_num = 1000
        args.balance = 0
        args.s_name = names[args.s]
        args.t_name = names[args.t]
        args.color_label = False

    if args.dset =='DomainNet':
        names = ['c', 'p', 'r', 's']
        args.class_num = 126
        args.balance = 0
        args.s_name = names[args.s]
        args.t_name = names[args.t]
        args.color_label = False


    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    cudnn.deterministic = True
    cudnn.benchmark = False

    data_folder = './data/'
    args.dset_path = data_folder + args.dset

    args.name = names[args.s][0].upper() + names[args.t][0].upper()

    setting_name = '{}_{}_WO_{}_h_{}_WL_{}_ItD_{}_PM_{}_cls_{}_ent_{}_bs_{}' \
                   '_r_{}_smo_{}_cats_{}_NoR_{}_norm_{}_Pre_{}_c_{}_cw_{}_clsw_{}_auto_{}_ti{}_skip_{}_inc_{}_init_{}' \
                   '_G_{}_D_{}_sf_{}_warm_{}_WG_{}_ds_{}_ly_{}_seed_{}'.format(
                    args.dset, args.tcls, args.trade_off, args.d_hidden, args.d_weight_label,
                    args.d_iter, args.point_mass, args.cls_weight, args.entropy,
                    args.batch_size, args.pm_ratio, args.label_smooth, args.cat_smooth,
                    args.NoRelu, args.normalize, args.pre_train, args.cot, args.cot_weight,
                    args.clsw, args.auto_ratio, args.test_interval, args.skip_first,
                    args.mass_inc, args.init_fc, args.opt_G, args.opt_D, args.sf,
                    args.warm, args.WG, args.detach_s, args.leaky, args.seed)

    print(setting_name)
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

    args.q = np.exp(np.log(args.pm_ratio) / (args.max_iterations - args.warm))

    train(args)
