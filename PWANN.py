import pdb


from torchvision import transforms
from models import network
import lr_schedule
from utils import *
from models.D import D2
from models.Grad import Grad_Penalty_w
import torch.optim as optim
from models.util import Concate_w, Entropy, cal_dloss_inc #, Entropy_whole
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
import datasets.partial as datasets
from datasets.partial import default_partial as partial_dataset
import torch.backends.cudnn as cudnn


def image_train(resize_size=256, crop_size=224):
        return transforms.Compose([
            transforms.Resize((resize_size, resize_size)),
            transforms.RandomCrop(crop_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

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
            data = next(iter_test)
            inputs = data[0]
            labels = data[1]
            inputs = inputs.cuda()
            ####
            outputs_all = model(inputs)

            if type(outputs_all) == torch.Tensor:
                outputs = outputs_all
            else:
                outputs = outputs_all[1]

            if start_test:
                all_output = outputs.float().cpu()
                all_label = labels.float()
                start_test = False
            else:
                all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                all_label = torch.cat((all_label, labels.float()), 0)

    _, predict = torch.max(all_output, 1)
    prob_output = nn.Softmax(dim=1)(all_output)
    accuracy = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size()[0])

    hist_tar = prob_output.sum(dim=0)
    hist_tar = hist_tar / hist_tar.sum()

    return accuracy, hist_tar, predict

def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # prepare dataset
    train_bs, test_bs = args.batch_size, args.batch_size * 2

    dsets = {}
    dataset = datasets.__dict__[args.dset]
    p_dataset = partial_dataset(dataset, args.tcls)
    dsets["source"] = dataset(root=args.dset_path, task=args.s_name, download=True, transform=image_train())
    dsets["target"] = p_dataset(root=args.dset_path, task=args.t_name, download=True, transform=image_train())
    dsets["test"] = p_dataset(root=args.dset_path, task=args.t_name, download=True, transform=image_test())

    dset_loaders = {}
    # configure balanced sampling for source dataloader for small class number
    if args.balance == 0:
        dset_loaders["source"] = DataLoader(dsets["source"], batch_size=train_bs, shuffle=True, num_workers=args.worker,
                                            prefetch_factor=8, drop_last=True)
    else:
        source_labels = torch.tensor(list(zip(*(dsets["source"].samples)))[1])
        train_batch_sampler = BalancedBatchSampler(source_labels, batch_size=train_bs)
        dset_loaders["source"] = DataLoader(dsets["source"], batch_sampler=train_batch_sampler, num_workers=args.worker,
                                            prefetch_factor=8)

    # configure target and test dataloader
    dset_loaders["target"] = DataLoader(dsets["target"], batch_size=train_bs, shuffle=True, num_workers=args.worker,
                                        drop_last=True, prefetch_factor=8)
    dset_loaders["test"]   = DataLoader(dsets["test"], batch_size=test_bs, shuffle=False, num_workers=args.worker,
                                        prefetch_factor=8)

    # configure networks
    if "ResNet" in args.net:
        if args.dset == 'ImageNetCaltech__':
            # use the original resnet structure to use the pretrained weights
            params = {"resnet_name": args.net, "use_bottleneck": False, "new_cls": False, 'class_num': args.class_num}
        else:
            params = {"resnet_name":args.net, "new_cls":True, "use_bottleneck":True, "bottleneck_dim":args.bottle_dim,
                      'class_num': args.class_num, 'init_fc':args.init_fc, "NoRelu":args.NoRelu,
                      # "normalize":args.normalize,
                      }

        base_network = network.ResNetFc(**params)
        base_network_aux = network.ResNetFc(**params)
    else:
        raise ValueError(f'Unknown backbone {args.net}')


    base_network_aux = base_network_aux.cuda()
    parameter_list_aux = base_network_aux.get_parameters()

    base_network = base_network.cuda()
    parameter_list = base_network.get_parameters()

    domain_D = D2(in_feature=base_network.output_num() + args.class_num, hidden_size=args.d_hidden,
                  leaky=args.d_leaky).to(device)

    G_P = Grad_Penalty_w(1000, gamma=1)

    optimizer_D = optim.RMSprop(domain_D.parameters(), lr=args.lr_D)

    # The optimizer setting follows from previous literatures. No tuning.
    optimizer_config = {"type": torch.optim.Adam,
                        "optim_params": {'lr': args.lr, "weight_decay": 5e-4,},
                        "lr_type": "inv",
                        "lr_param": {"lr": args.lr, "gamma": 0.001, "power": 0.75}
                        }

    optimizer = optimizer_config["type"](parameter_list,**(optimizer_config["optim_params"]))
    optimizer_pretrain = optimizer_config["type"](parameter_list, **(optimizer_config["optim_params"]))
    optimizer_aux = optimizer_config["type"](parameter_list_aux, **(optimizer_config["optim_params"]))

    schedule_param = optimizer_config["lr_param"]
    lr_scheduler = lr_schedule.schedule_dict[optimizer_config["lr_type"]]

    if args.label_smooth == 1:
        my_CrossEntropy = CrossEntropyLabelSmooth(args.class_num, epsilon=0.1)
    else:
        my_CrossEntropy = CrossEntropyLabelSmooth(args.class_num, epsilon=0.0)

    # pretrain the model on the source dataset
    source_only(base_network, args.pre_train, optimizer_pretrain, lr_scheduler, schedule_param, dset_loaders,
                my_CrossEntropy)


    # Estimate the mass
    if args.auto_ratio >= 0:
        # prepare the evaluation network, which will be used the result to guess the mass
        if args.dset == 'ImageNetCaltech':
            # Use the standard pretrained resnet50 net on ImageNet as the evaluation network.
            from torchvision import models
            evaluate_net = models.resnet50(pretrained=True).to('cuda')
        elif args.auto_ratio > 0:
            # Train the auxiliary network for a few steps as the evaluation network.
            # OfficeHome
            source_only(base_network_aux, args.auto_ratio, optimizer_aux, lr_scheduler, schedule_param, dset_loaders,
                        my_CrossEntropy)
            evaluate_net = base_network_aux
        else:
            # Use the pretrained network as the evaluation network.
            # DomainNet
            evaluate_net = base_network

        evaluate_net.train(False)

        source_label_dis = get_label_distribution(dsets["source"], args.class_num)
        _, _, predict_label = image_classification(dset_loaders, evaluate_net)
        estimated_mass = compute_mass(predict_label, args.class_num, source_label_dis, balance=args.balance)

        print(f'Predict label weighted ratio {estimated_mass:.2f}. Set point mass to {estimated_mass:.2f}.')
        args.point_mass = estimated_mass
        evaluate_net.train(True)

    # Start training
    for i in range(args.max_iterations + 1):
        if (i % args.test_interval == 0) or (i == args.max_iterations):
            # evaluate the model and update the class weights
            base_network.train(False)
            print(f'Start testing at iteration {i}.....')
            acc, class_weight, predict_label = image_classification(dset_loaders, base_network)
            print('Finish testing.....')
            args.writer.add_scalar('Test/t_acc', acc, i // args.test_interval)

            if args.visualization == 1:
                # Feature visualization
                source_feature, source_label, source_logit = collect_feature(dset_loaders["source"], base_network, device, 10000)
                target_feature, target_label, target_logit = collect_feature(dset_loaders["test"], base_network, device, 10000)

                source_norm, target_norm, cor_s, cor_t = norm_extract(source_feature, target_feature, source_label,
                                                                      train_bs, target_logit,
                                                                      domain_D, args.d_weight_label, my_CrossEntropy)

                visualize(cor_s, cor_t, source_label, target_label, source_norm, target_norm, source_logit, target_logit,
                          color_label=args.color_label, logpath=args.Log_path, name=i)

            if i == args.max_iterations:
                exit(0)
            base_network.train(True)

        optimizer = lr_scheduler(optimizer, i, **schedule_param)

        if i % len(dset_loaders["source"]) == 0:
            iter_source = iter(dset_loaders["source"])
        if i % len(dset_loaders["target"]) == 0:
            iter_target = iter(dset_loaders["target"])

        xs, ys, ind_s = next(iter_source)
        xt, yt, ind_t = next(iter_target)
        xs, xt, ys = xs.cuda(), xt.cuda(), ys.cuda()

        # extract feature jointly or seperatly. Different when base_network contains normalization layers
        if args.sf == 1:
            g_xs, f_g_xs = base_network(xs)
            g_xt, f_g_xt = base_network(xt)
        else:
            x = torch.cat((xs, xt), dim=0)
            g, f = base_network(x)
            g_xs, g_xt = g.chunk(2, dim=0)
            f_g_xs, f_g_xt = f.chunk(2, dim=0)



        ###############################################################
        ys_onehot = F.one_hot(ys, num_classes=args.class_num).float().to(g_xs.device)
        yt_predict = F.softmax(f_g_xt, -1)
        cor_s_d = Concate_w(g_xs.detach(), ys_onehot, weight=args.d_weight_label)
        cor_t_d = Concate_w(g_xt.detach(), yt_predict.detach(), weight=args.d_weight_label)

        cor_s_d.requires_grad_(True)
        cor_t_d.requires_grad_(True)

        # Upldate PWAN network
        for d in range(args.d_iter):
            potential_r = domain_D(cor_s_d)
            potential_f = domain_D(cor_t_d)

            d_loss = cal_dloss_inc(potential_r, potential_f, args.point_mass * args.q ** i)

            if d == 0:
                gp_loss, M, source_norm, all_norm, grad_all = G_P(d_loss, [cor_s_d, cor_t_d], args.point_mass * args.q ** i)
                args.writer.add_scalar('Train/M_grad', M, i)
            else:
                gp_loss = torch.tensor(0.)
                M = torch.tensor(0.)

            d_loss_all = d_loss + gp_loss

            optimizer_D.zero_grad()
            d_loss_all.backward()
            optimizer_D.step()

        cor_s_g = Concate_w(g_xs, ys_onehot.to('cuda'), weight=args.d_weight_label)
        cor_t_g = Concate_w(g_xt, yt_predict, weight=args.d_weight_label)

        if args.detach_s == 1:
            potential_r_g = domain_D(cor_s_g).detach()
        else:
            potential_r_g = domain_D(cor_s_g)
        potential_f_g = domain_D(cor_t_g)

        transfer_loss = -cal_dloss_inc(potential_r_g, potential_f_g, args.point_mass * args.q ** (i - args.warm))
        target_ent = Entropy(f_g_xt).mean()
        classifier_loss = my_CrossEntropy(f_g_xs, ys, weight=class_weight.to('cuda')[ys])

        if args.cot == 0:
            cot_loss = torch.tensor(0.).cuda()
        else:
            cot_loss = marginloss(f_g_xs, ys, classes=args.class_num)

        print(f'Training:\t step:{i}\tclassifier:{classifier_loss.item():.2f}\ttransfer_loss:{ transfer_loss.item():.2f}\t')

        args.writer.add_scalar('Train/norm on label', torch.max(torch.norm(grad_all[:, -args.class_num:], -1)), i)

        total_loss = classifier_loss * args.cls_weight + transfer_loss * args.trade_off + \
                     args.entropy * target_ent  + args.cot_weight * cot_loss

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        cls_acc = (torch.max(f_g_xs, 1)[1] == ys).sum() / ys.shape[0] * 100.
        tgt_acc = (torch.max(f_g_xt, 1)[1] == yt.to('cuda')).sum() / ys.shape[0] * 100.

        args.writer.add_scalar('Train/cls_loss', classifier_loss, i)
        args.writer.add_scalar('Train/d_loss', transfer_loss, i)
        args.writer.add_scalar('Train/s_acc', cls_acc, i)
        args.writer.add_scalar('Train/t_acc', tgt_acc, i)



if __name__ == "__main__":
    import get_args
    parser = get_args.get()
    parser = get_args.get_1(parser)
    args = parser.parse_args()

    if args.dset == 'OfficeHome':
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
    elif args.dset == 'VisDA2017':
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
    elif args.dset == 'ImageNetCaltech':
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

    elif args.dset == 'DomainNet':
        names = ['c', 'p', 'r', 's']
        args.class_num = 126
        args.balance = 0
        args.s_name = names[args.s]
        args.t_name = names[args.t]
        args.color_label = False
    else:
        raise ValueError(f'Unknown dataset {args.dset}')

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    cudnn.deterministic = True
    cudnn.benchmark = False

    data_folder = './data/'
    args.dset_path = os.path.join(data_folder, args.dset)

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

    print('---------------')
    print(f'Settings: {setting_name}')
    print('---------------')

    args.Log_path = os.path.join('LOG', setting_name, args.name)
    if not os.path.isdir(args.Log_path):
        os.makedirs(args.Log_path)

    config_path = os.path.join(args.Log_path, 'config.txt')
    with open(config_path, 'w') as f:
        f.write('  \n'.join([f'{k}: \t{v}' for k, v in vars(args).items()]))

    tf_log = os.path.join(args.Log_path, '0', 'LOG')
    args.writer = SummaryWriter(tf_log)

    args.q = np.exp(np.log(args.pm_ratio) / (args.max_iterations - args.warm))

    train(args)
