import random
import time
import warnings
import sys
import argparse
import shutil
import os
import numpy as np
import os.path as osp

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.optim import SGD
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
import torch.nn.functional as F

sys.path.append('../..')
from common.utils.analysis import collect_feature, tsne
from common.utils.data import ForeverDataIterator
from common.utils.meter import AverageMeter, ProgressMeter
from common.utils.logger import CompleteLogger

sys.path.append('.')
from openset_class_detection import ImageClassifier, SupConLoss, Known_class_detection, Unknown_class_detection
import utils
import loss as newloss
import network
device = torch.device("cuda")

def main(args: argparse.Namespace):
    logger = CompleteLogger(args.log, args.phase)
    print(args)

    if args.seed is not None:
        random.seed(args.seed)
        os.environ['PYTHONHASHSEED'] = str(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    cudnn.benchmark = True

    train_transform = utils.TwoStrongTransform()
    val_transform = utils.get_val_transform()
    print("train transform: ", train_transform)
    print("val_transform: ", val_transform)

    train_source_dataset, train_target_dataset, val_dataset, test_dataset, num_classes, args.class_names = \
        utils.get_dataset(args.root, args.source, args.target, train_transform, val_transform)
    train_source_loader = DataLoader(train_source_dataset, batch_size=args.batch_size,
                                     shuffle=True, num_workers=args.workers, drop_last=True)
    train_target_loader = DataLoader(train_target_dataset, batch_size=args.batch_size,
                                     shuffle=True, num_workers=args.workers, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)

    len_source_loader = len(train_source_loader)
    len_target_loader = len(train_target_loader)
    min_n_batch = min(len_source_loader, len_target_loader)
    max_n_batch = max(len_source_loader, len_target_loader)
    print('min_n_batch: ', min_n_batch, ' max_n_batch：', max_n_batch)
    if min_n_batch != 0:
        args.iters_per_epoch = max_n_batch

    train_source_iter = ForeverDataIterator(train_source_loader)
    train_target_iter = ForeverDataIterator(train_target_loader)

    # create model
    print("=> using pre-trained model '{}'".format(args.arch))
    backbone = utils.get_model(args.arch)
    pool_layer = nn.Identity() if args.no_pool else None
    classifier = ImageClassifier(backbone, num_classes, args.bottleneck_dim).to(device)
    
    # define optimizer and lr scheduler
    optimizer = SGD(classifier.get_parameters(), args.lr, momentum=args.momentum, weight_decay=args.wd, nesterov=True)
    lr_scheduler = LambdaLR(optimizer, lambda x:  args.lr * (1. + args.lr_gamma * float(x)) ** (-args.lr_decay))

    # define loss function
    ctl_fn = SupConLoss(temperature=0.1).cuda()
    unknown_csl_fn = Unknown_class_detection(num_classes).cuda()
    known_csl_fn = Known_class_detection(num_classes).cuda()
    
    # resume from the best checkpoint
    if args.phase == 'analysis':
        print('load checkpoint: best model')
        checkpoint = torch.load(logger.get_checkpoint_path('best'), map_location='cpu')
        classifier.load_state_dict(checkpoint)
        # extract features from both domains
        feature_extractor = nn.Sequential(classifier.backbone, classifier.bottleneck).to(device)
        source_feature, labels_s = collect_feature(train_source_loader, feature_extractor, device)
        target_feature, labels_t = collect_feature(train_target_loader, feature_extractor, device)
        # plot t-SNE
        tSNE_filename = osp.join(logger.visualize_directory, 'a_u.png')
        tsne.visualize(source_feature, target_feature,labels_s, labels_t, tSNE_filename)
        print("Saving t-SNE to", tSNE_filename)
        return

    # image classification test
    if args.phase == 'test':
        print('load checkpoint: best model')
        checkpoint = torch.load(logger.get_checkpoint_path('best'), map_location='cpu')
        classifier.load_state_dict(checkpoint)
        acc = utils.validate(val_loader, classifier, args, device)
        # print("Classification Accuracy = {:0.4f}".format(acc))
        return

    # start training
    best_acc = 0.
    for epoch in range(args.epochs):
        # train for one epoch
        train(train_source_iter, train_target_iter, classifier, ctl_fn, unknown_csl_fn, known_csl_fn,
              optimizer, lr_scheduler, epoch, args,num_classes)

        # evaluate on validation set
        acc = utils.validate(val_loader, classifier, args, device)

        # remember best acc@1 and save checkpoint
        torch.save(classifier.state_dict(), logger.get_checkpoint_path('latest'))
        if acc > best_acc:
            shutil.copy(logger.get_checkpoint_path('latest'), logger.get_checkpoint_path('best'))
        best_acc = max(acc, best_acc)

    print("best_accu = {:.3f}".format(best_acc))

    # evaluate on test set
    classifier.load_state_dict(torch.load(logger.get_checkpoint_path('best')))
    acc = utils.validate(val_loader, classifier, args, device)
    print("test_accu = {:.3f}".format(acc))

    logger.close()

def bce_loss(output, target):
    output_neg = 1 - output
    target_neg = 1 - target
    result = torch.mean(target * torch.log(output + 1e-6))
    result += torch.mean(target_neg * torch.log(output_neg + 1e-6))
    return -torch.mean(result)

def NOUN(feature, ad_net):
    ad_out = ad_net(feature)
    batch_size = ad_out.size(0) // 2
    dc_target = torch.from_numpy(np.array([[1]] * batch_size + [[0]] * batch_size)).float().cuda()
    return nn.BCELoss()(ad_out, dc_target)

def get_prototype_weight(center_feat, feat,num_classes):
    N, C = feat.shape#32*256
    class_numbers = num_classes#8
    feat_proto_distance = -torch.ones((N, class_numbers)).to(feat.device)#32,7,256
    for i in range(class_numbers):
       # cc = center_feat[i].expand(N, -1)#256->32*256
        feat_proto_distance[:, i] = torch.norm(center_feat[i].expand(N, -1) - feat, 2, dim=1,)
    feat_nearest_proto_distance, feat_nearest_proto = feat_proto_distance.min(dim=1, keepdim=True)
    feat_proto_distance = feat_proto_distance - feat_nearest_proto_distance
    weight = F.softmax(-feat_proto_distance, dim=1)#32*7
    
    softmax_out = nn.Softmax(dim=1)(-feat_proto_distance)   
    entropy2 = newloss.Entropy(softmax_out) # unknown classes
    prt_loss =torch.mean(entropy2)
    
    return weight,prt_loss

 ################################# 
def train(train_source_iter: ForeverDataIterator, train_target_iter: ForeverDataIterator, model: ImageClassifier,
          ctl_fn: nn.Module, unknown_csl_fn: nn.Module, known_csl_fn: nn.Module, optimizer: SGD,
          lr_scheduler: LambdaLR, epoch: int, args: argparse.Namespace,num_classes):
    batch_time = AverageMeter('Time', ':4.2f')
    data_time = AverageMeter('Data', ':3.1f')
    losses = AverageMeter('Loss', ':3.2f')
    csl_losses = AverageMeter('Cst Loss', ':5.4f')
    ctl_losses = AverageMeter('Tdl Loss', ':5.4f')
    sce_losses = AverageMeter('Sdl Loss', ':5.4f')
   
    ad_net2 = network.AdversarialNetwork(512, 1024, max_iter=10000).cuda()
    
    progress = ProgressMeter(
        args.iters_per_epoch,
        [losses, sce_losses, ctl_losses, csl_losses],
        prefix="Epoch: [{}]".format(epoch))
    

    # switch to train mode
    model.train()

    end = time.time()
    
    for i in range(args.iters_per_epoch):
        
        (x_s_ori, x_s, _, _), labels_s = next(train_source_iter)
        (x_t_ori, x_t_w, x_t_s1,x_t_s2), labels_t = next(train_target_iter)
        bsz = labels_s.shape[0]

        x_t = torch.cat([x_t_w, x_t_s1], dim=0)
        x_s = x_s.to(device)
        x_t = x_t.to(device)
        labels_s = labels_s.to(device)
        labels_t = labels_t.to(device)
        x_t_s2 = x_t_s2.to(device)
        x_t_w = x_t_w.to(device)
        x_t_s1 = x_t_s1.to(device)
        x_t_ori = x_t_ori.to(device)
        x_s_ori = x_s_ori.to(device)

        # measure data loading time
        data_time.update(time.time() - end)

        # compute output
        y_s, f_s = model(x_s)
        y_t, f_t = model(x_t)
        y_t_w, y_t_s1 = torch.split(y_t, [bsz, bsz], dim=0)
        f_t_w, f_t_s1 = torch.split(f_t, [bsz, bsz], dim=0)
        
        # compute prototype
        softmax_s = nn.Softmax(dim=1)(y_s)
        softmax_t1 = nn.Softmax(dim=1)(y_t_w)
        softmax_t2 = nn.Softmax(dim=1)(y_t_s1)
        outputs = torch.cat((y_s, y_t_w), dim=0)
        softmax_out = nn.Softmax(dim=1)(outputs)   
        entropy = newloss.Entropy(softmax_out)     
        gt_list = labels_s.tolist()
        gt_list = np.unique(gt_list)

        if i == 0:
            center_feat = torch.randn(softmax_s.shape[1], f_s.shape[1]).cuda().detach()
            target_center_feat1 = torch.randn(softmax_t1.shape[1], f_t_w.shape[1]).cuda().detach()
            target_center_feat2 = torch.randn(softmax_t2.shape[1], f_t_w.shape[1]).cuda().detach()
        else:
            for c in range(len(gt_list)):
                c_idx = (labels_s==gt_list[c]).nonzero().squeeze()
                c_feat = torch.index_select(f_s, 0, c_idx)
                c_ctr = torch.mean(c_feat, dim=0)
                center_feat[gt_list[c], :] = 0.5*center_feat[gt_list[c], :].detach() + (1-0.5)*c_ctr.squeeze().detach()
                
            out_t_t1 = F.softmax(y_t_w, dim=1)
            pseudo_mask = out_t_t1.argmax(dim=1)
            gt_list = pseudo_mask.tolist()
            gt_list = np.unique(gt_list)
            for c in range(len(gt_list)):
                c_idx = (pseudo_mask==gt_list[c]).nonzero().squeeze()
                c_feat = torch.index_select(f_t_w, 0, c_idx)
                c_ctr = torch.mean(c_feat, dim=0)
                target_center_feat1[gt_list[c], :] = 0.5*target_center_feat1[gt_list[c], :].detach() + (1-0.5)*c_ctr.squeeze().detach()
        
            out_t_t2 = F.softmax(y_t_s1, dim=1)
            pseudo_mask = out_t_t2.argmax(dim=1)
            gt_list = pseudo_mask.tolist()
            gt_list = np.unique(gt_list)
            for c in range(len(gt_list)):
                c_idx = (pseudo_mask==gt_list[c]).nonzero().squeeze()
                c_feat = torch.index_select(f_t_s1, 0, c_idx)
                c_ctr = torch.mean(c_feat, dim=0)
                target_center_feat2[gt_list[c], :] = 0.5*target_center_feat2[gt_list[c], :].detach() + (1-0.5)*c_ctr.squeeze().detach()

            
        target_center1 = target_center_feat1.detach()        
        target_center2 = target_center_feat2.detach() 
        
        center = center_feat.detach()# source prototypes
        cond_p_s = torch.mm(softmax_s, center).detach()
        cond_p_t = torch.mm(softmax_t1, center).detach()
        cond_p_de = torch.cat((cond_p_s, cond_p_t), dim=0)
        norm_factor = torch.norm(torch.cat((f_s, f_t_w), dim=0))/torch.norm(cond_p_de)
        norm_factor = norm_factor.detach()
        feat1 = torch.cat((torch.cat((f_s, f_t_w), dim=0), norm_factor*cond_p_de), dim=1)  
          
        cond_p_t = torch.mm(softmax_t2, center).detach()
        cond_p_de = torch.cat((cond_p_s, cond_p_t), dim=0)
        norm_factor = torch.norm(torch.cat((f_s, f_t_s1), dim=0))/torch.norm(cond_p_de)
        norm_factor = norm_factor.detach()
        feat2 = torch.cat((torch.cat((f_s, f_t_s1), dim=0), norm_factor*cond_p_de), dim=1) 
        
        weight_1,_=get_prototype_weight(target_center1, f_t_w,num_classes)
        weight_2,_=get_prototype_weight(target_center2, f_t_w,num_classes)
        pred_weight_1 = y_t_w * weight_1
        mask_u_w_1 = pred_weight_1.argmax(dim=1)
        pred_weight_2 = y_t_s1 * weight_2
        mask_u_w_2 = pred_weight_2.argmax(dim=1)
        
        
        if epoch > args.pretrain_epoch:
            # Unknown and Known Class Separation
            y_t_t, _ = model(x_t_w.to(device), grad_reverse=True)
            out_t_t = F.softmax(y_t_t, dim=1)
            prob1_t = torch.sum(out_t_t[:, :num_classes - 1], 1).view(-1, 1)
            prob2_t = out_t_t[:, num_classes - 1].contiguous().view(-1, 1)
            esl_loss = bce_loss(prob1_t, prob2_t)
            
            # Multiveiw Consistency Learning
            num_lb, num_ulb = x_s.shape[0], x_t_w.shape[0]
            preds, preds_fp = model(torch.cat((x_s, x_t_w)), need_fp=True)
            
            pred_s, pred_u_w = preds.split([num_lb, num_ulb])
            pred_u_w_fp = preds_fp[num_lb:]
            pred_u_w_fp = F.softmax(pred_u_w_fp, dim=1)
            pred_u_s,_ = model(torch.cat((x_t_s1, x_t_s2)))
            
            pred_u_s1,pred_u_s2=pred_u_s.chunk(2)
            pred_u_s1 = F.softmax(pred_u_s1, dim=1)
            pred_u_s2 = F.softmax(pred_u_s2, dim=1)
            
            pred_u_w = pred_u_w.detach()
            pred_u_w = F.softmax(pred_u_w, dim=1)
            
            #####################
            unknown_csl = unknown_csl_fn(logits_s1=pred_u_s1, logits_s2=pred_u_s2, pred_u_w_fp=pred_u_w_fp,
                   mask_u_1=mask_u_w_1, mask_u_2=mask_u_w_2).cuda()
   
            known_csl = known_csl_fn(logits_s1=pred_u_s1, logits_s2=pred_u_s2, pred_u_w_fp=pred_u_w_fp,
                   mask_u_1=mask_u_w_1, mask_u_2=mask_u_w_2, pred_u_w=pred_u_w).cuda()
            
            
            _,prt_loss= get_prototype_weight(center_feat, f_t_w,num_classes)
           
           
        else:
            unknown_csl = torch.tensor([0]).cuda()
            known_csl = torch.tensor([0]).cuda()
            prt_loss = torch.tensor([0]).cuda()
            esl_loss = torch.tensor([0]).cuda()

        # source domain discriminative loss
        sce_loss = F.cross_entropy(y_s, labels_s)
        # contrastive learning loss
        ctl_loss = ctl_fn(f_t)
        # consistency self-training loss
        csl_loss = unknown_csl + known_csl
        
        loss = sce_loss + 0.5 * (ctl_loss + csl_loss) + esl_loss + prt_loss
        
        losses.update(loss.item(), labels_s.size(0))
        csl_losses.update(csl_loss.item(), labels_s.size(0))
        ctl_losses.update(ctl_loss.item(), labels_s.size(0))
        sce_losses.update(sce_loss.item(), labels_s.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        lr_scheduler.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PUMCL for Openset Domain Adaptation')
    # dataset parameters
    parser.add_argument('-root', metavar='DIR',default='dataset',
                        help='root path of dataset')
    parser.add_argument('-s', '--source',  default='UCMD',help='source domain')
    parser.add_argument('-t', '--target',  default='NWPU',help='target domain')
    parser.add_argument('--train-resizing', type=str, default='default')
    parser.add_argument('--val-resizing', type=str, default='default')
    parser.add_argument('--resize-size', type=int, default=224,
                        help='the image size after resizing')
    parser.add_argument('--no-hflip', action='store_true',
                        help='no random horizontal flipping during training')
    parser.add_argument('--norm-mean', type=float, nargs='+',
                        default=(0.485, 0.456, 0.406), help='normalization mean')
    parser.add_argument('--norm-std', type=float, nargs='+',
                        default=(0.229, 0.224, 0.225), help='normalization std')
    # model parameters
    parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet50',
                        choices=utils.get_model_names(),
                        help='backbone architecture: ' +
                             ' | '.join(utils.get_model_names()) +
                             ' (default: resnet50)')
    parser.add_argument('--no-pool', action='store_true',
                        help='no pool layer after the feature extractor.')
    parser.add_argument('--bottleneck-dim', default=256, type=int,
                        help='Dimension of bottleneck')
    parser.add_argument('--scratch', action='store_true', help='whether train from scratch.')
    parser.add_argument('--trade-off', default=1., type=float,
                        help='the trade-off hyper-parameter for transfer loss')
    # training parameters
    parser.add_argument('-b', '--batch-size', default=32, type=int,
                        metavar='N',
                        help='mini-batch size (default: 64)')
    parser.add_argument('--lr', '--learning-rate', default=0.003, type=float,
                        metavar='LR', help='initial learning rate', dest='lr')
    parser.add_argument('--lr-gamma', default=0.0003, type=float, help='parameter for lr scheduler')
    parser.add_argument('--lr-decay', default=0.75, type=float, help='parameter for lr scheduler')
    parser.add_argument('--pretrain-epoch', default=5, type=int, help='pretrain epoch for discriminative feature learning')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--wd', '--weight-decay', default=0.0005, type=float,
                        metavar='W', help='weight decay (default: 5e-4)')
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 2)')
    parser.add_argument('--epochs', default=70, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('-i', '--iters-per-epoch', default=500, type=int,
                        help='Number of iterations per epoch')
    parser.add_argument('-p', '--print-freq', default=70, type=int,
                        metavar='N', help='print frequency (default: 100)')
    parser.add_argument('--seed', default=None, type=int,
                        help='seed for initializing training. ')
    parser.add_argument('--per-class-eval', default=True, action='store_true',
                        help='whether output per-class accuracy during evaluation')
    parser.add_argument("--log", type=str, default='logs',
                        help="Where to save logs, checkpoints and debugging images.")
    parser.add_argument("--phase", type=str, default='train', choices=['train', 'test','analysis'],
                        help="When phase is 'test', only test the model."
                             "When phase is 'analysis', only analysis the model.")
    args = parser.parse_args()
    main(args)

