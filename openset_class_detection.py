import torch
import torch.nn as nn
from typing import Optional, Sequence
import torch.nn.functional as F
from common.modules.classifier import Classifier as ClassifierBase

class GradientReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return x

    @staticmethod
    def backward(ctx, grad_output):
        return -grad_output

class GradientReverseLayer(torch.nn.Module):
    def __init__(self):
        super(GradientReverseLayer, self).__init__()

    def forward(self, x):
        return GradientReverse.apply(x)
    
class ImageClassifier(ClassifierBase):
    def __init__(self, backbone: nn.Module, num_classes: int, bottleneck_dim: Optional[int] = 256, **kwargs):
        bottleneck = nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=(1, 1)),
            nn.Flatten(),
            nn.Linear(backbone.out_features, bottleneck_dim),
            nn.BatchNorm1d(bottleneck_dim),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(bottleneck_dim, bottleneck_dim),
            nn.BatchNorm1d(bottleneck_dim),
            nn.ReLU(),
            nn.Dropout()
        )
        super(ImageClassifier, self).__init__(backbone, num_classes, bottleneck, bottleneck_dim, **kwargs)
        self.grl = GradientReverseLayer()
       
    def forward(self, x: torch.Tensor, grad_reverse: Optional[bool] = False, need_fp=False):
        features = self.backbone(x)
        features = self.bottleneck(features)
        if grad_reverse:
            features = self.grl(features)
            outputs = self.head1(features)
            return outputs, features
        outputs = self.head1(features)
       
        if need_fp:
            outs = self.head1(torch.cat((features, nn.Dropout2d(0.5)(features))))
            out, out_fp = outs.chunk(2)
            return out, out_fp
       
        if self.training:
            return outputs, features
        else: 
            return outputs


class Unknown_class_detection(nn.Module):

    def __init__(self, num_classes):
        super(Unknown_class_detection, self).__init__()
        self.num_classes = num_classes

    def forward(self, logits_s1, logits_s2, pred_u_w_fp, mask_u_1, mask_u_2) -> torch.Tensor:

        bsz = logits_s1.size()[0]
        device = logits_s1.device
        unknown_class_ind = self.num_classes - 1

        mask_1 = (mask_u_1>=unknown_class_ind).float()
        mask_2 = (mask_u_2>=unknown_class_ind).float()
        unkonwn_nums_1 = mask_1.sum().item()
        unkonwn_nums_2 = mask_2.sum().item()
        
        label_np = [unknown_class_ind for i in range(bsz)]
        label = torch.Tensor(label_np).type(torch.int64).to(device)
        loss_u_s1 = F.cross_entropy(logits_s1, label, reduction='none').to(device)        
        loss_u_s2 = F.cross_entropy(logits_s2, label, reduction='none').to(device)       
        loss_u_w_fp = F.cross_entropy(pred_u_w_fp, label, reduction='none').to(device)      
        loss_u = (loss_u_s1 * 0.25 + loss_u_s2 * 0.25 + loss_u_w_fp * 0.5)
        
        loss=0
        if unkonwn_nums_1 != 0.0:
            loss += (loss_u * mask_1).sum() / unkonwn_nums_1
        if unkonwn_nums_2 != 0.0:
            loss += (loss_u * mask_2).sum() / unkonwn_nums_2
        if unkonwn_nums_1 == 0.0 and unkonwn_nums_2 == 0.0:
            loss = torch.tensor([0])
            
        return loss
    
class Known_class_detection(nn.Module):

    def __init__(self, num_classes: float):
        super(Known_class_detection, self).__init__()
        self.num_classes = num_classes

                   
    def forward(self, logits_s1, logits_s2, pred_u_w_fp, mask_u_1, mask_u_2, pred_u_w):
        device = logits_s1.device
        unknown_class_ind = self.num_classes - 1
        max_prob, label_m = torch.max(pred_u_w, dim=1)
        mask_1 = (mask_u_1<unknown_class_ind).float()
        mask_2 = (mask_u_2<unknown_class_ind).float()
        konwn_nums_1 = mask_1.sum().item()
        konwn_nums_2 = mask_2.sum().item()
       
        loss_u_s1 = F.cross_entropy(logits_s1, label_m, reduction='none').to(device)
        loss_u_s2 = F.cross_entropy(logits_s2, label_m, reduction='none').to(device)
        loss_u_w_fp = F.cross_entropy(pred_u_w_fp, label_m, reduction='none').to(device)
        loss_u = (loss_u_s1 * 0.25 + loss_u_s2 * 0.25 + loss_u_w_fp * 0.5)
        
        loss=0
        if konwn_nums_1 != 0.0:
            loss += (loss_u * mask_1 ).sum() / konwn_nums_1
        if konwn_nums_2 != 0.0:
            loss += (loss_u * mask_2 ).sum() / konwn_nums_2
        if konwn_nums_1 == 0.0 and konwn_nums_2 == 0.0:
            loss = torch.tensor([0])
        
        return loss
    
class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf

        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))
        batch_size = features.shape[0] // 2
        features = F.normalize(features, dim=1)
        f_t1, f_t2 = torch.split(features, [batch_size, batch_size], dim=0)
        features = torch.cat([f_t1.unsqueeze(1), f_t2.unsqueeze(1)], dim=1)


        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        # prevent computing log(0), which will produce Nan in the loss
        # log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-6)

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss