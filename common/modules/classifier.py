"""
@author: Junguang Jiang
@contact: JiangJunguang1123@outlook.com
"""
from typing import Tuple, Optional, List, Dict
import torch.nn as nn
import torch
import torch.nn.functional as F

__all__ = ['Classifier']
    
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
    
class Classifier(nn.Module):
    """A generic Classifier class for domain adaptation.

    Args:
        backbone (torch.nn.Module): Any backbone to extract 2-d features from data
        num_classes (int): Number of classes
        bottleneck (torch.nn.Module, optional): Any bottleneck layer. Use no bottleneck by default
        bottleneck_dim (int, optional): Feature dimension of the bottleneck layer. Default: -1
        head (torch.nn.Module, optional): Any classifier head. Use :class:`torch.nn.Linear` by default
        finetune (bool): Whether finetune the classifier or train from scratch. Default: True

    .. note::
        Different classifiers are used in different domain adaptation algorithms to achieve better accuracy
        respectively, and we provide a suggested `Classifier` for different algorithms.
        Remember they are not the core of algorithms. You can implement your own `Classifier` and combine it with
        the domain adaptation algorithm in this algorithm library.

    .. note::
        The learning rate of this classifier is set 10 times to that of the feature extractor for better accuracy
        by default. If you have other optimization strategies, please over-ride :meth:`~Classifier.get_parameters`.

    Inputs:
        - x (tensor): input data fed to `backbone`

    Outputs:
        - predictions: classifier's predictions
        - features: features after `bottleneck` layer and before `head` layer

    Shape:
        - Inputs: (minibatch, *) where * means, any number of additional dimensions
        - predictions: (minibatch, `num_classes`)
        - features: (minibatch, `features_dim`)

    """

    def __init__(self, backbone: nn.Module, num_classes: int, bottleneck: Optional[nn.Module] = None,
                 bottleneck_dim: Optional[int] = -1, head: Optional[nn.Module] = None, finetune=True, pool_layer=None):
        super(Classifier, self).__init__()
        self.backbone = backbone
        self.num_classes = num_classes
        if pool_layer is None:
            self.pool_layer = nn.Sequential(
                nn.AdaptiveAvgPool2d(output_size=(1, 1)),
                nn.Flatten()
            )
        else:
            self.pool_layer = pool_layer
        if bottleneck is None:
            self.bottleneck = nn.Identity()
            self._features_dim = backbone.out_features
        else:
            self.bottleneck = bottleneck
            assert bottleneck_dim > 0
            self._features_dim = bottleneck_dim

        if head is None:
            self.head1 = nn.Linear(self._features_dim, num_classes)
            self.head2 = nn.Linear(self._features_dim, num_classes)
        else:
            self.head1 = head
            self.head2 = head
        self.finetune = finetune
        self.grl = GradientReverseLayer()

    @property
    def features_dim(self) -> int:
        """The dimension of features before the final `head` layer"""
        return self._features_dim

    def forward(self, x: torch.Tensor,grad_reverse=False,need_fp=False ) -> Tuple[torch.Tensor, torch.Tensor,bool,bool]:
        features=self.backbone(x)
        if grad_reverse:
            features = self.grl(features)
        f = self.pool_layer(features)
        f = self.bottleneck(f)
        predictions = self.head(f)
        if need_fp:
            outs = self.head(torch.cat((f, nn.Dropout2d(0.5)(f))))
            out, out_fp = outs.chunk(2)
            return out, out_fp
        if self.training:
            return predictions, f
        else:
            return predictions

    def get_parameters(self, base_lr=1.0) -> List[Dict]:
        """A parameter list which decides optimization hyper-parameters,
            such as the relative learning rate of each layer
        """
        params = [
            {"params": self.backbone.parameters(), "lr": 0.1 * base_lr if self.finetune else 1.0 * base_lr},
            {"params": self.bottleneck.parameters(), "lr": 1.0 * base_lr},
            {"params": self.head1.parameters(), "lr": 1.0 * base_lr},
            {"params": self.head2.parameters(), "lr": 1.0 * base_lr},
        ]

        return params


class ImageClassifier(Classifier):
    pass
