import sys
import time
import timm
import torch
import torch.nn.functional as F
import torchvision.transforms as T

sys.path.append('../../..')
import common.vision.models as models
from common.vision.transforms import ResizeImage
from common.vision.transforms.randaugment import RandAugment
from common.utils.metric import accuracy, ConfusionMatrix
from common.utils.meter import AverageMeter, ProgressMeter

from data_loader import get_rs_dataset, get_rs_class_name, get_rs_dataset_imgpath

def get_model_names():
    return sorted(
        name for name in models.__dict__
        if name.islower() and not name.startswith("__")
        and callable(models.__dict__[name])
    ) + timm.list_models()


def get_model(model_name):
    if model_name in models.__dict__:
        # load models from common.vision.models
        backbone = models.__dict__[model_name](pretrained=True)
    else:
        return Exception('error in build model process')

    return backbone


def get_dataset(root, source, target, train_source_transform, val_transform, train_target_transform=None):
    if train_target_transform is None:
        train_target_transform = train_source_transform

    train_source_dataset = get_rs_dataset(root, source=True, dataset_name=source, transform=train_source_transform, appli='train')
    train_target_dataset = get_rs_dataset(root, source=False, dataset_name=target, transform=train_target_transform, appli='train')
    val_dataset = get_rs_dataset(root, source=False, dataset_name=target, transform=val_transform, appli='train')
    test_dataset = val_dataset
    class_names = get_rs_class_name(target)
    num_classes = len(class_names)

    return train_source_dataset, train_target_dataset, val_dataset, test_dataset, num_classes, class_names

def get_target_dataset(root, target, val_transform):
    target_query_dataset = get_rs_dataset_imgpath(root, dataset_name=target, transform=val_transform, appli='test')
    target_database_dataset = get_rs_dataset_imgpath(root, dataset_name=target, transform=val_transform, appli='database')
    class_names = get_rs_class_name(target)
    num_classes = len(class_names)
    
    return target_query_dataset, target_database_dataset, num_classes

class TwoStrongTransform(object):
    def __init__(self):
        normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.original = T.Compose([
            ResizeImage(256),
            T.CenterCrop(224),
            T.ToTensor(),
            normalize
        ])
        
        self.weak = T.Compose([
            ResizeImage(256),
            T.CenterCrop(224),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            normalize
        ])
        self.strong1 = T.Compose([
            ResizeImage(256),
            T.CenterCrop(224),
            T.RandomHorizontalFlip(),
            T.RandomApply([
                T.ColorJitter(0.4, 0.4, 0.4, 0.0)
            ], p=1.0),
            RandAugment(3, 5),
            T.ToTensor(),
            normalize,
        ])

        self.strong2 = T.Compose([
            ResizeImage(256),
            T.CenterCrop(224),
            T.RandomHorizontalFlip(),
            T.RandomApply([
                T.ColorJitter(0.4, 0.4, 0.4, 0.0)
            ], p=1.0),
            RandAugment(3, 5),
            T.ToTensor(),
            normalize,
        ])
        
    def __call__(self, x):
        original = self.original(x)
        weak = self.weak(x)
        strong1 = self.strong1(x)
        strong2 = self.strong2(x)
        return original, weak, strong1, strong2

def get_val_transform(resizing='default'):

    if resizing == 'default':
        transform = T.Compose([
            ResizeImage((256, 256)),
            T.CenterCrop((224, 224)),
        ])
    elif resizing == 'res.':
        transform = T.Resize((224, 224))
    elif resizing == 'res.|crop':
        transform = T.Compose([
            T.Resize(256),
            T.CenterCrop(224),
        ])
    else:
        raise NotImplementedError(resizing)
    return T.Compose([
        transform,
        T.ToTensor()
    ])


def validate(val_loader, model, args, device) -> float:
    batch_time = AverageMeter('Time', ':6.3f')
    confmat = ConfusionMatrix(len(args.class_names))
    progress = ProgressMeter(
        len(val_loader),
        [batch_time],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()
    if args.per_class_eval:
        confmat = ConfusionMatrix(len(args.class_names))
    else:
        confmat = None
    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            images = images.to(device)
            target = target.to(device)

            # compute output
            output = model(images)

            # measure accuracy and record loss
            confmat.update(target, output.argmax(1))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i)

        h,acc_global, accs, iu = confmat.compute()
        all_acc = torch.mean(accs).item() * 100
        known = torch.mean(accs[:-1]).item() * 100
        unknown = accs[-1].item() * 100
        h_score = 2 * known * unknown / (known + unknown)
        print(h)
        if confmat:
            print(confmat.format(args.class_names))
        print(' * All {all:.3f} Known {known:.3f} Unknown {unknown:.3f} H-score {h_score:.3f}'
                .format(all=all_acc, known=known, unknown=unknown, h_score=h_score))

    return h_score
