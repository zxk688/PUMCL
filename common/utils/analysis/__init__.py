import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import tqdm


def collect_feature(data_loader: DataLoader, feature_extractor: nn.Module,
                                   device: torch.device, max_num_features=None) -> torch.Tensor:
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
    all_targets = []
    with torch.no_grad():
        for i, (images, target) in enumerate(tqdm.tqdm(data_loader)):
            image_tensor = images[1].to(device)#ours-1,ssouda-0
            feature = feature_extractor(image_tensor).cpu()
            # feature = feature_extractor(images).cpu()
            all_features.append(feature)
            if max_num_features is not None and i >= max_num_features:
                break
            all_targets.append(target)
    return torch.cat(all_features, dim=0),torch.stack(all_targets)
