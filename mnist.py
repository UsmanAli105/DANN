import torchvision.datasets as datasets
from torch.utils.data import SubsetRandomSampler, DataLoader
from torchvision import transforms
import torch
import params

transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.1307,), (0.3081,))
                                ])

mnist_train_dataset = datasets.MNIST(root='../data/MNIST', train=True, download=True,
                                     transform=transform)
mnist_valid_dataset = datasets.MNIST(root='../data/MNIST', train=True, download=True,
                                     transform=transform)
mnist_test_dataset = datasets.MNIST(root='../data/MNIST', train=False, transform=transform)

indices = list(range(len(mnist_train_dataset)))
validation_size = 5000
train_idx, valid_idx = indices[validation_size:], indices[:validation_size]
train_sampler = SubsetRandomSampler(train_idx)
valid_sampler = SubsetRandomSampler(valid_idx)

loader_kwargs = dict(batch_size=params.batch_size,
           num_workers=params.num_workers,
           pin_memory=getattr(params, 'pin_memory', False),
           persistent_workers=getattr(params, 'persistent_workers', False))

if hasattr(params, 'prefetch_factor') and params.num_workers > 0:
  loader_kwargs['prefetch_factor'] = params.prefetch_factor

mnist_train_loader = DataLoader(
  mnist_train_dataset,
  sampler=train_sampler,
  **loader_kwargs
)

mnist_valid_loader = DataLoader(
  mnist_valid_dataset,
  sampler=train_sampler,
  **loader_kwargs
)

mnist_test_loader = DataLoader(
  mnist_test_dataset,
  **loader_kwargs
)


def one_hot_embedding(labels, num_classes=10):
    """Embedding labels to one-hot form.

    Args:
      labels: (LongTensor) class labels, sized [N,].
      num_classes: (int) number of classes.

    Returns:
      (tensor) encoded labels, sized [N, #classes].
    """
    y = torch.eye(num_classes)
    return y[labels]
