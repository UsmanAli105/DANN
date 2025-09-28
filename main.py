import torch
import train
import mnist
import mnistm
import model
import params

try:
    import wandb
    _WANDB = True
except ImportError:
    _WANDB = False


def main():
    source_train_loader = mnist.mnist_train_loader
    target_train_loader = mnistm.mnistm_train_loader

    if not torch.cuda.is_available():
        print("No GPUs available. (wandb integration expects CUDA for this project)")
        return

    # Initialize models
    encoder = model.Extractor().cuda()
    classifier = model.Classifier().cuda()
    discriminator = model.Discriminator().cuda()

    # Source-only run
    source_run = None
    if _WANDB:
        source_run = wandb.init(project="DANN", name="source-only", reinit=True, config={
            'mode': 'source_only',
            'batch_size': params.batch_size,
            'epochs': params.epochs,
            'num_workers': params.num_workers
        })
    train.source_only(encoder, classifier, source_train_loader, target_train_loader, wandb_run=source_run)
    if source_run is not None:
        source_run.finish()

    # Reinitialize models for fair comparison (optional but typical in experiments)
    encoder = model.Extractor().cuda()
    classifier = model.Classifier().cuda()
    discriminator = model.Discriminator().cuda()

    dann_run = None
    if _WANDB:
        dann_run = wandb.init(project="DANN", name="dann", reinit=True, config={
            'mode': 'dann',
            'batch_size': params.batch_size,
            'epochs': params.epochs,
            'num_workers': params.num_workers
        })
    train.dann(encoder, classifier, discriminator, source_train_loader, target_train_loader, wandb_run=dann_run)
    if dann_run is not None:
        dann_run.finish()


if __name__ == "__main__":
    main()
