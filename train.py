import torch
import numpy as np
import utils
import torch.optim as optim
import torch.nn as nn
import test
import mnist
import mnistm
from utils import save_model
from utils import visualize
from utils import set_model_mode
import params

try:
    import wandb
    _WANDB_AVAILABLE = True
except ImportError:  # graceful degradation if wandb not installed
    _WANDB_AVAILABLE = False

# Source : 0, Target :1
source_test_loader = mnist.mnist_test_loader
target_test_loader = mnistm.mnistm_test_loader


def source_only(encoder, classifier, source_train_loader, target_train_loader, wandb_run=None):
    print("Training with only the source dataset")

    classifier_criterion = nn.CrossEntropyLoss().cuda()
    optimizer = optim.SGD(
        list(encoder.parameters()) +
        list(classifier.parameters()),
        lr=0.01, momentum=0.9)
    scaler = torch.cuda.amp.GradScaler(enabled=getattr(params, 'use_amp', False))

    for epoch in range(params.epochs):
        print(f"Epoch: {epoch}")
        set_model_mode('train', [encoder, classifier])

        start_steps = epoch * len(source_train_loader)
        total_steps = params.epochs * len(target_train_loader)

        for batch_idx, (source_data, target_data) in enumerate(zip(source_train_loader, target_train_loader)):
            source_image, source_label = source_data
            p = float(batch_idx + start_steps) / total_steps

            source_image = torch.cat((source_image, source_image, source_image), 1)  # MNIST convert to 3 channel
            source_image, source_label = source_image.cuda(non_blocking=True), source_label.cuda(non_blocking=True)

            utils.optimizer_scheduler(optimizer=optimizer, p=p)
            optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=getattr(params, 'use_amp', False)):
                source_feature = encoder(source_image)
                class_pred = classifier(source_feature)
                class_loss = classifier_criterion(class_pred, source_label)

            scaler.scale(class_loss).backward()
            scaler.step(optimizer)
            scaler.update()
            if (batch_idx + 1) % 100 == 0:
                total_processed = batch_idx * len(source_image)
                total_dataset = len(source_train_loader.dataset)
                percentage_completed = 100. * batch_idx / len(source_train_loader)
                print(f'[{total_processed}/{total_dataset} ({percentage_completed:.0f}%)]\tClassification Loss: {class_loss.item():.4f}')
            if wandb_run is not None:
                wandb_run.log({
                    'phase': 'train',
                    'mode': 'source_only',
                    'epoch': epoch,
                    'step': epoch * len(source_train_loader) + batch_idx,
                    'loss/classification': class_loss.item(),
                    'lr': optimizer.param_groups[0]['lr']
                })

        acc = test.tester(encoder, classifier, None, source_test_loader, target_test_loader, training_mode='Source_only')
        if wandb_run is not None:
            wandb_run.log({
                'epoch': epoch,
                'eval/source_accuracy': acc['Source']['accuracy'],
                'eval/target_accuracy': acc['Target']['accuracy']
            })

    save_model(encoder, classifier, None, 'Source-only')
    visualize(encoder, 'Source-only')


def dann(encoder, classifier, discriminator, source_train_loader, target_train_loader, wandb_run=None):
    print("Training with the DANN adaptation method")

    classifier_criterion = nn.CrossEntropyLoss().cuda()
    discriminator_criterion = nn.CrossEntropyLoss().cuda()

    optimizer = optim.SGD(
        list(encoder.parameters()) +
        list(classifier.parameters()) +
        list(discriminator.parameters()),
        lr=0.01,
        momentum=0.9)
    scaler = torch.cuda.amp.GradScaler(enabled=getattr(params, 'use_amp', False))

    for epoch in range(params.epochs):
        print(f"Epoch: {epoch}")
        set_model_mode('train', [encoder, classifier, discriminator])

        start_steps = epoch * len(source_train_loader)
        total_steps = params.epochs * len(target_train_loader)

        for batch_idx, (source_data, target_data) in enumerate(zip(source_train_loader, target_train_loader)):

            source_image, source_label = source_data
            target_image, target_label = target_data

            p = float(batch_idx + start_steps) / total_steps
            alpha = 2. / (1. + np.exp(-10 * p)) - 1

            source_image = torch.cat((source_image, source_image, source_image), 1)

            source_image, source_label = source_image.cuda(non_blocking=True), source_label.cuda(non_blocking=True)
            target_image, target_label = target_image.cuda(non_blocking=True), target_label.cuda(non_blocking=True)
            combined_image = torch.cat((source_image, target_image), 0)

            utils.optimizer_scheduler(optimizer=optimizer, p=p)
            optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=getattr(params, 'use_amp', False)):
                combined_feature = encoder(combined_image)
                source_feature = combined_feature[:source_image.size(0)]

                class_pred = classifier(source_feature)
                class_loss = classifier_criterion(class_pred, source_label)

                domain_pred = discriminator(combined_feature, alpha)
                domain_source_labels = torch.zeros(source_label.shape[0], device=source_label.device, dtype=torch.long)
                domain_target_labels = torch.ones(target_label.shape[0], device=target_label.device, dtype=torch.long)
                domain_combined_label = torch.cat((domain_source_labels, domain_target_labels), 0)
                domain_loss = discriminator_criterion(domain_pred, domain_combined_label)
                total_loss = class_loss + domain_loss

            scaler.scale(total_loss).backward()
            scaler.step(optimizer)
            scaler.update()

            if (batch_idx + 1) % 100 == 0:
                print('[{}/{} ({:.0f}%)]\tTotal Loss: {:.4f}\tClassification Loss: {:.4f}\tDomain Loss: {:.4f}'.format(
                    batch_idx * len(target_image), len(target_train_loader.dataset), 100. * batch_idx / len(target_train_loader), total_loss.item(), class_loss.item(), domain_loss.item()))
            if wandb_run is not None:
                wandb_run.log({
                    'phase': 'train',
                    'mode': 'dann',
                    'epoch': epoch,
                    'step': epoch * len(source_train_loader) + batch_idx,
                    'loss/total': total_loss.item(),
                    'loss/classification': class_loss.item(),
                    'loss/domain': domain_loss.item(),
                    'alpha': alpha,
                    'lr': optimizer.param_groups[0]['lr']
                })

        acc = test.tester(encoder, classifier, discriminator, source_test_loader, target_test_loader, training_mode='DANN')
        if wandb_run is not None:
            log_payload = {
                'epoch': epoch,
                'eval/source_accuracy': acc['Source']['accuracy'],
                'eval/target_accuracy': acc['Target']['accuracy'],
            }
            if 'Domain' in acc:
                log_payload['eval/domain_accuracy'] = acc['Domain']['accuracy']
            wandb_run.log(log_payload)

    save_model(encoder, classifier, discriminator, 'DANN')
    visualize(encoder, 'DANN')
