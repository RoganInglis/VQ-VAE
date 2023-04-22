import math
import torch
import mlflow
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from vqvae import VQVAE
from basic_runner import BasicRunner


def main():
    # Params
    params = {
        'batch_size': 32,
        'hidden_size': 256,
        'commitment_cost': 0.25,
        'num_embeddings': 8*8*10,  # 640
        'embedding_dim': 512,
        'lr': 2e-4,
        'iters': 250000
    }

    mlflow.log_params(params)

    model = VQVAE(
        hidden_size=params['hidden_size'],
        num_embeddings=params['num_embeddings'],
        embedding_dim=params['embedding_dim'],
        commitment_cost=params['commitment_cost']
    )

    # Create datasets
    input_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    training_data = datasets.CIFAR10(
        root='data/external/',
        train=True,
        download=True,
        transform=input_transform,
    )

    test_data = datasets.CIFAR10(
        root='data/external/',
        train=False,
        download=True,
        transform=input_transform
    )

    train_dataloader = DataLoader(dataset=training_data, batch_size=params['batch_size'])
    test_dataloader = DataLoader(dataset=test_data, batch_size=params['batch_size'])

    def loss_fn(pred, true):
        return pred[1]

    optimizer = torch.optim.Adam(model.parameters(), lr=params['lr'])

    runner = BasicRunner(
        model=model,
        train_dataloader=train_dataloader,
        test_dataloader=test_dataloader,
        loss_fn=loss_fn,
        optimizer=optimizer
    )

    runner.train(epochs=math.ceil(params['iters']/len(train_dataloader)))


if __name__ == '__main__':
    main()
