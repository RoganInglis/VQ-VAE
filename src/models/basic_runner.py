import torch
from tqdm import tqdm

from mlflow import log_metric, log_image


class BasicRunner(object):
    def __init__(self, model, train_dataloader, test_dataloader, loss_fn, optimizer, log_freq=100):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f'Using {self.device} device')

        self.model = model.to(self.device)
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader
        self.loss_fn = loss_fn
        self.optimizer = optimizer

        self.log_freq = log_freq

    def train_epoch(self, epoch):
        self.model.train()

        progress = tqdm(total=len(self.train_dataloader))
        for batch, (X, y) in enumerate(self.train_dataloader):
            X, y = X.to(self.device), y.to(self.device)

            pred = self.model(X)
            loss = self.loss_fn(pred, y)

            if batch % self.log_freq == 0:
                log_metric('Train Loss', loss, step=int(batch + len(self.train_dataloader)*epoch))

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            progress.update()
            progress.set_postfix(loss=loss.item())

    def train(self, epochs=50):
        progress = tqdm(desc='Train Epochs', total=epochs)
        for epoch in tqdm(range(epochs)):
            # Train model for one epoch
            self.train_epoch(epoch)

            # Test model
            test_loss, correct = self.test(epoch)

            progress.update()
            progress.set_postfix(TestLoss=test_loss, Accuracy=100 * correct)

    def test(self, epoch):
        if self.test_dataloader is None:
            print('WARNING: test_dataloader is None - can\'t test model')
            return

        size = len(self.test_dataloader.dataset)
        num_batches = len(self.test_dataloader)

        self.model.eval()
        test_loss, correct = 0, 0
        with torch.no_grad():
            for X, y in tqdm(self.test_dataloader, desc='Testing'):
                X, y = X.to(self.device), y.to(self.device)

                pred = self.model(X)
                test_loss += self.loss_fn(pred, y).item()

            image = torch.cat(
                [((X[0].permute(dims=(1, 2, 0)) + 1) * 0.5), ((pred[0][0].permute(dims=(1, 2, 0)) + 1) * 0.5)],
                dim=1
            )
            log_image(image.detach().cpu().numpy(), f'images_epoch_{epoch}.png')

                #correct += (pred.argmax(1) == y).type(torch.float).sum().item()
        test_loss /= num_batches
        #correct /= size

        log_metric('Test Loss', test_loss, step=len(self.train_dataloader)*(epoch + 1))
        #log_metric('Accuracy', 100*correct)

        return test_loss, correct
