import torch
from torch import nn
import torch.functional as F


def mse_loss(true, pred):
    return torch.mean((true - pred)**2)


class ResidualBlock(nn.Module):
    def __init__(self, hidden_size=256):
        super(ResidualBlock, self).__init__()
        """
        3 × 3 blocks (implemented as ReLU, 3x3 conv, ReLU, 1x1 conv)
        """

        self.residual_block = nn.Sequential(
            nn.ReLU(True),
            nn.Conv2d(hidden_size, hidden_size, 3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(hidden_size, hidden_size, 1)
        )

    def forward(self, x):
        return x + self.residual_block(x)


class CNNEncoder(nn.Module):
    def __init__(self, hidden_size=256, output_dim=512):
        super(CNNEncoder, self).__init__()

        """
        The encoder consists
        of 2 strided convolutional layers with stride 2 and window size 4 × 4, followed by two residual
        3 × 3 blocks (implemented as ReLU, 3x3 conv, ReLU, 1x1 conv), all having 256 hidden units
        """
        self.encoder = nn.Sequential(
            nn.Conv2d(3, hidden_size, 4, 2, padding=1),
            nn.ReLU(True),
            nn.Conv2d(hidden_size, hidden_size, 4, 2, padding=1),
            ResidualBlock(hidden_size),
            ResidualBlock(hidden_size)
        )

    def forward(self, x):
        return self.encoder(x)


class CNNDecoder(nn.Module):
    def __init__(self, hidden_size=256, output_dim=3):
        super(CNNDecoder, self).__init__()
        """
        The decoder similarly has two residual 3 × 3 blocks, followed by two transposed convolutions with stride
        2 and window size 4 × 4.
        """

        self.decoder = nn.Sequential(
            ResidualBlock(hidden_size),
            ResidualBlock(hidden_size),
            nn.ReLU(True),
            nn.ConvTranspose2d(hidden_size, hidden_size, 4, 2, padding=1),
            nn.ReLU(True),
            nn.ConvTranspose2d(hidden_size, output_dim, 4, 2, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        return self.decoder(x)


class VectorQuantizer(nn.Module):
    """
    Quantize a set of input embedding vectors to values from a pre-defined codebook of embedding vectors. Also compute
    the latent loss terms for training
    """
    def __init__(self, num_embeddings=8*8*10, embedding_dim=512, commitment_cost=0.25):
        super(VectorQuantizer, self).__init__()

        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost

        self.embeddings = nn.Embedding(num_embeddings=num_embeddings, embedding_dim=embedding_dim)

    def quantize(self, z):
        # Compute distances
        d1 = torch.sum(z ** 2, dim=1, keepdim=True)
        d2 = 2 * torch.matmul(z, self.embeddings.weight.T)
        d3 = torch.sum(self.embeddings.weight ** 2, dim=1, keepdim=True).T
        distances = d1 - d2 + d3  # (BHW, N) compute (z - e)**2 for all z & e

        # Select embeddings from codebook with minimum distance for each input embedding
        k = torch.argmin(distances, dim=1)  # (BHW)
        q = self.embeddings.weight[k]  # (BHW, C)
        return q

    def latent_loss(self, z, q):
        e_loss = mse_loss(z.detach(), q)  # Vector quantization loss to train the embeddings
        q_loss = self.commitment_cost * mse_loss(z, q.detach())  # Encoder commitment loss - to encourage it to 'commit' to particular embeddings and to prevent the output from growing
        loss = e_loss + q_loss
        return loss

    def forward(self, inputs):
        """

        :param inputs: Float32 tensor with shape (-1, channels) and dtype float32
        :return:
            output: Tensor with shape (-1, channels) and dtype float32. Inputs quantized to codebook embeddings
            loss: Scalar tensor with dtype float32. The latent loss value comprising of the quantization loss and the
                  commitment loss
        """
        # inputs - (B, C, H, W)
        inputs = torch.permute(inputs, dims=(0, 2, 3, 1))  # (B, H, W, C)
        z = torch.reshape(inputs, shape=(-1, self.embedding_dim)).contiguous()  # (BHW, C)

        q = self.quantize(z)

        loss = self.latent_loss(z, q)

        q = torch.reshape(q, shape=inputs.shape)
        output = torch.permute(q, dims=(0, 3, 1, 2)).contiguous()  # (B, C, H, W)
        return output, loss


class VQVAE(nn.Module):
    def __init__(self, hidden_size=256, num_embeddings=8*8*10, embedding_dim=512, commitment_cost=0.25):
        super(VQVAE, self).__init__()
        self.encoder = CNNEncoder(hidden_size, output_dim=embedding_dim)

        self.vector_quantizer = VectorQuantizer(
            num_embeddings=num_embeddings,
            embedding_dim=embedding_dim,
            commitment_cost=commitment_cost
        )

        self.decoder = CNNDecoder(hidden_size)

    def forward(self, x):
        z = self.encoder(x)

        q, latent_loss = self.vector_quantizer(z)

        y = self.decoder(z + (q - z).detach())

        reconstruction_loss = mse_loss(y, x)

        loss = reconstruction_loss + latent_loss

        return y, loss
