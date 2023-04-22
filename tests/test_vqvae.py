from unittest import TestCase
import torch
from src.models.vqvae import ResidualBlock, CNNEncoder, CNNDecoder, VectorQuantizer, VQVAE

DEVICE = 'cuda'


class TestResidualBlock(TestCase):
    def test_forward(self):
        residual_block = ResidualBlock(256).to(DEVICE)
        input_tensor = torch.rand(size=(1, 256, 16, 16), dtype=torch.float32, device=DEVICE)
        output_tensor = residual_block(input_tensor)

        self.assertEqual(output_tensor.shape, (1, 256, 16, 16))
        self.assertEqual(output_tensor.dtype, torch.float32)


class TestCNNEncoder(TestCase):
    def test_forward(self):
        cnn_encoder = CNNEncoder(256).to(DEVICE)
        input_tensor = torch.rand(size=(1, 3, 64, 64), dtype=torch.float32, device=DEVICE)
        output_tensor = cnn_encoder(input_tensor)

        self.assertEqual(output_tensor.shape, (1, 256, 16, 16))
        self.assertEqual(output_tensor.dtype, torch.float32)


class TestCNNDecoder(TestCase):
    def test_forward(self):
        cnn_decoder = CNNDecoder(256, 3).to(DEVICE)
        input_tensor = torch.rand(size=(1, 256, 16, 16), dtype=torch.float32, device=DEVICE)
        output_tensor = cnn_decoder(input_tensor)

        self.assertEqual(output_tensor.shape, (1, 3, 64, 64))
        self.assertEqual(output_tensor.dtype, torch.float32)
        self.assertTrue(torch.less_equal(torch.max(output_tensor), 1))
        self.assertTrue(torch.greater_equal(torch.min(output_tensor), -1))


class TestVectorQuantizer(TestCase):
    def test_forward(self):
        vector_quantizer = VectorQuantizer(
            num_embeddings=64,
            embedding_dim=256,
            commitment_cost=0.25
        ).to(DEVICE)
        input_tensor = torch.rand(size=(1, 256, 16, 16), dtype=torch.float32, device=DEVICE)
        output_tensor, loss = vector_quantizer(input_tensor)

        self.assertEqual(output_tensor.shape, input_tensor.shape)
        self.assertEqual(output_tensor.dtype, input_tensor.dtype)
        self.assertEqual(loss.shape, ())
        self.assertEqual(loss.dtype, torch.float32)


class TestVQVAE(TestCase):
    def test_forward(self):
        vq_vae = VQVAE(
            hidden_size=256,
            num_embeddings=128,
            embedding_dim=512,
            commitment_cost=0.25
        ).to(DEVICE)

        input_tensor = torch.rand(size=(1, 3, 64, 64), dtype=torch.float32, device=DEVICE)
        output_tensor, loss = vq_vae(input_tensor)

        self.assertEqual(output_tensor.shape, input_tensor.shape)
        self.assertEqual(output_tensor.dtype, input_tensor.dtype)
        self.assertEqual(loss.shape, ())
        self.assertEqual(loss.dtype, torch.float32)
