import unittest
import torch
from model import SFARNet
from utils.loss import soft_iou_loss, deep_supervision_loss


class NetworkTests(unittest.TestCase):
    def test_training_step_and_inference(self):
        torch.set_num_threads(1)
        torch.manual_seed(1)
        model = SFARNet(base_channels=2)
        x = torch.rand(1, 1, 17, 19)
        target = torch.zeros_like(x)
        target[..., 8:10, 9:11] = 1
        maps = model(x, return_aux=True)
        self.assertEqual(len(maps), 6)
        for p in maps:
            self.assertEqual(p.shape, x.shape)
        loss = deep_supervision_loss(maps, target)
        loss.backward()
        for name, parameter in model.named_parameters():
            self.assertIsNotNone(parameter.grad, name)
            self.assertTrue(torch.isfinite(parameter.grad).all(), name)
        torch.optim.Adam(model.parameters(), lr=1e-4).step()
        model.eval()
        with torch.inference_mode():
            output = model(x)
        self.assertEqual(output.shape, x.shape)
        self.assertTrue(torch.isfinite(output).all())

    def test_loss_edges(self):
        zero = torch.zeros(2, 1, 4, 4)
        one = torch.ones_like(zero)
        self.assertEqual(soft_iou_loss(zero, zero).item(), 0)
        self.assertEqual(soft_iou_loss(one, one).item(), 0)
        self.assertGreater(soft_iou_loss(zero, one).item(), 0.99)


if __name__ == '__main__':
    unittest.main()
