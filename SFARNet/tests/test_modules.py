import unittest
import torch
from model.modules import SGEM, FSCAM, CAWR, haar_dwt, haar_idwt


class ModuleTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        torch.set_num_threads(1)

    def test_haar_roundtrip_and_gradient(self):
        for h, w in ((8, 10), (7, 9), (1, 1)):
            x = torch.randn(2, 3, h, w, dtype=torch.float64, requires_grad=True)
            y = haar_idwt(haar_dwt(x), (h, w))
            torch.testing.assert_close(y, x, rtol=1e-12, atol=1e-12)
            y.sum().backward()
            torch.testing.assert_close(x.grad, torch.ones_like(x))

    def test_shapes_and_backpropagation(self):
        for h, w in ((8, 8), (7, 9)):
            for model, args, shape in (
                (SGEM(4), [torch.randn(2, 4, h, w, requires_grad=True)], (2, 8, h, w)),
                (FSCAM(4), [torch.randn(2, 4, h, w, requires_grad=True)], (2, 4, h, w)),
                (CAWR(4), [torch.randn(2, 4, h, w, requires_grad=True),
                           torch.randn(2, 8, (h+1)//2, (w+1)//2, requires_grad=True)], (2, 4, h, w))):
                y = model(*args)
                self.assertEqual(tuple(y.shape), shape)
                self.assertTrue(torch.isfinite(y).all())
                y.square().mean().backward()
                for x in args:
                    self.assertIsNotNone(x.grad)
                    self.assertTrue(torch.isfinite(x.grad).all())
                for name, p in model.named_parameters():
                    self.assertIsNotNone(p.grad, name)
                    self.assertTrue(torch.isfinite(p.grad).all(), name)

    def test_state_reset_and_batch_independence(self):
        model = FSCAM(4).eval()
        x = torch.randn(2, 4, 7, 9)
        state = {k: v.clone() for k, v in model.state_dict().items()}
        with torch.inference_mode():
            y = model(x)
            model(torch.randn_like(x))
            torch.testing.assert_close(y, model(x))
            single = torch.cat([model(t[None]) for t in x])
            torch.testing.assert_close(y, single, rtol=1e-4, atol=1e-6)
        for key, value in model.state_dict().items():
            torch.testing.assert_close(value, state[key])

    def test_analytic_state_gradient(self):
        k, v = torch.randn(2, 8, 4), torch.randn(2, 8, 4)
        state = torch.randn(2, 4, 4, requires_grad=True)
        error = k @ state - v
        expected, = torch.autograd.grad(error.square().sum() / (2 * k.shape[1]), state)
        actual = k.transpose(1, 2) @ error / k.shape[1]
        torch.testing.assert_close(actual, expected)


if __name__ == '__main__':
    unittest.main()
