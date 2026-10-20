"""Small synthetic interface demonstration, not a detection benchmark."""
import torch
from model.modules import SGEM, FSCAM, CAWR


def main():
    torch.manual_seed(0)
    torch.set_num_threads(1)
    shallow = torch.randn(1, 16, 16, 16)
    deep = SGEM(16)(torch.nn.functional.avg_pool2d(shallow, 2))
    refined_skip = FSCAM(16)(shallow)
    output = CAWR(16)(refined_skip, deep)
    print('shallow:', tuple(shallow.shape))
    print('deep:', tuple(deep.shape))
    print('reconstruction:', tuple(output.shape))
    output.square().mean().backward()
    print('Synthetic forward/backward complete; this is not a trained detector.')


if __name__ == '__main__':
    main()
