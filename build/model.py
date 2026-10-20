from model import SFARNet


def build_model(config):
    """Construct the reference network from a plain configuration dictionary."""
    return SFARNet(base_channels=int(config.get('base_channels', 16)))
