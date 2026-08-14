import torch
from models.detection.craft import CRAFT
from models.recognition.crnn import CRNN

def test_craft_forward():
    model = CRAFT(pretrained=False)
    dummy_input = torch.randn(2, 3, 320, 320)
    region, affinity = model(dummy_input)
    assert region.shape == (2, 1, 160, 160)
    assert affinity.shape == (2, 1, 160, 160)

def test_crnn_forward():
    model = CRNN(img_height=32, num_channels=1, num_classes=100)
    dummy_input = torch.randn(2, 1, 32, 128)
    output = model(dummy_input)
    assert output.shape[0] == 2
    assert output.shape[2] == 100
