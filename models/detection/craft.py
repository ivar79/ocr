import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import cv2
import numpy as np
import yaml
import urllib.request
from pathlib import Path
import os

class double_conv(nn.Module):
    def __init__(self, in_ch, mid_ch, out_ch):
        super(double_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch + mid_ch, mid_ch, kernel_size=1),
            nn.BatchNorm2d(mid_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class CRAFT(nn.Module):
    def __init__(self, pretrained=False, config_path="configs/config.yaml"):
        super(CRAFT, self).__init__()
        
        # Load config thresholds
        self.text_threshold = 0.7
        self.link_threshold = 0.4
        self.low_text = 0.4
        
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
                if config and 'detection' in config:
                    det_config = config['detection']
                    self.text_threshold = det_config.get('text_threshold', self.text_threshold)
                    self.link_threshold = det_config.get('link_threshold', self.link_threshold)
                    self.low_text = det_config.get('low_text', self.low_text)
        except Exception:
            pass

        # Backbone: VGG16 with BatchNorm
        weights = models.VGG16_BN_Weights.DEFAULT if pretrained else None
        vgg16_bn = models.vgg16_bn(weights=weights)
        self.slice1 = torch.nn.Sequential(*list(vgg16_bn.features.children())[:12])
        self.slice2 = torch.nn.Sequential(*list(vgg16_bn.features.children())[12:19])
        self.slice3 = torch.nn.Sequential(*list(vgg16_bn.features.children())[19:29])
        self.slice4 = torch.nn.Sequential(*list(vgg16_bn.features.children())[29:39])
        
        self.slice5 = torch.nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(512, 1024, kernel_size=3, padding=6, dilation=6),
            nn.Conv2d(1024, 1024, kernel_size=1)
        )
        
        # Upsampling with skip connections
        self.upconv1 = double_conv(1024, 512, 256)
        self.upconv2 = double_conv(512, 256, 128)
        self.upconv3 = double_conv(256, 128, 64)
        self.upconv4 = double_conv(128, 64, 32)

        self.conv_cls = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(32, 16, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, kernel_size=1), nn.ReLU(inplace=True),
            nn.Conv2d(16, 2, kernel_size=1)
        )

    def forward(self, x):
        sources = []
        x = self.slice1(x)
        sources.append(x)
        x = self.slice2(x)
        sources.append(x)
        x = self.slice3(x)
        sources.append(x)
        x = self.slice4(x)
        sources.append(x)
        x = self.slice5(x)

        # U-Net style skip connections
        y = F.interpolate(x, size=sources[3].size()[2:], mode='bilinear', align_corners=False)
        y = torch.cat([y, sources[3]], dim=1)
        y = self.upconv1(y)

        y = F.interpolate(y, size=sources[2].size()[2:], mode='bilinear', align_corners=False)
        y = torch.cat([y, sources[2]], dim=1)
        y = self.upconv2(y)

        y = F.interpolate(y, size=sources[1].size()[2:], mode='bilinear', align_corners=False)
        y = torch.cat([y, sources[1]], dim=1)
        y = self.upconv3(y)

        y = F.interpolate(y, size=sources[0].size()[2:], mode='bilinear', align_corners=False)
        y = torch.cat([y, sources[0]], dim=1)
        feature = self.upconv4(y)

        y = self.conv_cls(feature)
        
        region_score = torch.sigmoid(y[:, 0:1, :, :])
        affinity_score = torch.sigmoid(y[:, 1:2, :, :])

        return region_score, affinity_score

    def load_weights(self, path):
        device = next(self.parameters()).device
        self.load_state_dict(torch.load(path, map_location=device))

    @classmethod
    def download_weights(cls, dest_dir="models/pretrained"):
        import os
        try:
            import gdown
        except ImportError:
            print("Please install gdown (pip install gdown)")
            return
            
        Path(dest_dir).mkdir(parents=True, exist_ok=True)
        dest_path = os.path.join(dest_dir, "craft_weights.pth")
        
        if os.path.exists(dest_path):
            print(f"Weights already exist at {dest_path}")
            return dest_path
            
        url = "https://drive.google.com/file/d/1Jk4eGD7crsqCCg9C9VjCLkMN3ze8kutZ/view?usp=sharing"
        print(f"Downloading CRAFT weights to {dest_path}...")
        gdown.download(url, dest_path, quiet=False, fuzzy=True)
        return dest_path

    @torch.no_grad()
    def detect(self, image: np.ndarray):
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            
        # Normalize and tensorize
        img_tensor = torch.from_numpy(image).float() / 255.0
        # (H, W, C) -> (B, C, H, W)
        img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0)
        
        # Mean/Std normalization (VGG standards)
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        img_tensor = (img_tensor - mean) / std
        
        device = next(self.parameters()).device
        img_tensor = img_tensor.to(device)
        
        region_score, affinity_score = self.forward(img_tensor)
        
        region_score = region_score[0, 0].cpu().numpy()
        affinity_score = affinity_score[0, 0].cpu().numpy()
        
        # Simple extraction using threshold and connected components
        text_score = (region_score + affinity_score) / 2.0
        text_score[text_score < self.low_text] = 0
        
        _, text_binary = cv2.threshold(text_score, self.text_threshold, 255, cv2.THRESH_BINARY)
        text_binary = np.uint8(text_binary)
        
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(text_binary, connectivity=4)
        
        boxes = []
        for i in range(1, num_labels):
            x, y, w, h, area = stats[i]
            if area < 10:
                continue
                
            # Scale back if the network shrunk the map (CRAFT usually shrinks by 2)
            # In our U-Net architecture, the output size is the same as slice1 output which is 1/2 size
            scale_x = image.shape[1] / text_binary.shape[1]
            scale_y = image.shape[0] / text_binary.shape[0]
            
            x1, y1 = int(x * scale_x), int(y * scale_y)
            x2, y2 = int((x + w) * scale_x), int((y + h) * scale_y)
            
            boxes.append([x1, y1, x2, y2])
            
        return boxes
