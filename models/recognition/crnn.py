import torch
import torch.nn as nn
import cv2
import numpy as np

class CRNN(nn.Module):
    def __init__(self, img_height, num_channels, num_classes, hidden_size=256):
        super(CRNN, self).__init__()
        
        self.cnn = nn.Sequential(
            nn.Conv2d(num_channels, 64, 3, 1, 1), nn.ReLU(True), nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, 1, 1), nn.ReLU(True), nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, 3, 1, 1), nn.BatchNorm2d(256), nn.ReLU(True),
            nn.Conv2d(256, 256, 3, 1, 1), nn.ReLU(True), nn.MaxPool2d((2, 1), (2, 1)),
            nn.Conv2d(256, 512, 3, 1, 1), nn.BatchNorm2d(512), nn.ReLU(True),
            nn.Conv2d(512, 512, 3, 1, 1), nn.ReLU(True), nn.MaxPool2d((2, 1), (2, 1)),
            nn.Conv2d(512, 512, 2, 1, 0), nn.BatchNorm2d(512), nn.ReLU(True),
        )
        
        self.rnn1 = nn.LSTM(512, hidden_size, bidirectional=True, batch_first=True)
        self.linear1 = nn.Linear(hidden_size * 2, hidden_size)
        
        self.rnn2 = nn.LSTM(hidden_size, hidden_size, bidirectional=True, batch_first=True)
        self.linear2 = nn.Linear(hidden_size * 2, num_classes)
        
        self.last_confidence = 1.0
        self.charset = None
        
    def _get_charset(self):
        if self.charset is None:
            from utils.dataset import CharsetManager
            self.charset = CharsetManager()
        return self.charset
    
    def forward(self, x):
        conv = self.cnn(x)                       # (B, C, H, W)
        conv = conv.squeeze(2)                   # (B, C, W) (Assuming H=1)
        conv = conv.permute(0, 2, 1)             # (B, W, C)
        
        rnn_out, _ = self.rnn1(conv)
        rnn_out = self.linear1(rnn_out)
        
        rnn_out, _ = self.rnn2(rnn_out)
        output = self.linear2(rnn_out)           # (B, T, num_classes)
        return output
        
    def ctc_greedy_decoder(self, preds):
        """Greedy Decoder for CTC"""
        preds = torch.softmax(preds, dim=-1)
        # Get max prob and indices
        probs, indices = torch.max(preds, dim=-1)
        
        # Calculate confidence
        self.last_confidence = float(torch.mean(probs).item())
        
        indices = indices[0].cpu().numpy()
        charset = self._get_charset()
        
        decoded = []
        prev_idx = -1
        for idx in indices:
            if idx != prev_idx and idx != charset.char_to_idx.get(charset.blank_token, 0):
                char = charset.idx_to_char.get(idx, "")
                if char:
                    decoded.append(char)
            prev_idx = idx
            
        return "".join(decoded)

    @torch.no_grad()
    def recognize(self, image: np.ndarray) -> str:
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
        h, w = image.shape
        target_h = 32
        # Resize preserving aspect ratio
        ratio = target_h / max(h, 1)
        target_w = int(w * ratio)
        target_w = max(32, target_w) # min width
        
        image = cv2.resize(image, (target_w, target_h))
        
        # Convert to tensor
        tensor = torch.from_numpy(image).float() / 255.0
        tensor = tensor.unsqueeze(0).unsqueeze(0) # (1, 1, H, W)
        
        device = next(self.parameters()).device
        tensor = tensor.to(device)
        
        preds = self.forward(tensor)
        text = self.ctc_greedy_decoder(preds)
        return text
