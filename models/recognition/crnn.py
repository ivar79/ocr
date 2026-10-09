import torch
import torch.nn as nn
import cv2
import numpy as np

class CRNN(nn.Module):
    def __init__(self, img_height, num_channels, num_classes, hidden_size=256,
                 cnn_scale=1.0):
        super(CRNN, self).__init__()

        self.img_height = img_height
        self.num_channels = num_channels
        self.num_classes = num_classes
        self.hidden_size = hidden_size
        self.cnn_scale = cnn_scale

        def c(n):
            """Channel count scaled by cnn_scale (floor at 8)."""
            return max(8, int(round(n * cnn_scale)))

        self.cnn = nn.Sequential(
            nn.Conv2d(num_channels, c(64), 3, 1, 1), nn.ReLU(True), nn.MaxPool2d(2, 2),
            nn.Conv2d(c(64), c(128), 3, 1, 1), nn.ReLU(True), nn.MaxPool2d(2, 2),
            nn.Conv2d(c(128), c(256), 3, 1, 1), nn.BatchNorm2d(c(256)), nn.ReLU(True),
            nn.Conv2d(c(256), c(256), 3, 1, 1), nn.ReLU(True), nn.MaxPool2d((2, 1), (2, 1)),
            nn.Conv2d(c(256), c(512), 3, 1, 1), nn.BatchNorm2d(c(512)), nn.ReLU(True),
            nn.Conv2d(c(512), c(512), 3, 1, 1), nn.ReLU(True), nn.MaxPool2d((2, 1), (2, 1)),
            nn.Conv2d(c(512), c(512), 2, 1, 0), nn.BatchNorm2d(c(512)), nn.ReLU(True),
        )

        cnn_out = c(512)
        self.rnn1 = nn.LSTM(cnn_out, hidden_size, bidirectional=True, batch_first=True)
        self.linear1 = nn.Linear(hidden_size * 2, hidden_size)

        self.rnn2 = nn.LSTM(hidden_size, hidden_size, bidirectional=True, batch_first=True)
        self.linear2 = nn.Linear(hidden_size * 2, num_classes)

        self.last_confidence = 1.0
        self.charset = None

    def arch(self) -> dict:
        """Architecture parameters, stored in checkpoints for reliable loading."""
        return {
            'img_height': self.img_height,
            'num_channels': self.num_channels,
            'num_classes': self.num_classes,
            'hidden_size': self.hidden_size,
            'cnn_scale': self.cnn_scale,
        }

    def save_checkpoint(self, path):
        torch.save({'arch': self.arch(), 'model': self.state_dict()}, path)

    @classmethod
    def from_checkpoint(cls, path, map_location='cpu'):
        """Load a checkpoint saved by save_checkpoint (or a plain state_dict)."""
        ckpt = torch.load(path, map_location=map_location, weights_only=False)
        if isinstance(ckpt, dict) and 'arch' in ckpt and 'model' in ckpt:
            arch = ckpt['arch']
            model = cls(
                img_height=arch['img_height'],
                num_channels=arch['num_channels'],
                num_classes=arch['num_classes'],
                hidden_size=arch['hidden_size'],
                cnn_scale=arch['cnn_scale'],
            )
            model.load_state_dict(ckpt['model'])
            return model
        # Plain state_dict fallback (legacy weights); infer classes from the
        # final linear layer and assume the full-size architecture.
        num_classes = int(ckpt['linear2.weight'].shape[0])
        model = cls(img_height=32, num_channels=1, num_classes=num_classes)
        model.load_state_dict(ckpt)
        return model
        
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
