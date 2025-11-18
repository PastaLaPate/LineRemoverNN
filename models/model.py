import torch
from torch import nn
import torch.nn.functional as F

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available() else "cpu"
)
print(f"[LineRemoverNN] Using {device} device")

class AttentionGate(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super().__init__()
        # F_g: Gating signal features (from the decoder path)
        # F_l: Skip connection features (from the encoder path)
        # F_int: Intermediate features

        # The skip connection features are processed
        self.W_l = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        
        # The gating signal features are processed
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        
        # The final attention map convolution
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid() # Attention weights are scaled between 0 and 1
        )
        
        self.relu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, g, x):
        # g is the gating signal (e.g., d3)
        # x is the skip connection (e.g., x3)
        
        g_up = F.interpolate(self.W_g(g), size=x.size()[2:], mode='bilinear', align_corners=True) # Resize g to match x's size
        x_l = self.W_l(x)
        
        # Concatenate and apply ReLU
        psi = self.relu(x_l + g_up)
        
        # Calculate the attention map
        psi = self.psi(psi)
        
        # Apply the attention map to the skip connection features
        return x * psi

def conv_block(in_c, out_c):
    return nn.Sequential(
        nn.Conv2d(in_c, out_c, 3, padding=1, padding_mode="reflect"),
        nn.BatchNorm2d(out_c), # BatchNorm is often better for general denoising than InstanceNorm
        nn.LeakyReLU(0.2, inplace=True), # Increased slope slightly
        
        # Dilation helps see "longer" structures like lines
        nn.Conv2d(out_c, out_c, 3, padding=2, dilation=2, padding_mode="reflect"),
        nn.BatchNorm2d(out_c),
        nn.LeakyReLU(0.2, inplace=True),
    )

def upsample_block(in_c, out_c):
    return nn.Sequential(
        nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
        nn.Conv2d(in_c, out_c, 1) 
    )
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()

        # --- Encoder ---
        self.c1 = conv_block(1, 32) # x1
        self.c2 = conv_block(32, 64) # x2
        self.c3 = conv_block(64, 128) # x3

        # --- Bottleneck ---
        self.c4 = conv_block(128, 256) # b

        # --- Decoder ---
        
        # D3 Stage (Input: b=256, x3=128)
        self.u3 = upsample_block(256, 128) # Output: 128
        self.att3 = AttentionGate(F_g=128, F_l=128, F_int=64) # F_int is usually half of F_l/F_g
        self.dc3 = conv_block(128 + 128, 128) # Combined features: 256 -> 128

        # D2 Stage (Input: d3=128, x2=64)
        self.u2 = upsample_block(128, 64) # Output: 64
        self.att2 = AttentionGate(F_g=64, F_l=64, F_int=32)
        self.dc2 = conv_block(64 + 64, 64)   # Combined features: 128 -> 64

        # D1 Stage (Input: d2=64, x1=32)
        self.u1 = upsample_block(64, 32) # Output: 32
        self.att1 = AttentionGate(F_g=32, F_l=32, F_int=16)
        self.dc1 = conv_block(32 + 32, 32)    # Combined features: 64 -> 32

        # --- Final Mapping ---
        self.final = nn.Conv2d(32, 1, kernel_size=1)

    def forward(self, x):
        # Encoder
        x1 = self.c1(x)
        x2 = self.c2(F.max_pool2d(x1, 2))
        x3 = self.c3(F.max_pool2d(x2, 2))

        # Bottleneck
        b = self.c4(F.max_pool2d(x3, 2))

        # Decoder with Attention Gates
        
        # D3 Stage
        d3 = self.u3(b)               # Upsample
        x3_att = self.att3(d3, x3)    # Apply attention to skip connection x3
        d3 = torch.cat([d3, x3_att], dim=1)
        d3 = self.dc3(d3)

        # D2 Stage
        d2 = self.u2(d3)
        x2_att = self.att2(d2, x2)
        d2 = torch.cat([d2, x2_att], dim=1)
        d2 = self.dc2(d2)

        # D1 Stage
        d1 = self.u1(d2)
        x1_att = self.att1(d1, x1)
        d1 = torch.cat([d1, x1_att], dim=1)
        d1 = self.dc1(d1)

        # Direct prediction of the clean image
        out = self.final(d1)
        
        # Force output to be [0, 1] range (Assuming 0=Black, 1=White)
        return torch.sigmoid(out)