import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from torchvision import transforms, models

device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)

# ---------------------------------------------------
# TRANSFORM
# ---------------------------------------------------
transform = transforms.Compose([

    transforms.Resize((256, 256)),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ---------------------------------------------------
# DECODER BLOCK
# ---------------------------------------------------
class DecoderBlock(nn.Module):

    def __init__(self, in_c, skip_c, out_c):

        super().__init__()

        self.conv = nn.Sequential(

            nn.Conv2d(
                in_c + skip_c,
                out_c,
                3,
                padding=1
            ),

            nn.BatchNorm2d(out_c),

            nn.ReLU(inplace=True),

            nn.Conv2d(
                out_c,
                out_c,
                3,
                padding=1
            ),

            nn.BatchNorm2d(out_c),

            nn.ReLU(inplace=True)
        )

    def forward(self, x, skip):

        x = F.interpolate(
            x,
            scale_factor=2,
            mode="bilinear",
            align_corners=False
        )

        x = torch.cat([x, skip], dim=1)

        return self.conv(x)

# ---------------------------------------------------
# RESNET50 U-NET
# ---------------------------------------------------
class ResNet50_UNet(nn.Module):

    def __init__(self, num_keypoints):

        super().__init__()

        resnet = models.resnet50(
            weights=models.ResNet50_Weights.IMAGENET1K_V1
        )

        self.enc0 = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu
        )

        self.pool = resnet.maxpool

        self.enc1 = resnet.layer1
        self.enc2 = resnet.layer2
        self.enc3 = resnet.layer3
        self.enc4 = resnet.layer4

        self.center = nn.Conv2d(
            2048,
            1024,
            1
        )

        self.dec4 = DecoderBlock(
            1024,
            1024,
            512
        )

        self.dec3 = DecoderBlock(
            512,
            512,
            256
        )

        self.dec2 = DecoderBlock(
            256,
            256,
            128
        )

        self.dec1 = DecoderBlock(
            128,
            64,
            64
        )

        self.head = nn.Conv2d(
            64,
            num_keypoints,
            1
        )

    def forward(self, x):

        e0 = self.enc0(x)

        e1 = self.enc1(
            self.pool(e0)
        )

        e2 = self.enc2(e1)

        e3 = self.enc3(e2)

        e4 = self.enc4(e3)

        x = self.center(e4)

        x = self.dec4(x, e3)

        x = self.dec3(x, e2)

        x = self.dec2(x, e1)

        x = self.dec1(x, e0)

        x = self.head(x)

        x = F.interpolate(
            x,
            size=(64, 64)
        )

        return x

# ---------------------------------------------------
# LOAD MODELS
# ---------------------------------------------------
def load_keypoint_models():

    side = ResNet50_UNet(9)

    rear = ResNet50_UNet(4)

    side.load_state_dict(
        torch.load(
            'models/best_keypoint_resnet50_unet_side.pth',
            map_location=device
        )
    )

    rear.load_state_dict(
        torch.load(
            'models/best_keypoint_resnet50_unet_rear.pth',
            map_location=device
        )
    )

    side.eval().to(device)

    rear.eval().to(device)

    return side, rear

# ---------------------------------------------------
# GET KEYPOINTS
# ---------------------------------------------------
def get_keypoints(model, image):

    orig_w, orig_h = image.size

    img_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():

        heatmaps = model(img_tensor)[0].cpu()

    coords = []

    for hm in heatmaps:

        idx = hm.argmax()

        x = (idx % 64) * 4
        y = (idx // 64) * 4

        x = x * orig_w / 256
        y = y * orig_h / 256

        coords.append([x, y])

    return np.array(coords)