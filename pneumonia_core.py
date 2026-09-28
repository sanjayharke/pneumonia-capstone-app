
"""Shared training/deployment definitions. Research classification; not clinical diagnosis."""
import io
import numpy as np
import pydicom
from pydicom.pixels import apply_modality_lut, apply_voi_lut
from PIL import Image, ImageOps
import torch
from torch import nn
from torchvision import models, transforms

IMAGE_SIZE = 224
MODEL_NAMES = ['scratch_cnn', 'resnet18_linear', 'densenet121_linear',
               'resnet18_custom', 'densenet121_custom', 'mobilenet_v3_small',
               'vgg16_linear', 'resnet50_linear', 'mobilenet_v2_linear', 'efficientnet_b0_linear',
               'resnet18_custom_frozen', 'densenet121_custom_frozen',
               'resnet18_lr3e4_custom', 'densenet121_lr3e4_custom', 'resnet18_lr1e4_custom', 'scratch_se_cnn', 'convnext_tiny']


def decode_dicom(source):
    """Use modality and VOI transforms, normalize polarity, and retain no header in pixels.

    source is a pathname or byte stream. Per-image min/max scaling is fixed in advance;
    it is not fitted on validation/test data. Native grayscale remains grayscale.
    Padding is excluded from normalization; spatial orientation is not guessed.
    """
    ds = pydicom.dcmread(source)
    if int(getattr(ds, 'NumberOfFrames', 1)) != 1:
        raise ValueError('Only single-frame chest images are supported.')
    raw = ds.pixel_array
    photo = str(getattr(ds, 'PhotometricInterpretation', ''))
    if raw.ndim == 3 and raw.shape[-1] == 3 and photo in ('RGB', 'YBR_FULL', 'YBR_FULL_422'):
        # pydicom pixel_array converts supported YBR to RGB before this luminance conversion.
        original = raw.astype(np.float32)
        gray = original @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
        valid = np.ones(gray.shape, dtype=bool)
    elif raw.ndim == 2 and photo in ('MONOCHROME1', 'MONOCHROME2'):
        original = raw.astype(np.float32)
        valid = np.ones(raw.shape, dtype=bool)
        if hasattr(ds, 'PixelPaddingValue'):
            low = float(ds.PixelPaddingValue)
            high = float(getattr(ds, 'PixelPaddingRangeLimit', low))
            valid = ~((raw >= min(low, high)) & (raw <= max(low, high)))
        gray = np.asarray(apply_voi_lut(apply_modality_lut(raw, ds), ds), dtype=np.float32)
        if photo == 'MONOCHROME1':
            gray = gray.max() + gray.min() - gray
    else:
        raise ValueError(f'Unsupported pixel layout: {photo}, shape={raw.shape}')
    if not np.isfinite(gray).all() or not valid.any():
        raise ValueError('Invalid pixels or all-padding image.')
    lo, hi = float(gray[valid].min()), float(gray[valid].max())
    if hi <= lo:
        raise ValueError('Constant image.')
    normalized = np.clip((gray - lo) / (hi - lo), 0, 1)
    normalized[~valid] = 0
    display = Image.fromarray(np.round(normalized * 255).astype('uint8'), mode='L')
    return ds, raw, original, normalized, display


def letterbox(image):
    # Preserve the entire field of view; center cropping might remove peripheral findings.
    return ImageOps.pad(image.convert('L'), (IMAGE_SIZE, IMAGE_SIZE),
                        method=Image.Resampling.BILINEAR, color=0)


def tensor_transform(name, training=False):
    ops = []
    if training:
        # Small rotation only: no vertical flip or aggressive crop of anatomical evidence.
        ops += [transforms.RandomRotation(5, interpolation=transforms.InterpolationMode.BILINEAR)]
    if not name.startswith('scratch'):
        ops += [transforms.Grayscale(num_output_channels=3)]
    ops += [transforms.ToTensor()]
    if not name.startswith('scratch'):
        ops += [transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]
    return transforms.Compose(ops)


def custom_head(n):
    return nn.Sequential(nn.Linear(n, 256), nn.ReLU(), nn.Dropout(0.4),
                         nn.Linear(256, 64), nn.ReLU(), nn.Dropout(0.2), nn.Linear(64, 1))


class SqueezeExcitation(nn.Module):
    """Channel attention; random initialization, no pretrained parameters."""
    def __init__(self,channels):
        super().__init__()
        self.gate=nn.Sequential(nn.AdaptiveAvgPool2d(1),nn.Conv2d(channels,max(4,channels//8),1),
            nn.ReLU(),nn.Conv2d(max(4,channels//8),channels,1),nn.Sigmoid())
    def forward(self,x):return x*self.gate(x)


def make_model(name, pretrained=True):
    """Explicit ImageNet weight versions; loading saved weights never downloads a backbone."""
    if name.endswith('_custom_frozen'): name=name[:-7]
    if name.startswith('scratch'):
        layers = []
        incoming = 1
        for outgoing in (16, 32, 64, 128):
            layers += [nn.Conv2d(incoming, outgoing, 3, padding=1), nn.BatchNorm2d(outgoing),
                       nn.ReLU(), nn.MaxPool2d(2)]
            if name=='scratch_se_cnn':layers.append(SqueezeExcitation(outgoing))
            incoming = outgoing
        return nn.Sequential(*layers, nn.AdaptiveAvgPool2d(1), nn.Flatten(),
                             nn.Dropout(0.3), nn.Linear(128, 1))
    if name.startswith('resnet18'):
        model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
        model.fc = custom_head(model.fc.in_features) if name.endswith('custom') else nn.Linear(model.fc.in_features, 1)
    elif name.startswith('densenet121'):
        model = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None)
        model.classifier = custom_head(model.classifier.in_features) if name.endswith('custom') else nn.Linear(model.classifier.in_features, 1)
    elif name == 'mobilenet_v3_small':
        model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None)
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, 1)
    elif name == 'vgg16_linear':
        model = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1 if pretrained else None)
        # Replace the large ImageNet classifier with global pooling and a binary head.
        model.avgpool = nn.AdaptiveAvgPool2d(1)
        model.classifier = nn.Linear(512, 1)
    elif name == 'resnet50_linear':
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None)
        model.fc = nn.Linear(model.fc.in_features, 1)
    elif name == 'mobilenet_v2_linear':
        model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1 if pretrained else None)
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, 1)
    elif name == 'convnext_tiny':
        model = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1 if pretrained else None)
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features,1)
    elif name == 'efficientnet_b0_linear':
        model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None)
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, 1)
    else:
        raise ValueError(name)
    return model


def trainable_stage(model, name, finetune=False):
    if name.startswith('scratch'):
        for p in model.parameters(): p.requires_grad = True
        return
    for p in model.parameters(): p.requires_grad = False
    head = model.fc if name.startswith('resnet') else model.classifier
    for p in head.parameters(): p.requires_grad = True
    if finetune:
        if name.startswith('resnet'): block = model.layer4
        elif name.startswith('densenet'): block = model.features.denseblock4
        elif name=='convnext_tiny': block = model.features[-1]
        else: block = model.features[-2:]
        for p in block.parameters(): p.requires_grad = True


def inference_image(source, extension):
    if extension.lower() == '.dcm':
        image = decode_dicom(source)[-1]
    else:
        image = Image.open(source)
        image.load()
        image = image.convert('L')
        arr = np.asarray(image, dtype=np.float32)
        if arr.max() <= arr.min(): raise ValueError('Constant image.')
        image = Image.fromarray(np.round((arr-arr.min())/(arr.max()-arr.min())*255).astype('uint8'))
    return letterbox(image)


def calibrated_probability(logits, a=1., b=0.):
    z = np.clip(a * np.asarray(logits, dtype=float) + b, -60, 60)
    return 1 / (1 + np.exp(-z))
