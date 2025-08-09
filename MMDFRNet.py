import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    """
    Residual block with attention mechanism
    Adds input (residual) to the processed output with attention modulation
    """

    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
        self.attention = AttentionModule(channels)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        residual = self.attention(residual)  # Apply attention to residual connection
        out += residual
        out = self.relu(out)
        return out


class AttentionModule(nn.Module):
    """
    Simple channel-wise attention module
    Generates attention weights through 1x1 convolution and sigmoid activation
    """

    def __init__(self, channels):
        super(AttentionModule, self).__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.conv(x)
        out = self.sigmoid(out)  # Generate attention weights in [0,1] range
        out = x * out  # Element-wise multiplication for attention modulation
        return out


class MultiModalFusion(nn.Module):
    """
    Multi-modal fusion module for SAR and optical features
    Includes feature recalibration (attention gates) and residual connection
    """

    def __init__(self, spatial_channels, temporal_channels):
        super(MultiModalFusion, self).__init__()
        total_channels = spatial_channels + temporal_channels

        # Attention gate for optical features
        self.optical_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # Global average pooling
            nn.Conv2d(spatial_channels, spatial_channels, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(spatial_channels, spatial_channels, kernel_size=1, bias=False),
            nn.Sigmoid()
        )

        # Attention gate for SAR features
        self.sar_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # Global average pooling
            nn.Conv2d(temporal_channels, temporal_channels, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(temporal_channels, temporal_channels, kernel_size=1, bias=False),
            nn.Sigmoid()
        )

        # Feature enhancement after concatenation
        self.enhance = nn.Sequential(
            nn.Conv2d(total_channels, total_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(total_channels),
            nn.ReLU(inplace=True)
        )

        # Fusion convolution layers
        self.fusion = nn.Sequential(
            nn.Conv2d(total_channels, total_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(total_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(total_channels, total_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(total_channels)
        )

        # Residual connection for stable training
        self.residual = nn.Sequential(
            nn.Conv2d(total_channels, total_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(total_channels)
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        batch_size, channels, h, w = x.size()
        split_size = channels // 2  # Assumes equal channel split between modalities

        # Split input into SAR and optical features
        sar_features = x[:, :split_size]
        optical_features = x[:, split_size:]

        # Apply modality-specific attention
        optical_attention = self.optical_gate(optical_features)
        sar_attention = self.sar_gate(sar_features)

        # Enhance features with attention weights
        enhanced_optical = optical_features * optical_attention
        enhanced_sar = sar_features * sar_attention

        # Combine and process features
        combined = torch.cat([enhanced_optical, enhanced_sar], dim=1)
        enhanced = self.enhance(combined)

        # Fusion with residual connection
        out = self.fusion(enhanced)
        residual = self.residual(combined)
        out = self.relu(out + residual)

        return out


class ASPPConv1x1(nn.Sequential):
    """1x1 convolution module for ASPP"""

    def __init__(self, in_channels, out_channels):
        modules = [nn.Conv2d(in_channels, out_channels, 1, bias=False),
                   nn.BatchNorm2d(out_channels),
                   nn.ReLU(inplace=True), ]
        super(ASPPConv1x1, self).__init__(*modules)


class ASPPConv(nn.Sequential):
    """Dilated convolution module for ASPP"""

    def __init__(self, in_channels, out_channels, dilation):
        modules = [nn.Conv2d(in_channels, out_channels, kernel_size=3,
                             padding=dilation, dilation=dilation, bias=False),
                   nn.BatchNorm2d(out_channels),
                   nn.ReLU(inplace=True), ]
        super(ASPPConv, self).__init__(*modules)


class ASPPPooling(nn.Sequential):
    """Pooling module for ASPP with output size matching"""

    def __init__(self, in_channels, out_channels):
        modules = [nn.AdaptiveAvgPool2d(1),
                   nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                   nn.BatchNorm2d(out_channels),
                   nn.ReLU(inplace=True), ]
        super(ASPPPooling, self).__init__(*modules)

    def forward(self, x):
        size = x.shape[-2:]  # Get spatial dimensions
        for mod in self:
            x = mod(x)
        # Upsample to match original spatial size
        return F.interpolate(x, size=size, mode='bilinear', align_corners=True)


class ASPP(nn.Module):
    """
    Atrous Spatial Pyramid Pooling module
    Captures multi-scale contextual information using different dilation rates
    """

    def __init__(self, in_channels, out_channels):
        super(ASPP, self).__init__()
        # Convolution modules with different receptive fields
        modules = [ASPPConv1x1(in_channels, out_channels),
                   ASPPConv(in_channels, out_channels, dilation=1),
                   ASPPConv(in_channels, out_channels, dilation=2),
                   ASPPConv(in_channels, out_channels, dilation=4),
                   ASPPPooling(in_channels, out_channels), ]
        self.convs = nn.ModuleList(modules)

        # Project combined features to target channels
        self.project = nn.Sequential(nn.Conv2d(5 * out_channels, out_channels, 1, bias=False),
                                     nn.BatchNorm2d(out_channels),
                                     nn.ReLU(inplace=True),
                                     nn.Dropout2d(0.5))  # Regularization

    def forward(self, x):
        output = []
        for mod in self.convs:
            output.append(mod(x))  # Collect outputs from different scales
        x = torch.cat(output, dim=1)  # Concatenate along channel dimension
        x = self.project(x)
        return x


class MultiScaleFeatureFusion(nn.Module):
    """
    Fuses features from different convolutional scales (1x1, 3x3, and dilated 3x3)
    """

    def __init__(self, in_channels, out_channels):
        super(MultiScaleFeatureFusion, self).__init__()
        self.conv1x1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.conv3x3 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.conv3x3_d2 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=2, dilation=2),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # Extract features at different scales
        conv1x1 = self.conv1x1(x)
        conv3x3 = self.conv3x3(x)
        conv3x3_d2 = self.conv3x3_d2(x)
        # Element-wise addition for fusion
        fused_features = torch.add(torch.add(conv3x3, conv1x1), conv3x3_d2)
        return fused_features


class SARFeatureExtractor(nn.Module):
    """
    Feature extraction module specifically designed for SAR data
    Uses dilation to capture larger receptive fields
    """

    def __init__(self, in_channels, out_channels):
        super(SARFeatureExtractor, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels // 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels // 2),
            nn.ReLU(inplace=True)
        )

        self.conv_dilation = nn.Sequential(
            nn.Conv2d(out_channels // 2, out_channels // 2, kernel_size=3, padding=2, dilation=2),
            nn.BatchNorm2d(out_channels // 2),
            nn.ReLU(inplace=True)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv_dilation(x1)  # Dilated convolution for larger context
        x = torch.cat([x1, x2], dim=1)  # Concatenate features
        x = self.conv2(x)
        return x


class OpticalFeatureExtractor(nn.Module):
    """
    Feature extraction module specifically designed for optical data
    Uses depth-wise separable convolution and multi-kernel convolutions
    """

    def __init__(self, in_channels, out_channels):
        super(OpticalFeatureExtractor, self).__init__()

        # Depth-wise separable convolution
        self.conv_ds = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels),
            nn.Conv2d(in_channels, out_channels // 2, kernel_size=1),
            nn.BatchNorm2d(out_channels // 2),
            nn.ReLU(inplace=True)
        )

        # 1x1 convolution for local features
        self.conv_1x1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels // 4, kernel_size=1),
            nn.BatchNorm2d(out_channels // 4),
            nn.ReLU(inplace=True)
        )

        # 5x5 convolution for larger receptive field
        self.conv_5x5 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels // 4, kernel_size=5, padding=2),
            nn.BatchNorm2d(out_channels // 4),
            nn.ReLU(inplace=True)
        )

        # Fusion convolution
        self.fusion = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x1 = self.conv_ds(x)
        x2 = self.conv_1x1(x)
        x3 = self.conv_5x5(x)
        x = torch.cat([x1, x2, x3], dim=1)  # Concatenate multi-scale features
        x = self.fusion(x)
        return x


class DoubleConv(nn.Module):
    """
    Two consecutive convolution layers with batch normalization and ReLU
    Supports both standard and depth-wise separable convolutions
    """

    def __init__(self, in_channels, out_channels, depthwise_separable=False):
        super(DoubleConv, self).__init__()
        if depthwise_separable:
            self.conv = nn.Sequential(
                # Depth-wise convolution
                nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels),
                # Point-wise convolution
                nn.Conv2d(in_channels, out_channels, kernel_size=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )
        else:
            self.conv = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )

    def forward(self, x):
        return self.conv(x)


class ResidualConv(nn.Module):
    """
    Residual convolution block with skip connection
    Adjusts channel dimensions for residual connection when needed
    """

    def __init__(self, in_channels, out_channels):
        super(ResidualConv, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Skip connection (identity if channels match, 1x1 conv otherwise)
        self.skip_connection = nn.Sequential()
        if in_channels != out_channels:
            self.skip_connection = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        residual = self.skip_connection(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual  # Add residual connection
        out = self.relu(out)
        return out


class MultiModalUNet(nn.Module):
    """
    Multi-modal UNet for fusing SAR and optical data
    Uses separate encoders for each modality and fusion modules at multiple levels
    """

    def __init__(self, sar_channels, optical_channels, out_channels):
        super(MultiModalUNet, self).__init__()

        # SAR encoder path
        self.sar_encoder = nn.ModuleDict({
            'conv1': SARFeatureExtractor(sar_channels, 64),
            'residual1': ResidualBlock(64),
            'pool1': nn.MaxPool2d(kernel_size=2, stride=2),
            'conv2': SARFeatureExtractor(64, 128),
            'residual2': ResidualBlock(128),
            'pool2': nn.MaxPool2d(kernel_size=2, stride=2),
            'conv3': SARFeatureExtractor(128, 256),
            'residual3': ResidualBlock(256),
            'pool3': nn.MaxPool2d(kernel_size=2, stride=2),
            'conv4': SARFeatureExtractor(256, 512),
            'residual4': ResidualBlock(512),
            'pool4': nn.MaxPool2d(kernel_size=2, stride=2),
            'conv5': SARFeatureExtractor(512, 1024),
            'residual5': ResidualBlock(1024)
        })

        # Optical encoder path
        self.optical_encoder = nn.ModuleDict({
            'conv1': OpticalFeatureExtractor(optical_channels, 64),
            'residual1': ResidualBlock(64),
            'pool1': nn.MaxPool2d(kernel_size=2, stride=2),
            'conv2': OpticalFeatureExtractor(64, 128),
            'residual2': ResidualBlock(128),
            'pool2': nn.MaxPool2d(kernel_size=2, stride=2),
            'conv3': OpticalFeatureExtractor(128, 256),
            'residual3': ResidualBlock(256),
            'pool3': nn.MaxPool2d(kernel_size=2, stride=2),
            'conv4': OpticalFeatureExtractor(256, 512),
            'residual4': ResidualBlock(512),
            'pool4': nn.MaxPool2d(kernel_size=2, stride=2),
            'conv5': OpticalFeatureExtractor(512, 1024),
            'residual5': ResidualBlock(1024)
        })

        # Bottleneck and fusion components
        self.bottleneck = nn.ModuleDict({
            'concat': nn.Conv2d(2048, 2048, 1, bias=False),
            'mmf1': MultiModalFusion(1024, 1024),
            'aspp': ASPP(2048, 1024),
            'conv4': DoubleConv(1024, 2048)
        })

        # Decoder components
        self.decoder = nn.ModuleDict({
            'up5': nn.ConvTranspose2d(2048, 1024, kernel_size=2, stride=2),
            'mmf2': MultiModalFusion(512, 512),
            'fusion1': MultiScaleFeatureFusion(2048, 1024),
            'conv5': DoubleConv(1024, 1024),

            'up6': nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2),
            'mmf3': MultiModalFusion(256, 256),
            'fusion2': MultiScaleFeatureFusion(1024, 512),
            'conv6': DoubleConv(512, 512),

            'up7': nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2),
            'mmf4': MultiModalFusion(128, 128),
            'fusion3': MultiScaleFeatureFusion(512, 256),
            'conv7': DoubleConv(256, 256),

            'up8': nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),
            'mmf5': MultiModalFusion(64, 64),
            'fusion4': MultiScaleFeatureFusion(256, 128),
            'conv8': DoubleConv(128, 64),

            'conv9': nn.Conv2d(64, out_channels, kernel_size=1)
        })

    def forward(self, sar_data, optical_data):
        # Extract features from SAR encoder
        x1_sar = self.sar_encoder['conv1'](sar_data)
        x1_sar_residual = self.sar_encoder['residual1'](x1_sar)
        x1_sar_pool = self.sar_encoder['pool1'](x1_sar_residual)

        x2_sar = self.sar_encoder['conv2'](x1_sar_pool)
        x2_sar_residual = self.sar_encoder['residual2'](x2_sar)
        x2_sar_pool = self.sar_encoder['pool2'](x2_sar_residual)

        x3_sar = self.sar_encoder['conv3'](x2_sar_pool)
        x3_sar_residual = self.sar_encoder['residual3'](x3_sar)
        x3_sar_pool = self.sar_encoder['pool3'](x3_sar_residual)

        x4_sar = self.sar_encoder['conv4'](x3_sar_pool)
        x4_sar_residual = self.sar_encoder['residual4'](x4_sar)
        x4_sar_pool = self.sar_encoder['pool4'](x4_sar_residual)

        x5_sar = self.sar_encoder['conv5'](x4_sar_pool)
        x5_sar_residual = self.sar_encoder['residual5'](x5_sar)

        # Extract features from optical encoder
        x1_optical = self.optical_encoder['conv1'](optical_data)
        x1_optical_residual = self.optical_encoder['residual1'](x1_optical)
        x1_optical_pool = self.optical_encoder['pool1'](x1_optical_residual)

        x2_optical = self.optical_encoder['conv2'](x1_optical_pool)
        x2_optical_residual = self.optical_encoder['residual2'](x2_optical)
        x2_optical_pool = self.optical_encoder['pool2'](x2_optical_residual)

        x3_optical = self.optical_encoder['conv3'](x2_optical_pool)
        x3_optical_residual = self.optical_encoder['residual3'](x3_optical)
        x3_optical_pool = self.optical_encoder['pool3'](x3_optical_residual)

        x4_optical = self.optical_encoder['conv4'](x3_optical_pool)
        x4_optical_residual = self.optical_encoder['residual4'](x4_optical)
        x4_optical_pool = self.optical_encoder['pool4'](x4_optical_residual)

        x5_optical = self.optical_encoder['conv5'](x4_optical_pool)
        x5_optical_residual = self.optical_encoder['residual5'](x5_optical)

        # Process bottleneck
        x = torch.cat([x5_sar_residual, x5_optical_residual], dim=1)
        x = self.bottleneck['concat'](x)
        x_mmf1 = self.bottleneck['mmf1'](x)
        x = self.bottleneck['aspp'](x_mmf1)
        x = self.bottleneck['conv4'](x)

        # Decoder path
        x1 = self.decoder['up5'](x)
        x = torch.cat([x4_sar_residual, x4_optical_residual], dim=1)
        x_mmf2 = self.decoder['mmf2'](x)
        x = torch.cat([x_mmf2, x1], dim=1)
        x = self.decoder['fusion1'](x)
        x = self.decoder['conv5'](x)

        x2 = self.decoder['up6'](x)
        x = torch.cat([x3_sar_residual, x3_optical_residual], dim=1)
        x_mmf3 = self.decoder['mmf3'](x)
        x = torch.cat([x_mmf3, x2], dim=1)
        x = self.decoder['fusion2'](x)
        x = self.decoder['conv6'](x)

        x3 = self.decoder['up7'](x)
        x = torch.cat([x2_sar_residual, x2_optical_residual], dim=1)
        x_mmf4 = self.decoder['mmf4'](x)
        x = torch.cat([x_mmf4, x3], dim=1)
        x = self.decoder['fusion3'](x)
        x = self.decoder['conv7'](x)

        x4 = self.decoder['up8'](x)
        x = torch.cat([x1_sar_residual, x1_optical_residual], dim=1)
        x_mmf5 = self.decoder['mmf5'](x)
        x = torch.cat([x_mmf5, x4], dim=1)
        x = self.decoder['fusion4'](x)
        x = self.decoder['conv8'](x)

        # Final output layer
        output = self.decoder['conv9'](x)
        return output


# Example usage
if __name__ == "__main__":
    # Create model instance (example parameters)
    sar_channels = 12  # Typical for SAR data
    optical_channels = 10  # Typical for RGB optical data
    out_channels = 2  # Number of output classes

    model = MultiModalUNet(sar_channels, optical_channels, out_channels)

    # Test with dummy data
    sar_input = torch.randn(2, sar_channels, 256, 256)  # (batch_size, channels, height, width)
    optical_input = torch.randn(2, optical_channels, 256, 256)

    output = model(sar_input, optical_input)
    print(f"Input SAR shape: {sar_input.shape}")
    print(f"Input Optical shape: {optical_input.shape}")
    print(f"Output shape: {output.shape}")