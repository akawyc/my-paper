from torch import nn
import torch
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1, stride=stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        self.conv2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1, stride=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.conv3 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, padding=0, stride=stride)
    
    def forward(self, x):
        y = F.relu(self.bn1(self.conv1(x)))  # 使用 ReLU 激活
        y = self.bn2(self.conv2(y))  # 进行 BN 和卷积操作
        z = self.conv3(x)  # 1x1 卷积操作用于通道数匹配
        return F.relu(y + z)  # 最终激活


class ImgEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        # 修改残差块的参数以适应512x512的RGB图像
        self.res_block1 = ResidualBlock(in_channels=3, out_channels=64, stride=2)  # 输入: (batch, 3, 512, 512) -> 输出: (batch, 64, 256, 256)
        self.res_block2 = ResidualBlock(in_channels=64, out_channels=128, stride=2)  # 输入: (batch, 64, 256, 256) -> 输出: (batch, 128, 128, 128)
        self.res_block3 = ResidualBlock(in_channels=128, out_channels=256, stride=2)  # 输入: (batch, 128, 128, 128) -> 输出: (batch, 256, 64, 64)
        self.res_block4 = ResidualBlock(in_channels=256, out_channels=512, stride=2)  # 输入: (batch, 256, 64, 64) -> 输出: (batch, 512, 32, 32)
        
        # 对特征进行线性映射
        self.fc = nn.Linear(in_features=512 * 32 * 32, out_features=128)  # 展平后的特征维度
        
        # 归一化
        self.ln = nn.LayerNorm(128)

    def forward(self, x):
        # 通过各个残差块
        x = self.res_block1(x)
        x = self.res_block2(x)
        x = self.res_block3(x)
        x = self.res_block4(x)
        
        # 展平并通过全连接层
        x = x.view(x.size(0), -1)  # 展平为 [batch_size, 512 * 32 * 32]
        x = self.fc(x)
        
        # 归一化
        x = self.ln(x)
        return x


if __name__ == '__main__':
    img_encoder = ImgEncoder()
    
    # 创建一个模拟的图像输入，尺寸为 [batch_size, channels, height, width] = [1, 3, 512, 512]
    x = torch.randn(1, 3, 512, 512)
    
    # 获取图像嵌入
    y = img_encoder(x)
    print(f"Image embedding shape: {y.shape}")
