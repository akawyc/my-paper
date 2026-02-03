from torch import nn
import torch
import torch.nn.functional as F

class WatermarkEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        # 假设你有10个水印张量，每个水印张量的形状是 (1, 77, 768)
        self.conv1 = nn.Conv1d(in_channels=768, out_channels=512, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels=512, out_channels=256, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(in_channels=256, out_channels=128, kernel_size=3, padding=1)
        
        # 全连接层
        self.fc1 = nn.Linear(128 * 77, 1024)  # 128 * 77 是卷积后的输出
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 128)  # 最终的水印嵌入维度

        # 层归一化
        self.ln = nn.LayerNorm(128)
    
    def forward(self, x):
        # x 形状是 (batch_size, 77, 768)
        
        # 确保 x 具有正确的维度
        if x.dim() != 3:
            raise ValueError(f"Expected input tensor with 3 dimensions, but got {x.dim()} dimensions")
        
        x = x.permute(0, 2, 1)  # 转换为 (batch_size, 768, 77)，适应 1D 卷积

        # 通过卷积层
        x = F.relu(self.conv1(x))  # [batch_size, 512, 77]
        x = F.relu(self.conv2(x))  # [batch_size, 256, 77]
        x = F.relu(self.conv3(x))  # [batch_size, 128, 77]
        
        # 展平为向量
        x = x.view(x.size(0), -1)  # 展平为 (batch_size, 128 * 77)
        
        # 通过全连接层
        x = F.relu(self.fc1(x))  # [batch_size, 1024]
        x = F.relu(self.fc2(x))  # [batch_size, 512]
        x = self.fc3(x)  # [batch_size, 128] 最终的水印嵌入
        
        # 层归一化
        x = self.ln(x)
        return x

if __name__ == '__main__':
    # 假设输入是一个水印张量，形状是 [batch_size, 77, 768]
    watermark_encoder = WatermarkEncoder()
    
    # 创建一个模拟的水印输入，尺寸为 [batch_size, 77, 768]
    x = torch.randn(1, 77, 768)
    
    # 获取水印嵌入
    y = watermark_encoder(x)
    print(f"Watermark embedding shape: {y.shape}")
