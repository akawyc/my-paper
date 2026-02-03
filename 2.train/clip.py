from torch import nn
import torch
from img_encoder import ImgEncoder
from watermark_encoder import WatermarkEncoder  # 修改为水印编码器

class CLIP(nn.Module):
    def __init__(self):
        super().__init__()
        self.img_enc = ImgEncoder()  # 图像编码器
        self.watermark_enc = WatermarkEncoder()  # 水印编码器

    def forward(self, img_x, watermark_x):
        # 获取图像嵌入
        img_emb = self.img_enc(img_x)
        # 获取水印嵌入
        watermark_emb = self.watermark_enc(watermark_x)
        # 计算相似度
        return img_emb @ watermark_emb.T  # 点积计算相似度

if __name__ == '__main__':
    clip = CLIP()

    # 创建一个模拟的图像输入，尺寸为 [batch_size, channels, height, width] = [5, 3, 512, 512]
    img_x = torch.randn(5, 3, 512, 512)  # 5张RGB图像，大小为 512x512

    # 创建一个模拟的水印输入，尺寸为 [batch_size, 77, 768] = [5, 77, 768]
    watermark_x = torch.randn(5, 77, 768)  # 5个水印张量

    # 计算图像和水印的相似度
    logits = clip(img_x, watermark_x)
    
    print(logits.shape)  # 输出相似度矩阵的形状
