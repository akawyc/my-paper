# generate_watermarks.py
import torch

def generate_watermarks(num_watermarks, shape, difference_strength=1.0, device='cuda'):
    """
    生成多个差异性较大的水印张量
    :param num_watermarks: 水印张量的数量
    :param shape: 水印张量的形状 (如 (1, 77, 768))
    :param difference_strength: 控制水印之间差异的强度
    :param device: 当前设备 (cuda 或 cpu)
    :return: 生成的水印张量列表
    """
    watermarks = []
    for i in range(num_watermarks):
        watermark = torch.randn(shape).to(device)  # 随机生成水印张量
        if i > 0:
            watermark = watermark + difference_strength * torch.randn_like(watermark)  # 增加差异
        watermarks.append(watermark)
    
    # 保存水印张量为文件
    torch.save(watermarks, 'watermarks.pt')
    print(f"Saved {num_watermarks} watermark tensors to 'watermarks.pt'")

if __name__ == "__main__":
    # 生成 10 个差异性较大的水印张量，形状为 (1, 77, 768)
    generate_watermarks(10, (1, 77, 768), difference_strength=1.0, device='cuda')
