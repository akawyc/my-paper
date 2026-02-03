from torch.utils.data import Dataset
import os
from PIL import Image
import torch
from torchvision.transforms import ToTensor, Compose

# 自定义数据集
class WatermarkedImageDataset(Dataset):
    def __init__(self, image_folder, watermark_tensors, transform=None):
        super().__init__()
        self.image_folder = image_folder
        self.watermark_tensors = watermark_tensors
        self.image_files = [f for f in os.listdir(image_folder) if f.endswith('.png')]  # 获取所有PNG文件
        self.transform = transform  # 如果有其他需要的预处理步骤

    def get_info_from_filename(self, filename):
        """
        根据图像文件名获取水印张量编号、提示词编号和水印强度
        例如：'cyberpunk_brain_interface_surgery_scene_watermark1_intensity0.25.png'
        """
        parts = filename.split('_')  # 根据下划线分割文件名
        
        # 提取提示词部分
        prompt_id = "_".join(parts[:-2])  # 提取提示词，假设水印编号和强度部分是最后两个部分
        prompt_id = prompt_id.replace('_', ' ')  # 如果需要，恢复空格
        
        # 提取水印编号
        watermark_id = int(parts[-2].replace('watermark', ''))  # 'watermark1' -> 1
        
        # 提取水印强度
        intensity = float(parts[-1].replace('intensity', '').replace('.png', ''))  # 'intensity0.25' -> 0.25
        
        # 获取对应水印张量
        watermark_tensor = self.watermark_tensors[watermark_id - 1]  # 假设水印编号从1开始
        
        return watermark_tensor, prompt_id, intensity

    def __len__(self):
        """返回数据集的大小"""
        return len(self.image_files)

    def __getitem__(self, index):
        """
        获取指定索引的图像及其对应的水印张量
        """
        image_file = self.image_files[index]
        image_path = os.path.join(self.image_folder, image_file)
        
        # 加载图像
        img = Image.open(image_path).convert('RGB')  # 确保图像是RGB格式
        
        # 应用预处理（如果有）
        if self.transform:
            img = self.transform(img)
        else:
            # 如果没有传递 transform 参数，手动将图像转换为 Tensor
            img = ToTensor()(img)  # 将PIL图像转换为Tensor并归一化到[0,1]

        # 获取水印张量以及其他信息
        watermark_tensor, prompt_id, intensity = self.get_info_from_filename(image_file)
        
        # 返回处理后的图像，水印张量，提示词和强度
        return img, watermark_tensor, prompt_id, intensity
        

# 如果需要转换（如果你想应用其他预处理方式）
transform = Compose([ToTensor()])

# 数据集实例
image_folder = '/root/autodl-tmp/stable-diffusion/sypng'  # 数据集文件夹路径
watermark_tensors = torch.load('/root/autodl-tmp/stable-diffusion/watermarks.pt')  # 假设你已经生成了水印并保存

dataset = WatermarkedImageDataset(image_folder, watermark_tensors, transform=transform)

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    
    # 打印示例数据
    img, watermark_tensor, prompt_id, intensity = dataset[0]
    print("Prompt:", prompt_id)
    print("Watermark Tensor:", watermark_tensor)
    
    # 显示图像
    plt.imshow(img.permute(1, 2, 0))  # permute是因为Tensor的维度是 (C, H, W)
    plt.show()
