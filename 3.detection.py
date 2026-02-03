import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from clip import CLIP

class WatermarkExtractor:
    def __init__(self, model_path, watermarks_path):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = CLIP().to(self.device)
        
        # 加载模型权重（添加strict=False以兼容可能的维度不匹配）
        state_dict = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(state_dict, strict=False)
        self.model.eval()
        
        # 加载所有候选水印张量
        watermark_data = torch.load(watermarks_path)
        if isinstance(watermark_data, list):
            self.watermark_tensors = torch.stack(watermark_data)
        else:
            self.watermark_tensors = watermark_data
        
        # 确保水印张量是3维的 [N, 77, 768]
        if self.watermark_tensors.dim() == 4:
            self.watermark_tensors = self.watermark_tensors.squeeze(1)
        
        print(f"加载了{len(self.watermark_tensors)}个水印模板，形状: {self.watermark_tensors.shape}")
        
        # 图像预处理（调整为与训练时一致）
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.repeat(3, 1, 1)),  # 灰度转RGB
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    
    def extract_watermark(self, image_path, top_k=3):
        """
        从图像中提取最可能的水印
        """
        try:
            # 加载并预处理图像
            img = Image.open(image_path).convert('L')  # 确保灰度输入
            img_tensor = self.transform(img).unsqueeze(0).to(self.device)
            print(f"输入图像处理后的形状: {img_tensor.shape}")
            
            # 准备水印张量
            watermark_tensors = self.watermark_tensors.to(self.device)
            print(f"水印张量形状: {watermark_tensors.shape}")
            
            # 计算相似度
            with torch.no_grad():
                logits = self.model(img_tensor, watermark_tensors)
                probs = F.softmax(logits, dim=-1)[0]
            
            # 获取top-k结果
            top_probs, top_indices = probs.topk(top_k)
            return top_indices.cpu().numpy(), top_probs.cpu().numpy()
        
        except Exception as e:
            print(f"提取过程中发生错误: {str(e)}")
            return None, None
    
    def visualize_results(self, image_path, top_k=3):
        """可视化提取结果"""
        indices, probs = self.extract_watermark(image_path, top_k)
        if indices is None:
            return
        
        # 显示原始图像
        img = Image.open(image_path)
        plt.figure(figsize=(15, 5))
        plt.subplot(1, top_k+1, 1)
        plt.imshow(img, cmap='gray')
        plt.title("Input Image")
        plt.axis('off')
        
        # 显示匹配结果
        for i, (idx, prob) in enumerate(zip(indices, probs)):
            plt.subplot(1, top_k+1, i+2)
            watermark = self.watermark_tensors[idx].cpu().numpy()
            
            # 可视化水印模板（根据实际水印格式调整）
            if watermark.ndim == 2:  # [77, 768]
                plt.imshow(watermark, cmap='viridis')
            else:  # 其他格式
                plt.imshow(watermark.mean(axis=0), cmap='viridis')
                
            plt.title(f"Match #{i+1}\nProb: {prob:.2%}")
            plt.axis('off')
        
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    # 初始化提取器（确认路径正确）
    extractor = WatermarkExtractor(
        model_path="/root/autodl-tmp/stable-diffusion/mnist-clip-main/model.pth",
        watermarks_path="/root/autodl-tmp/stable-diffusion/watermarks.pt"
    )
    
    # 测试图像路径（确认文件存在）
    test_image = "/root/autodl-tmp/stable-diffusion/sypng/a_beautiful_golden_retriever_running_in_a_field_watermark3_intensity0.05.png"
    
    # 检查文件是否存在
    try:
        Image.open(test_image)
    except FileNotFoundError:
        print(f"错误：测试图像不存在于路径 {test_image}")
        exit()
    
    # 提取并可视化结果
    print("提取结果：")
    indices, probs = extractor.extract_watermark(test_image)
    
    if indices is not None:
        for i, (idx, prob) in enumerate(zip(indices, probs)):
            print(f"Top {i+1}: 水印索引 {idx} (置信度: {prob:.2%})")
        
        # 可视化
        extractor.visualize_results(test_image)