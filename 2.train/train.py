import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import os
from dataset import WatermarkedImageDataset
from clip import CLIP
import torch.multiprocessing as mp

# 设置 multiprocessing 的 start method 为 'spawn'
mp.set_start_method('spawn', force=True)

def validate_shapes(imgs, watermarks):
    """验证输入张量形状的工具函数"""
    print(f"图像张量形状: {imgs.shape} (应为 [B,C,H,W])")
    print(f"水印张量形状: {watermarks.shape} (应为 [B,77,768])")
    assert watermarks.dim() == 3, f"水印必须是3维张量，但得到 {watermarks.dim()} 维"

if __name__ == '__main__':
    # 配置设备
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"使用设备: {DEVICE}")

    # 加载数据集
    image_folder = '/root/autodl-tmp/stable-diffusion/sypng'
    watermark_data = torch.load('/root/autodl-tmp/stable-diffusion/watermarks.pt')
    
    # 将水印数据转换为张量
    if isinstance(watermark_data, list):
        print("检测到水印数据为列表，正在转换为张量...")
        watermark_tensors = torch.stack(watermark_data)
    else:
        watermark_tensors = watermark_data
    
    # 自动处理4维水印数据
    if torch.is_tensor(watermark_tensors) and watermark_tensors.dim() == 4:
        watermark_tensors = watermark_tensors.squeeze(1)
    
    print(f"最终水印数据形状: {watermark_tensors.shape}")

    dataset = WatermarkedImageDataset(image_folder, watermark_tensors)

    # 创建 DataLoader (修改点：移除pin_memory因为数据可能已在GPU)
    BATCH_SIZE = 10
    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        persistent_workers=True,
        pin_memory=False,  # 禁用pin_memory
        multiprocessing_context='spawn'
    )

    # 初始化模型
    model = CLIP().to(DEVICE)
    
    # 尝试加载模型 (修改点：忽略不匹配的权重)
    try:
        state_dict = torch.load('model.pth', map_location=DEVICE)
        # 过滤掉不匹配的权重
        model_state_dict = model.state_dict()
        matched_state_dict = {k: v for k, v in state_dict.items() 
                            if k in model_state_dict and v.shape == model_state_dict[k].shape}
        model.load_state_dict(matched_state_dict, strict=False)
        print(f"加载已有模型权重，匹配了{len(matched_state_dict)}/{len(model_state_dict)}个参数")
    except Exception as e:
        print(f"初始化新模型，原因: {str(e)}")

    # 定义优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # 训练参数
    ITER_BATCH_COUNT = 100000
    TARGET_COUNT = 10

    # 训练循环
    for i in range(ITER_BATCH_COUNT):
        while True:
            try:
                imgs, watermark_tensors, _, _ = next(iter(dataloader))
                break
            except RuntimeError as e:
                print(f"数据加载错误: {str(e)}，重试中...")
                continue
            
            # 验证输入形状
            validate_shapes(imgs, watermark_tensors)

            # 选择包含所有目标水印的批次
            selected_imgs = []
            selected_watermarks = []
            watermark_used = set()

            for watermark_id in range(TARGET_COUNT):
                for j in range(len(imgs)):
                    if (torch.equal(watermark_tensors[j], watermark_tensors[watermark_id]) 
                        and watermark_id not in watermark_used):
                        selected_imgs.append(imgs[j])
                        selected_watermarks.append(watermark_tensors[j])
                        watermark_used.add(watermark_id)
                        break

            if len(selected_imgs) == TARGET_COUNT:
                imgs = torch.stack(selected_imgs)
                watermark_tensors = torch.stack(selected_watermarks)
                break

        # 确保数据在CPU上 (修复pin_memory错误)
        imgs = imgs.cpu().to(DEVICE)
        watermark_tensors = watermark_tensors.cpu().to(DEVICE)

        # 模型前向传播
        logits = model(imgs, watermark_tensors)

        # 计算损失
        targets = torch.arange(0, TARGET_COUNT).to(DEVICE)
        loss_i = F.cross_entropy(logits, targets)
        loss_t = F.cross_entropy(logits.permute(1, 0), targets)
        loss = (loss_i + loss_t) / 2

        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 保存和日志
        if i % 1000 == 0:
            print(f"iter: {i}, loss: {loss.item():.4f}")
            torch.save(model.state_dict(), 'model_temp.pth')
            os.replace('model_temp.pth', 'model.pth')
            print("模型已保存")

    print("训练完成")