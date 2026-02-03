import torch
import numpy as np
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm, trange
from pytorch_lightning import seed_everything
from torch import autocast
import os

from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler

# 从配置文件创建模型
def load_model_from_config(config, ckpt, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    model.cuda()
    model.eval()
    return model

def main():
    # 设置随机数种子
    seed_everything(42)
    
    # 加载配置文件
    config = OmegaConf.load("/root/autodl-tmp/stable-diffusion/configs/stable-diffusion/v1-inference.yaml")  # 使用绝对路径

    # 从加载的配置文件创建模型
    model = load_model_from_config(config, "/root/autodl-tmp/stable-diffusion/models/ldm/stable-diffusion-v1/model.ckpt")  # 使用绝对路径

    # 判断GPU或者CPU
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # 将模型送到CUDA设备 或者 CPU
    model = model.to(device)

    # 创建ddim 采样器
    sampler = DDIMSampler(model)

    # 批次大小
    batch_size = 1
    # 起始编码 (可以设置为噪声向量)
    start_code = torch.randn([batch_size, 4, 64, 64]).to(device)  # 或者根据需要初始化start_code
    # CFG 缩放
    scale = 7.5
    # 采样步数
    ddim_steps = 50
    # 通道数
    C = 4
    # 高
    H = 512
    # 宽
    W = 512
    # 下采样因子
    f = 8
    # 采样数量
    n_samples = 1
    # 确定性因子
    ddim_eta = 0.0
    # 迭代次数
    n_iter = 1
    
    # 提示词（目前为空，你可以将你的1000个提示词填入这里）
    prompts = [
        # 填写提示词...
        "an ancient temple hidden deep in the jungle"

"a futuristic metropolis bathed in neon lights"

    ]
    
    # 设置水印强度为0.2
    intensity = 0.2

    # 加载水印张量
   # watermarks = torch.load('watermarks.pt')  # 假设你已经生成了10个水印并保存

    print("Loaded watermarks:", len(watermarks))

    # 存储路径：修改为你想保存的目录
    save_directory = '/root/autodl-tmp/stable-diffusion/png'
    os.makedirs(save_directory, exist_ok=True)  # 确保保存目录存在

    # 一些杂事
    precision_scope = autocast
    img_count = 0  # 用于计数生成的图像数量
    with torch.no_grad():
        with precision_scope("cuda"):
            with model.ema_scope():
                for n in trange(n_iter, desc="Sampling"):
                    for prompt in tqdm(prompts, desc="Prompt"):
                        for i in range(len(watermarks)):  # 每个水印张量生成一张图像
                            watermark_tensor = watermarks[i].to(device)  # 使用不同的水印张量
                            print(f"Using prompt: {prompt}, Watermark: {i+1}, Intensity: {intensity}")

                            # 获取提示词的条件嵌入
                            c = model.get_learned_conditioning([prompt])

                            # 将水印张量添加到条件嵌入中
                            c_with_watermark = c + intensity * watermark_tensor  # 将水印张量添加到条件嵌入中
                             
                            # 采样数据
                            shape = [C, H // f, W // f]
                            samples_ddim, _ = sampler.sample(S=ddim_steps,
                                                             conditioning=c_with_watermark,
                                                             batch_size=n_samples,
                                                             shape=shape,
                                                             verbose=False,
                                                             unconditional_guidance_scale=scale,
                                                             unconditional_conditioning=None,
                                                             eta=ddim_eta,
                                                             x_T=start_code)

                            # VAE 解码器 输出最终图像
                            x_samples_ddim = model.decode_first_stage(samples_ddim)
                            # 把图像值域缩放到 0-1之间
                            x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)
                            x_samples_ddim = x_samples_ddim.cpu().numpy()

                            print("image shape:", x_samples_ddim.shape)
                            # 图像值域从0-1缩放到0-255
                            x_sample = 255. * x_samples_ddim
                            
                            # 保存图像到指定目录，简化文件名格式为 提示词-水印编号
                            img_filename = f"{save_directory}/{prompt.replace(' ', '_')}_watermark{i+1}.png"
                            img1 = np.stack([x_sample[0][0, :, :], x_sample[0][1, :, :], x_sample[0][2, :, :]], axis=2)
                            img = Image.fromarray(img1.astype(np.uint8))
                            img.save(img_filename)  # 按顺序保存图像文件

                            img_count += 1
                            if img_count >= 10:  # 生成10000张图像
                                print(f"Generated {img_count} images.")
                                return

if __name__ == "__main__":
    main()
