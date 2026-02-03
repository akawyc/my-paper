import torch
import numpy as np
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm, trange
from pytorch_lightning import seed_everything
from torch import autocast

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
    seed_everything(42)

    # 加载配置和模型
    config = OmegaConf.load("/root/autodl-tmp/stable-diffusion/configs/stable-diffusion/v1-inference.yaml")
    model = load_model_from_config(config, "/root/autodl-tmp/stable-diffusion/models/ldm/stable-diffusion-v1/model.ckpt")

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)

    sampler = DDIMSampler(model)

    # 参数设置
    batch_size = 1
    C, H, W, f = 4, 512, 512, 8
    ddim_steps = 50
    ddim_eta = 0.0
    scale = 7.5
    n_iter = 1

    # 随机起始 latent
    start_code = torch.randn([batch_size, C, H // f, W // f]).to(device)

    # 提示词
    prompt = "A dragon flying over a castle at night, fantasy painting, high detail, concept art"
    data = [batch_size * [prompt]]

    # 多水印强度测试
    strength_list = [0.0, 0.05, 0.1, 0.17, 0.23]

    precision_scope = autocast
    with torch.no_grad():
        with precision_scope("cuda"):
            with model.ema_scope():
                for n in trange(n_iter, desc="Sampling"):
                    for prompts in tqdm(data, desc="data"):
                        # 获取 unconditional 向量
                        uc = model.get_learned_conditioning(batch_size * [""]) if scale != 1.0 else None

                        # 正常提示词编码
                        if isinstance(prompts, tuple):
                            prompts = list(prompts)
                        c = model.get_learned_conditioning(prompts).to(device)  # shape (1, 77, 768)
                        print("Prompt embedding shape:", c.shape)

                        # 生成一个随机水印张量
                        watermark_tensor = torch.randn_like(c)

                        for strength in strength_list:
                            print(f"Generating image with watermark strength: {strength:.2f}")

                            # 混合后的条件向量
                            c_mix = c + strength * watermark_tensor

                            # 执行采样
                            shape = [C, H // f, W // f]
                            samples_ddim, _ = sampler.sample(
                                S=ddim_steps,
                                conditioning=c_mix,
                                batch_size=batch_size,
                                shape=shape,
                                verbose=False,
                                unconditional_guidance_scale=scale,
                                unconditional_conditioning=uc,
                                eta=ddim_eta,
                                x_T=start_code,
                            )

                            # 解码图像
                            x_samples_ddim = model.decode_first_stage(samples_ddim)
                            x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, 0.0, 1.0)
                            x_samples_ddim = x_samples_ddim.cpu().numpy()

                            # 保存图像
                            x_sample = 255.0 * x_samples_ddim
                            img_arr = np.stack([x_sample[0][0], x_sample[0][1], x_sample[0][2]], axis=2)
                            img = Image.fromarray(img_arr.astype(np.uint8))
                            img.save(f"strength_{strength:.2f}.png")
                            print(f"Saved: strength_{strength:.2f}.png")

if __name__ == "__main__":
    main()
