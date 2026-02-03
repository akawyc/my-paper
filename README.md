MU-PP Mark
Enhancing Traceability in Multi-User Diffusion Models via Prompt Perturbation Watermarking

📄 Official PyTorch implementation of the paper:
“Enhancing Traceability in Multi-User Diffusion Models via Prompt Perturbation Watermarking”

🔍 Overview
MU-PP Mark is an implicit multi-user watermarking framework designed for text-to-image diffusion models (e.g., Stable Diffusion).
It enables reliable user attribution and provenance tracing of generated images without modifying model architectures or parameters.

Key idea
Instead of embedding watermarks into pixels or model weights, MU-PP Mark:

Injects user-specific watermark tensors into the prompt embedding space
Propagates identity information throughout the diffusion process
Recovers ownership using contrastive learning–based image–watermark matching
Fetched content

✨ Features
✅ Multi-user attribution (scales to dozens or hundreds of users)
✅ No modification to diffusion model weights
✅ Prompt-perturbation based implicit watermarking
✅ High robustness to compression, noise, blur, and color attacks
✅ High image fidelity (PSNR 37.01 dB @ α = 0.2)
✅ 99% Top-1 identification accuracy
📁 Repository Structure
stable-diffusion/
├── 1.watermark-injection/
│   ├── single-marking.py        # Generate a single watermarked image
│   ├── dataset-marking.py       # Generate large-scale watermarked dataset
│   ├── watermark_tensor.py      # Watermark tensor generation
│   └── watermarks.pt            # Pre-generated watermark tensors
│
├── 2.train/
│   ├── train.py                 # Contrastive training script
│   ├── dataset.py               # Dataset loader
│   ├── img_encoder.py           # Image encoder (ResNet-based)
│   ├── watermark_encoder.py     # Watermark encoder ([77, 768] tensor)
│   └── clip.py                  # CLIP-based text encoder wrapper
│
└── 3.watermark_retrieval.py     # Watermark detection / user attribution
Each folder corresponds to a stage in the MU-PP Mark pipeline:

Watermark embedding
Multi-user contrastive training
Watermark detection
⚙️ Installation
Requirements
Python ≥ 3.10
PyTorch ≥ 2.0
CUDA-enabled GPU recommended
Install dependencies:

pip install -r requirements.txt
Example requirements.txt:

torch>=2.0
torchvision
diffusers
transformers
numpy
opencv-python
lpips
tqdm
🚀 Usage
1️⃣ Watermark Embedding
Generate a single watermarked image
python 1.watermark-injection/single-marking.py \
  --prompt "A photo of a mountain landscape" \
  --user_id 0 \
  --alpha 0.2
Generate a watermarked dataset
python 1.watermark-injection/dataset-marking.py \
  --num_users 10 \
  --num_prompts 1000 \
  --alpha 0.2
α (watermark strength) controls the trade-off between image quality and watermark detectability.
Based on our experiments, α = 0.2 provides the best balance.

2️⃣ Contrastive Training
Train the multi-user watermark retrieval model:

python 2.train/train.py \
  --batch_size 10 \
  --num_users 10 \
  --lr 1e-3 \
  --epochs 100
⚠️ Important:
The training batch size must equal the number of users, as each batch contains exactly one watermark per user.

3️⃣ Watermark Detection (User Attribution)
Identify the source user of a generated image:

python 3.watermark_retrieval.py \
  --image_path example.png
The script outputs the user ID with the highest cosine similarity.

📊 Experimental Results
Metric	Value
Top-1 Accuracy	0.99
PSNR	37.01 dB
SSIM	0.93
LPIPS	0.04
Strong intra-class compactness (≈ 0.25)
Clear inter-class separation (≈ 0.70)
Robust against JPEG compression, blur, noise, and color distortions
🔁 Reproducibility Notes
Watermark tensors are provided in watermarks.pt
Random seeds can be fixed in watermark_tensor.py
All experiments were conducted with Stable Diffusion + CLIP text encoder
📜 License
This project is released under the MIT License.
See LICENSE for details.

📖 Citation
If you find this work useful, please cite:

@article{shi2025muppmark,
  title={Enhancing Traceability in Multi-User Diffusion Models via Prompt Perturbation Watermarking},
  author={Shi, Hui and Wang, Yuchen and Jin, Conghui and Liu, Mingyang},
  journal={},
  year={2025}
}
🙏 Acknowledgements
This work was supported by:

Liaoning Provincial Science and Technology Joint Plan (No. 2025-MSLH-435)
National Natural Science Foundation of China (Grant No. 61601214)
