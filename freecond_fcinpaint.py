import torch
import cv2
import json
import os
import numpy as np
from PIL import Image
import argparse
import pandas as pd
import torch
import torch.nn.functional as F
import numpy as np
from torchmetrics.multimodal import CLIPScore
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from urllib.request import urlretrieve 
from PIL import Image
import open_clip
import os
import hpsv2
import ImageReward as RM
import math
from tqdm import tqdm

from freecond_src.freecond import get_pipeline,fc_config

def to_masked(img1, mask_image):
    mask_image = mask_image.convert("L")

    # Create a black image of the same size as the RGB image
    black_image = Image.new("RGB", img1.size, color=(0, 0, 0))

    # Apply the mask: Combine the original image and the black image using the mask
    masked_image = Image.composite( black_image, img1, mask_image)
    return masked_image

def rle2mask(mask_rle, shape): # height, width
    starts, lengths = [np.asarray(x, dtype=int) for x in (mask_rle[0:][::2], mask_rle[1:][::2])]
    starts -= 1
    ends = starts + lengths
    binary_mask = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        binary_mask[lo:hi] = 1
    return binary_mask.reshape(shape)


class MetricsCalculator:
    def __init__(self, device,ckpt_path="data/ckpt") -> None:
        self.device=device
        # clip
        self.clip_metric_calculator = CLIPScore(model_name_or_path="openai/clip-vit-large-patch14").to(device)
        # lpips
        self.lpips_metric_calculator = LearnedPerceptualImagePatchSimilarity(net_type='squeeze').to(device)
        # aesthetic model
        self.aesthetic_model = torch.nn.Linear(768, 1)
        aesthetic_model_url = (
                    "https://github.com/LAION-AI/aesthetic-predictor/blob/main/sa_0_4_vit_l_14_linear.pth?raw=true"
                )
        aesthetic_model_ckpt_path=os.path.join(ckpt_path,"sa_0_4_vit_l_14_linear.pth")
        urlretrieve(aesthetic_model_url, aesthetic_model_ckpt_path)
        self.aesthetic_model.load_state_dict(torch.load(aesthetic_model_ckpt_path))
        self.aesthetic_model.eval()
        self.clip_model, _, self.clip_preprocess = open_clip.create_model_and_transforms('ViT-L-14', pretrained='openai')
        # image reward model
        self.imagereward_model = RM.load("ImageReward-v1.0")
 

    def calculate_image_reward(self,image,prompt):
        reward = self.imagereward_model.score(prompt, [image])
        return reward

    def calculate_hpsv21_score(self,image,prompt):
        result = hpsv2.score(image, prompt, hps_version="v2.1")[0]
        return result.item()

    def calculate_aesthetic_score(self,img):
        image = self.clip_preprocess(img).unsqueeze(0)
        with torch.no_grad():
            image_features = self.clip_model.encode_image(image)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            prediction = self.aesthetic_model(image_features)
        return prediction.cpu().item()

    def calculate_clip_similarity(self, img, txt):
        img = np.array(img)
        
        img_tensor=torch.tensor(img).permute(2,0,1).to(self.device)
        
        score = self.clip_metric_calculator(img_tensor, txt)
        score = score.cpu().item()
        
        return score
    
    def calculate_psnr(self, img_pred, img_gt, mask=None):
        img_pred = np.array(img_pred).astype(np.float32)/255.
        img_gt = np.array(img_gt).astype(np.float32)/255.

        assert img_pred.shape == img_gt.shape, "Image shapes should be the same."
        if mask is not None:
            mask = np.array(mask).astype(np.float32)
            img_pred = img_pred * mask
            img_gt = img_gt * mask
        
        difference = img_pred - img_gt
        difference_square = difference ** 2
        difference_square_sum = difference_square.sum()
        difference_size = mask.sum()

        mse = difference_square_sum/difference_size

        if mse < 1.0e-10:
            return 1000
        PIXEL_MAX = 1
        return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))

    
    def calculate_lpips(self, img_gt, img_pred, mask=None):
        img_pred = np.array(img_pred).astype(np.float32)/255
        img_gt = np.array(img_gt).astype(np.float32)/255
        assert img_pred.shape == img_gt.shape, "Image shapes should be the same."

        if mask is not None:
            mask = np.array(mask).astype(np.float32)
            img_pred = img_pred * mask 
            img_gt = img_gt * mask
            
        img_pred_tensor=torch.tensor(img_pred).permute(2,0,1).unsqueeze(0).to(self.device)
        img_gt_tensor=torch.tensor(img_gt).permute(2,0,1).unsqueeze(0).to(self.device)
            
        score =  self.lpips_metric_calculator(img_pred_tensor*2-1,img_gt_tensor*2-1)
        score = score.cpu().item()
        
        return score
    
    def calculate_mse(self, img_pred, img_gt, mask=None):
        img_pred = np.array(img_pred).astype(np.float32)/255.
        img_gt = np.array(img_gt).astype(np.float32)/255.

        assert img_pred.shape == img_gt.shape, "Image shapes should be the same."
        if mask is not None:
            mask = np.array(mask).astype(np.float32)
            img_pred = img_pred * mask
            img_gt = img_gt * mask
        
        difference = img_pred - img_gt
        difference_square = difference ** 2
        difference_square_sum = difference_square.sum()
        difference_size = mask.sum()

        mse = difference_square_sum/difference_size

        return mse.item()
    


parser = argparse.ArgumentParser()

parser.add_argument('--model', 
                    type=str, 
                    default="sd15",
                    help="Currently support [sd15, sd2, ds8, sdxl]")
parser.add_argument('--save_dir', 
                    type=str, 
                    default="trial 1",)
parser.add_argument('--data_csv', 
                    type=str, 
                    default="FCinpaint_bench_info.csv")
parser.add_argument('--blended', action='store_true')
parser.add_argument("--no_freecond",action="store_true")
parser.add_argument("--change_step",
                    type=int,
                    default=10)
parser.add_argument("--inf_step",
                    type=int,
                    default=50)
parser.add_argument("--fg_1",
                    type=float,
                    default=1)
parser.add_argument("--fg_2",
                    type=float,
                    default=1)
parser.add_argument("--bg_1",
                    type=float,
                    default=0)
parser.add_argument("--bg_2",
                    type=float,
                    default=0)
parser.add_argument("--lq_1",
                    type=float,
                    default=1)
parser.add_argument("--lq_2",
                    type=float,
                    default=1)
parser.add_argument("--hq_1",
                    type=float,
                    default=1)
parser.add_argument("--hq_2",
                    type=float,
                    default=1)
parser.add_argument("--q_th",
                    type=int,
                    default=8)
parser.add_argument("--gs",
                    type=float,
                    default=15)                                 
args = parser.parse_args()

device = "cuda" if torch.cuda.is_available() else "cpu"

pipe = get_pipeline(args.model).to(device)
if args.no_freecond:
    fc_control= fc_config(0,1,1,0,0,1,1,1,1,32)
else:
    fc_control= fc_config(args.change_step, args.fg_1, args.fg_2, args.bg_1, args.bg_2,
            args.hq_1, args.hq_2, args.lq_1, args.lq_2, args.q_th)
# memory optimization.
# pipe.enable_model_cpu_offload()
save_dir_path=f"runs/{args.model}/{args.save_dir}"


df=pd.read_csv(args.data_csv, index_col=None)
for index, data in df.iterrows():
    caption=data["prompt"]
    image_path=data["image"]
    mask_path=data["mask"]

    init_image=Image.open(image_path).resize((512,512)).convert("RGB")
    mask_image=Image.open(mask_path).resize((512,512))
    generator = torch.Generator(device).manual_seed(1234)
    nprompt="word, bad quality, bad anatomy, ugly, mutation, blurry, error"
    save_path= os.path.join(save_dir_path,image_path) 
    masked_image_save_path=save_path.replace(".jpg","_masked.jpg")

    image = pipe.freecond_forward_staged(
    fc_control,
    caption, init_image, mask_image,
    negative_prompt=nprompt,
    guidance_scale=args.gs,
    num_inference_steps=args.inf_step, 
    generator=generator,
    )[0]
        
    

    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))

    if args.blended:
        h=image.height
        w=image.width
        mask_np=cv2.resize(cv2.imread(mask_path),(h,w))/255
        image_np=np.array(image)
        init_image_np=cv2.resize(cv2.imread(image_path), (h,w))[:,:,::-1]

        # blur
        mask_blurred = cv2.GaussianBlur(mask_np*255, (21, 21), 0)/255
        mask_np = 1-(1-mask_np) * (1-mask_blurred)
        image_pasted=init_image_np * (1-mask_np) + image_np*mask_np
        image_pasted=image_pasted.astype(image_np.dtype)
        image=Image.fromarray(image_pasted)

    image.save(save_path)
    masked_image=to_masked(init_image, mask_image)
    masked_image.save(masked_image_save_path)

# evaluation
evaluation_df = pd.DataFrame(columns=['Image ID','Image Reward', 'HPS V2.1', 'Aesthetic Score', 'PSNR', 'LPIPS', 'MSE', 'CLIP Similarity'])

metrics_calculator=MetricsCalculator(device)

for index, data in tqdm(df.iterrows()):
    prompt=data["prompt"]
    image_path=data["image"]
    mask_path=data["mask"]

    src_image = Image.open(image_path).resize((512,512))

    tgt_image_path=os.path.join(save_dir_path, image_path)
    tgt_image = Image.open(tgt_image_path).resize((512,512))

    evaluation_result=[index]
        
    mask = cv2.resize(cv2.imread(mask_path),(512,512))//255
    mask = 1 - mask

    for metric in evaluation_df.columns.values.tolist()[1:]:
        print(f"evluating metric: {metric}")

        if metric == 'Image Reward':
            metric_result = metrics_calculator.calculate_image_reward(tgt_image,prompt)
            
        if metric == 'HPS V2.1':
            metric_result = metrics_calculator.calculate_hpsv21_score(tgt_image,prompt)
        
        if metric == 'Aesthetic Score':
            metric_result = metrics_calculator.calculate_aesthetic_score(tgt_image)
        
        if metric == 'PSNR':
            metric_result = metrics_calculator.calculate_psnr(src_image, tgt_image, mask)
        
        if metric == 'LPIPS':
            metric_result = metrics_calculator.calculate_lpips(src_image, tgt_image, mask)
        
        if metric == 'MSE':
            metric_result = metrics_calculator.calculate_mse(src_image, tgt_image, mask)
        
        if metric == 'CLIP Similarity':
            metric_result = metrics_calculator.calculate_clip_similarity(tgt_image, prompt)

        evaluation_result.append(metric_result)
    
    evaluation_df.loc[len(evaluation_df.index)] = evaluation_result

print("The averaged evaluation result:")
averaged_results=evaluation_df.mean(numeric_only=True)
print(averaged_results)
averaged_results.to_csv(os.path.join(save_dir_path,"evaluation_result_sum.csv"))
evaluation_df.to_csv(os.path.join(save_dir_path,"evaluation_result.csv"))

print(f"The generated images and evaluation results is saved in {save_dir_path}")