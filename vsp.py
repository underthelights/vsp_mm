import torch
from schedulers.scheduling_ddim import DDIMScheduler
import csv
from PIL import Image
import os
from pipelines.pipeline_stable_diffusion_xl import StableDiffusionXLPipeline
from pipelines.inverted_ve_pipeline import create_image_grid
from utils import memory_efficient, init_latent
import argparse
from diffusers.models.attention_processor import AttnProcessor


from pipelines.inverted_ve_pipeline import CrossFrameAttnProcessor, CrossFrameAttnProcessor_store, ACTIVATE_LAYER_CANDIDATE

results_dir = '/data/khshim/attention_map/gogh'

def create_image_grid(image_list, rows, cols, padding=10):
    # Ensure the number of rows and columns doesn't exceed the number of images
    rows = min(rows, len(image_list))
    cols = min(cols, len(image_list))

    # Get the dimensions of a single image
    image_width, image_height = image_list[0].size

    # Calculate the size of the output image
    grid_width = cols * (image_width + padding) - padding
    grid_height = rows * (image_height + padding) - padding

    # Create an empty grid image
    grid_image = Image.new('RGB', (grid_width, grid_height), (255, 255, 255))

    # Paste images into the grid
    for i, img in enumerate(image_list[:rows * cols]):
        row = i // cols
        col = i % cols
        x = col * (image_width + padding)
        y = row * (image_height + padding)
        grid_image.paste(img, (x, y))

    return grid_image

def transform_variable_name(input_str, attn_map_save_step):
    # Split the input string into parts using the dot as a separator
    parts = input_str.split('.')

    # Extract numerical indices from the parts
    indices = [int(part) if part.isdigit() else part for part in parts]

    # Build the desired output string
    output_str = f'pipe.unet.{indices[0]}[{indices[1]}].{indices[2]}[{indices[3]}].{indices[4]}[{indices[5]}].{indices[6]}.attn_map[{attn_map_save_step}]'

    return output_str


num_images_per_prompt = 4
seeds=[1] #craft_clay


activate_layer_indices_list = [
    # ((0,28),(108,140)),
    # ((0,48), (68,140)),
    # ((0,48), (88,140)),
    # ((0,48), (108,140)),
    # ((0,48), (128,140)),
    # ((0,48), (140,140)),
    # ((0,28), (68,140)),
    # ((0,28), (88,140)),
    # ((0,28), (108,140)),
    # ((0,28), (128,140)),
    # ((0,28), (140,140)),
    # ((0,8), (68,140)),
    # ((0,8), (88,140)),
    # ((0,8), (108,140)),
    # ((0,8), (128,140)),
    # ((0,8), (140,140)),
    # ((0,0), (68,140)),
    # ((0,0), (88,140)),
    ((0,0), (108,140)),
    # ((0,0), (128,140)),
    # ((0,0), (140,140))    
]

save_layer_list = [
        # 'up_blocks.0.attentions.1.transformer_blocks.0.attn1.processor', #68
        # 'up_blocks.0.attentions.1.transformer_blocks.4.attn2.processor', #78
        # 'up_blocks.0.attentions.2.transformer_blocks.0.attn1.processor', #88
        # 'up_blocks.0.attentions.2.transformer_blocks.4.attn2.processor', #108
        # 'up_blocks.1.attentions.0.transformer_blocks.0.attn1.processor', #128
        # 'up_blocks.1.attentions.2.transformer_blocks.1.attn1.processor', #138

        'up_blocks.0.attentions.2.transformer_blocks.0.attn1.processor', #108
        'up_blocks.0.attentions.2.transformer_blocks.0.attn2.processor', 
        'up_blocks.0.attentions.2.transformer_blocks.1.attn1.processor', 
        'up_blocks.0.attentions.2.transformer_blocks.1.attn2.processor', 
        'up_blocks.0.attentions.2.transformer_blocks.2.attn1.processor', 
        'up_blocks.0.attentions.2.transformer_blocks.2.attn2.processor', 
        'up_blocks.0.attentions.2.transformer_blocks.3.attn1.processor', 
        'up_blocks.0.attentions.2.transformer_blocks.3.attn2.processor', 
        'up_blocks.0.attentions.2.transformer_blocks.4.attn1.processor', 
        'up_blocks.0.attentions.2.transformer_blocks.4.attn2.processor',
        'up_blocks.0.attentions.2.transformer_blocks.5.attn1.processor', 
        'up_blocks.0.attentions.2.transformer_blocks.5.attn2.processor', 
        'up_blocks.0.attentions.2.transformer_blocks.6.attn1.processor', 
        'up_blocks.0.attentions.2.transformer_blocks.6.attn2.processor',
        'up_blocks.0.attentions.2.transformer_blocks.7.attn1.processor',
        'up_blocks.0.attentions.2.transformer_blocks.7.attn2.processor',
        'up_blocks.0.attentions.2.transformer_blocks.8.attn1.processor', 
        'up_blocks.0.attentions.2.transformer_blocks.8.attn2.processor', 
        'up_blocks.0.attentions.2.transformer_blocks.9.attn1.processor', 
        'up_blocks.0.attentions.2.transformer_blocks.9.attn2.processor',

        'up_blocks.1.attentions.0.transformer_blocks.0.attn1.processor',  #128
        'up_blocks.1.attentions.0.transformer_blocks.0.attn2.processor',
        'up_blocks.1.attentions.0.transformer_blocks.1.attn1.processor',
        'up_blocks.1.attentions.0.transformer_blocks.1.attn2.processor',
        'up_blocks.1.attentions.1.transformer_blocks.0.attn1.processor', 
        'up_blocks.1.attentions.1.transformer_blocks.0.attn2.processor', 
        'up_blocks.1.attentions.1.transformer_blocks.1.attn1.processor', 
        'up_blocks.1.attentions.1.transformer_blocks.1.attn2.processor',
        'up_blocks.1.attentions.2.transformer_blocks.0.attn1.processor',
        'up_blocks.1.attentions.2.transformer_blocks.0.attn2.processor', 
        'up_blocks.1.attentions.2.transformer_blocks.1.attn1.processor', 
        'up_blocks.1.attentions.2.transformer_blocks.1.attn2.processor',
]

attn_map_save_steps = [20]
# attn_map_save_steps = [10,20,30,40]
results_dir = '/data/khshim/attention_map/munch'


device = 'cuda' if torch.cuda.is_available() else 'cpu'
if device == 'cpu':
    torch_dtype = torch.float32
else:
    torch_dtype = torch.float16

import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument('--img_path', type=str, default='./data/ref')
# parser.add_argument('--tar_obj', type=str, default='Birds and a train in the distance')
parser.add_argument('--tar_obj', type=str, default=None)
parser.add_argument('--guidance_scale', type=float, default=7.0)
parser.add_argument('--output_num', type=int, default=5)
parser.add_argument('--result_dir', type=str, default='results')
parser.add_argument('--activate_step', type=int, default=50)
parser.add_argument('--color_cal_start_t', type=int, default=150, help='start t for color calibration')
parser.add_argument('--color_cal_window_size', type=int, default=50, help='window size for color calibration')

args = parser.parse_args()
# Read the second row from the CSV file
csv_path = 'data/prompts/data_short.csv'
target_object_list = []

with open(csv_path, 'r') as csvfile:
    csvreader = csv.DictReader(csvfile)
    for row in csvreader:
        target_object_list.append(row['caption_predicted'])


# Update args with the new target object and result directory
# args.tar_obj = tar_obj

def create_number_list(n):
    return list(range(n + 1))

def create_nested_list(t):
    return [[0, t]]

def create_prompt(style_name):
    pre_prompt_dicts = {
        "chinese-ink-paint_A horse": ("{prompt} of watercolor style with soft strokes and fluid composition, Plain background", ""),
        "cloud_a Cloud in the sky": ("{prompt} of hyper-realistic style with soft lighting, Blue sky background", ""),
        "digital-art_A robot": ("{prompt} of cyberpunk style with neon colors and dark futuristic backdrop, Dark background", ""),
        "fire_fire": ("{prompt} of realistic style with dynamic flames and high contrast, Black background", ""),
        "klimt_the kiss": ("{prompt} of Art Nouveau style with golden hues and intricate patterns, Golden background", ""),
        "line-art_an owl": ("{prompt} of minimalist black and white line art style, Plain background", ""),
        "low-poly_A cat": ("{prompt} of low-poly style with geometric shapes and modern design, Plain background", ""),
        "munch_The scream": ("{prompt} of Expressionist style with bold strokes and surreal tones, Abstract background", ""),
        "totoro_totoro holding a tiny umbrella in the rain": ("{prompt} of Studio Ghibli anime style with whimsical details, Rainy background", ""),
        "van-gogh_The Starry Night": ("{prompt} of Post-Impressionist style with swirling strokes and vibrant colors, Starry background", "")
    }



    if style_name in pre_prompt_dicts.keys():
        return pre_prompt_dicts[style_name]
    else:
        return None, None


tar_seeds = create_number_list(args.output_num)
activate_step_indices = create_nested_list(args.activate_step)

img_path = args.img_path
tar_obj = args.tar_obj
guidance_scale = args.guidance_scale


if not os.path.exists(args.result_dir):
    os.makedirs(args.result_dir)
result_dir = args.result_dir


image_name_list = os.listdir(img_path)


device = 'cuda' if torch.cuda.is_available() else 'cpu'
if device == 'cpu':
    torch_dtype = torch.float32
else:
    torch_dtype = torch.float16
vae_model_path = "stabilityai/sd-vae-ft-mse"
from diffusers import DDIMScheduler, AutoencoderKL

vae = AutoencoderKL.from_pretrained(vae_model_path, torch_dtype=torch_dtype)

pipe = StableDiffusionXLPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch_dtype)
print(f'[PIPELINE] StableDiffusion on the track')
memory_efficient(vae, device)

memory_efficient(pipe, device)

# blip_processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
# blip_model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b", torch_dtype=torch_dtype).to(device)

pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)

pipe.scheduler.fix_traj_t_start = args.color_cal_start_t
pipe.scheduler.fix_traj_t_end = args.color_cal_start_t - args.color_cal_window_size

str_activate_layer, str_activate_step = pipe.activate_layer(
                        activate_layer_indices=[[0, 0], [128, 140]], 
                        attn_map_save_steps=[], 
                        activate_step_indices=activate_step_indices,
                        use_shared_attention=False,
)


for i in range(len(target_object_list)):
    tar_obj = target_object_list[i]
    print(f'\n[Progressing {i}] to {tar_obj}')
    
    args.result_dir = os.path.join('results', tar_obj.replace(' ', '_'))
    
    for seed in seeds:
        for activate_layer_indices in activate_layer_indices_list:
            attn_procs = {}
            activate_layers = []
            str_activate_layer = ""
            for activate_layer_index in activate_layer_indices:
                activate_layers += ACTIVATE_LAYER_CANDIDATE[activate_layer_index[0]:activate_layer_index[1]]
                str_activate_layer += str(activate_layer_index)

            for name in pipe.unet.attn_processors.keys():
                if name in activate_layers:
                    if name in save_layer_list:
                        attn_procs[name] = CrossFrameAttnProcessor_store(unet_chunk_size=2, attn_map_save_steps=attn_map_save_steps)
                    else:
                        attn_procs[name] = CrossFrameAttnProcessor(unet_chunk_size=2)
                else:
                    attn_procs[name] = AttnProcessor()
            pipe.unet.set_attn_processor(attn_procs)

            with torch.no_grad():
                for image_name in image_name_list:
                    image_path = os.path.join(img_path, image_name)

                    real_img = Image.open(image_path).resize((1024, 1024), resample=Image.BICUBIC)


                    style_name = image_name.split('.')[0]

                    latents = []

                    base_prompt, negative_prompt = create_prompt(style_name)
                    # print(base_prompt, negative_prompt)
                    if base_prompt is not None:
                        ref_prompt = base_prompt.replace("{prompt}", style_name)
                        inf_prompt = base_prompt.replace("{prompt}", tar_obj)

                    for tar_seed in tar_seeds:
                        latents.append(init_latent(model=pipe, device_name=device, dtype=torch_dtype, seed=tar_seed))
                    # print(ref_prompt, inf_prompt)   
                    
                    latents = torch.cat(latents, dim=0)
                    print(f"Style Name: {style_name}\nTarget Object: {tar_obj}\nBase Prompt: {base_prompt}\nNegative Prompt: {negative_prompt}\nReference Prompt: {ref_prompt}\nInference Prompt: {inf_prompt}\n")
                    images = pipe(
                        prompt=ref_prompt,
                        guidance_scale=guidance_scale,
                        latents=latents,
                        num_images_per_prompt=len(tar_seeds),
                        target_prompt=inf_prompt,
                        use_inf_negative_prompt=False,
                        use_advanced_sampling=False,
                        use_prompt_as_null=True,
                        image=real_img
                    )[0]
                    
                    
                    grid = create_image_grid(images, 1, num_images_per_prompt)
                    save_name = f"{tar_obj}_from_{style_name}.png"
                    save_path = os.path.join(results_dir, save_name)

                    grid.save(save_path)
                    print("Saved image to: ", save_path)

                    for attn_map_save_step in attn_map_save_steps:
                        attn_map_save_name = f"{tar_obj}_src_{style_name}.pt"
                        attn_map_dic = {}
                        for activate_layer in save_layer_list:
                            attn_map_var_name = transform_variable_name(activate_layer, attn_map_save_step)
                            exec(f"attn_map_dic[\"{activate_layer}\"] = {attn_map_var_name}")

                        torch.save(attn_map_dic, os.path.join(results_dir, attn_map_save_name))
                        print("Saved attn map to: ", os.path.join(results_dir, attn_map_save_name))

