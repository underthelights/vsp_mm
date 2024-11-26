import torch
import os, sys
sys.path.append('/home/khshim/workspace/sd/vsp')
from pipelines.inverted_ve_pipeline import CrossFrameAttnProcessor, CrossFrameAttnProcessor_store, ACTIVATE_LAYER_CANDIDATE
from diffusers import DDIMScheduler, AutoencoderKL
import os
from PIL import Image
from utilsv2 import memory_efficient
from diffusers.models.attention_processor import AttnProcessor
from pipeline_stable_diffusion_xl_attn import StableDiffusionXLPipeline
import sys


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

# results_dir = 'saved_attention_map_results'
results_dir = '/data/khshim/attention_map'
if not os.path.exists(results_dir):
    os.makedirs(results_dir)

base_model_path = "runwayml/stable-diffusion-v1-5"
vae_model_path = "stabilityai/sd-vae-ft-mse"
image_encoder_path = "models/image_encoder/"


object_list = [
   "horse",
#    "woman",
#    "dog",
#    "horse",
#    "motorcycle"
]

target_object_list = [
    # "Null",
    "Birds and a train in the distance",
    # "clock",
    # "car"
    # "panda",
    # "bridge",
    # "flower"
]

prompt_neg_prompt_pair_dicts = {
    "ref_chinese-ink-paint_A horse": ("{prompt} of watercolor style with soft strokes and fluid composition, Plain background", ""),
    # "ref_cloud_a Cloud in the sky": ("{prompt} of hyper-realistic style with soft lighting, Blue sky background", ""),
    # "ref_digital-art_A robot": ("{prompt} of cyberpunk style with neon colors and dark futuristic backdrop, Dark background", ""),
    # "ref_fire_fire": ("{prompt} of realistic style with dynamic flames and high contrast, Black background", ""),
    # "ref_klimt_the kiss": ("{prompt} of Art Nouveau style with golden hues and intricate patterns, Golden background", ""),
    # "ref_line-art_an owl": ("{prompt} of minimalist black and white line art style, Plain background", ""),
    # "ref_low-poly_A cat": ("{prompt} of low-poly style with geometric shapes and modern design, Plain background", ""),
    # "ref_munch_The scream": ("{prompt} of Expressionist style with bold strokes and surreal tones, Abstract background", ""),
    # "ref_totoro_totoro holding a tiny umbrella in the rain": ("{prompt} of Studio Ghibli anime style with whimsical details, Rainy background", ""),
    # "ref_van-gogh_The Starry Night": ("{prompt} of Post-Impressionist style with swirling strokes and vibrant colors, Starry background", "")
}



noise_scheduler = DDIMScheduler(
    num_train_timesteps=1000,
    beta_start=0.00085,
    beta_end=0.012,
    beta_schedule="scaled_linear",
    clip_sample=False,
    set_alpha_to_one=False,
    steps_offset=1,
)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
if device == 'cpu':
    torch_dtype = torch.float32
else:
    torch_dtype = torch.float16

vae = AutoencoderKL.from_pretrained(vae_model_path, torch_dtype=torch_dtype)
pipe = StableDiffusionXLPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch_dtype)


memory_efficient(vae, device)
memory_efficient(pipe, device)

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
                    print(f"layer:{name}")
                    attn_procs[name] = CrossFrameAttnProcessor_store(unet_chunk_size=2, attn_map_save_steps=attn_map_save_steps)
                else:
                    print(f"layer:{name}")
                    attn_procs[name] = CrossFrameAttnProcessor(unet_chunk_size=2)
            else :
                attn_procs[name] = AttnProcessor()
        pipe.unet.set_attn_processor(attn_procs)


        for target_object in target_object_list:
            target_prompt = f"A photo of a {target_object}"

            for object in object_list:
                for key in prompt_neg_prompt_pair_dicts.keys():
                    prompt, negative_prompt = prompt_neg_prompt_pair_dicts[key]

                    generator = torch.Generator(device).manual_seed(seed) if seed is not None else None

                    images = pipe(
                        prompt=prompt.replace("{prompt}", object),
                        guidance_scale = 7.0,
                        num_images_per_prompt = num_images_per_prompt,
                        target_prompt = target_prompt,
                        generator=generator,

                    )[0]


                    #make grid
                    grid = create_image_grid(images, 1, num_images_per_prompt)

                    save_name = f"{key}_src_{object}_tgt_{target_object}_activate_layer_{str_activate_layer}_seed_{seed}.png"
                    save_path = os.path.join(results_dir, save_name)

                    grid.save(save_path)

                    print("Saved image to: ", save_path)

                    #save attn map
                    for attn_map_save_step in attn_map_save_steps:
                        attn_map_save_name = f"attn_map_raw_{key}_src_{object}_tgt_{target_object}_activate_layer_{str_activate_layer}_attn_map_step_{attn_map_save_step}_seed_{seed}.pt"
                        attn_map_dic = {}
                        # for activate_layer in activate_layers:
                        for activate_layer in save_layer_list:
                            attn_map_var_name = transform_variable_name(activate_layer, attn_map_save_step)
                            exec(f"attn_map_dic[\"{activate_layer}\"] = {attn_map_var_name}")

                        torch.save(attn_map_dic, os.path.join(results_dir, attn_map_save_name))
                        print("Saved attn map to: ", os.path.join(results_dir, attn_map_save_name))


