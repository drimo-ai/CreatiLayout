import torch
import os 
from utils.bbox_visualization import bbox_visualization,scale_boxes
from PIL import Image
from src.models.transformer_sd3_SiamLayout import SiamLayoutSD3Transformer2DModel
from src.pipeline.pipeline_sd3_CreatiLayout import CreatiLayoutSD3Pipeline

if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model_path = "stabilityai/stable-diffusion-3-medium-diffusers"
    ckpt_path = "HuiZhang0812/CreatiLayout"
    transformer_additional_kwargs = dict(attention_type="layout",strict=True)
    transformer = SiamLayoutSD3Transformer2DModel.from_pretrained(
         ckpt_path, subfolder="SiamLayout_SD3", torch_dtype=torch.float16,**transformer_additional_kwargs)
    pipe = CreatiLayoutSD3Pipeline.from_pretrained(model_path, transformer=transformer, torch_dtype=torch.float16)
    pipe = pipe.to(device)

    seed = 42
    batch_size = 1
    num_inference_steps = 50
    guidance_scale = 7.5
    height = 1024
    width = 1024

    save_root = "output"
    img_save_root = os.path.join(save_root,"images")
    os.makedirs(img_save_root,exist_ok=True)
    img_with_layout_save_root = os.path.join(save_root,"images_with_layout")
    os.makedirs(img_with_layout_save_root,exist_ok=True)


    global_caption = ["A picturesque scene features Spider Man standing confidently on a rugged rock by the sea, holding a drawing board with his hands. The board displays the words 'Creative Layout' in a playful, hand-drawn font. The serene sea shimmers under the setting sun. The sky is painted with a gradient of warm colors, from deep oranges to soft purples."]
    region_caption_list = [
        "Spider Man standing confidently on a rugged rock.",
        "A rugged rock by the sea.",
        "A drawing board with the words 'Creative Layout' in a playful, hand-drawn font.",
        "The serene sea shimmers under the setting sun.",
        "The sky is a shade of deep orange to soft purple."
    ]
    region_bboxes_list = [
        [0.40, 0.35, 0.55, 0.80],  
        [0.35, 0.75, 0.60, 0.95],  
        [0.40, 0.45, 0.55, 0.65], 
        [0.00, 0.30, 1.00, 0.90],  
        [0.00, 0.00, 1.00, 0.30]
    ]
    filename = "Spider Man"

    with torch.no_grad():
        images = pipe(prompt = global_caption*batch_size,
                    generator = torch.Generator(device=device).manual_seed(seed),
                    num_inference_steps = num_inference_steps,
                    guidance_scale = guidance_scale,
                    bbox_phrases = region_caption_list, 
                    bbox_raw = region_bboxes_list,
                    height = height,
                    width = width
                )
    images=images.images
    
    for j, image in enumerate(images):   

        image.save(os.path.join(img_save_root,f"{filename}_{j}.png")) 

        img_with_layout_save_name=os.path.join(img_with_layout_save_root,f"{filename}_{j}.png")

        white_image = Image.new('RGB', (width, height), color='rgb(256,256,256)')
        show_input = {"boxes":scale_boxes(region_bboxes_list,width,height),"labels":region_caption_list}

        bbox_visualization_img = bbox_visualization(white_image,show_input)
        image_with_bbox = bbox_visualization(image ,show_input)

        total_width = width*2
        total_height = height

        new_image = Image.new('RGB', (total_width, total_height))
        new_image.paste(bbox_visualization_img, (0, 0))
        new_image.paste(image_with_bbox, (width, 0))
        new_image.save(img_with_layout_save_name)

    
    