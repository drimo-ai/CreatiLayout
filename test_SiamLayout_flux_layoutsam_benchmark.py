
import torch
import os 
from torch.utils.data import DataLoader
from tqdm import tqdm
from IPython.core.debugger import set_trace 
from utils.bbox_visualization import bbox_visualization,scale_boxes
from PIL import Image
from src.models.transformer_flux_SiamLayout import FluxTransformer2DModel
from src.pipeline.pipeline_flux_CreatiLayout import CreatiLayoutFluxPipeline
from dataset.layoutsam_benchmark import BboxDataset
from datasets import load_dataset
'''
if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model_path = r"D:\qinglong\Stable-diffusion\flux\flux1-dev"
    ckpt_path = r"E:\CreatiLayout\CreatiLayout"
    dataset_path = "HuiZhang0812/LayoutSAM-eval"
    transformer_additional_kwargs = dict(
        attention_type="layout",
        double_blocks_index=[i for i in range(0,19,1)],  
        single_blocks_index=[i for i in range(0,38,1)],  
        is_add=True,
        max_boxes_token_length=30,
        fix_bbox_ids=True,
        strict=True
        )
    transformer = FluxTransformer2DModel.from_pretrained(
         ckpt_path, subfolder="SiamLayout_FLUX",torch_dtype=torch.float16,**transformer_additional_kwargs)
    pipe = CreatiLayoutFluxPipeline.from_pretrained(model_path, transformer=transformer, torch_dtype=torch.bfloat16)
    pipe = pipe.to(device)

    test_dataset = load_dataset(dataset_path, split='test')
    test_dataset = BboxDataset(test_dataset)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=1)

    seed = 42
    batch_size = 1
    num_inference_steps = 50
    height = 512
    width = 512
    save_root = "output/layoutSAM-eval-SiamLayout-FLUX"
    img_save_root = os.path.join(save_root,"images")
    os.makedirs(img_save_root,exist_ok=True)
    img_with_layout_save_root = os.path.join(save_root,"images_with_layout")
    os.makedirs(img_with_layout_save_root,exist_ok=True)

    #generation

    for i, batch in enumerate(tqdm(test_dataloader)):
        global_caption = batch["global_caption"]
        region_caption_list = [t[0] for t in batch["detail_region_caption_list"]]
        region_bboxes_list = batch["region_bboxes_list"][0]
        print("region_bboxes_list",region_bboxes_list)
        filename = batch["file_name"][0]
        with torch.no_grad():
            images = pipe(prompt = global_caption*batch_size,
                        generator = torch.Generator(device=device).manual_seed(seed),
                        num_inference_steps = num_inference_steps,
                        bbox_phrases = region_caption_list, 
                        bbox_raw = region_bboxes_list,
                        height = height,
                        width = width
                    )
        image=images.images[0]

        image.save(os.path.join(img_save_root,filename)) 

        img_with_layout_save_name=os.path.join(img_with_layout_save_root,filename)

        white_image = Image.new('RGB', (width, height), color='rgb(256,256,256)')
        show_input = {"boxes":scale_boxes(region_bboxes_list,width,height),"labels":region_caption_list}

        bbox_visualization_img = bbox_visualization(white_image,show_input,font_size=15)
        image_with_bbox = bbox_visualization(image ,show_input,font_size=15)

        total_width = width*2
        total_height = height

        new_image = Image.new('RGB', (total_width, total_height))
        new_image.paste(bbox_visualization_img, (0, 0))
        new_image.paste(image_with_bbox, (width, 0))
        new_image.save(img_with_layout_save_name)
'''
def adjust_and_normalize_bboxes(bboxes, orig_width, orig_height):
    normalized_bboxes = []
    for bbox in bboxes:
        x1, y1, x2, y2 = bbox
        x1_norm = round(x1 / orig_width, 3)
        y1_norm = round(y1 / orig_height, 3)
        x2_norm = round(x2 / orig_width, 3)
        y2_norm = round(y2 / orig_height, 3)
        normalized_bboxes.append([x1_norm, y1_norm, x2_norm, y2_norm])

    return normalized_bboxes

import ast
import numpy as np

if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #model_path = r"D:\qinglong\Stable-diffusion\flux\flux1-dev"
    #ckpt_path = r"E:\CreatiLayout\CreatiLayout"
    model_path = r"/root/autodl-tmp/flux1-dev"
    ckpt_path = r"/root/autodl-tmp/LayoutSAM-models"
    dataset_path = "/root/autodl-tmp/LayoutSAM-eval/data"
    #dataset_path = "HuiZhang0812/LayoutSAM-eval"
    transformer_additional_kwargs = dict(
        attention_type="layout",
        double_blocks_index=[i for i in range(0, 19, 1)],
        single_blocks_index=[i for i in range(0, 38, 1)],
        is_add=True,
        max_boxes_token_length=30,
        fix_bbox_ids=True,
        strict=True
    )
    transformer = FluxTransformer2DModel.from_pretrained(
         ckpt_path, subfolder="SiamLayout_FLUX",torch_dtype=torch.bfloat16,**transformer_additional_kwargs)
    pipe = CreatiLayoutFluxPipeline.from_pretrained(model_path, transformer=transformer, torch_dtype=torch.bfloat16)
    pipe = pipe.to(device)

    # load pulid
    pipe.load_pulid_models()
    pipe.load_pretrain()

    seed = 123
    batch_size = 1
    num_inference_steps = 50
    height = 512
    width = 512
    save_root = "output/layoutSAM-eval-SiamLayout-FLUX"
    img_save_root = os.path.join(save_root, "images")
    os.makedirs(img_save_root, exist_ok=True)
    img_with_layout_save_root = os.path.join(save_root, "images_with_layout")
    os.makedirs(img_with_layout_save_root, exist_ok=True)

    region_caption_list = ["A man in a suit sitting at the table, with a plate of food and wine glass in front of him","A man in a suit sitting at the table, with a plate of food and wine glass in front of him"]

    global_caption = "In an elegant dining room, two men are having dinner at opposite ends of a long formal table, with warm lighting creating an atmospheric ambiance"
    mask_xyxy = [[40, 100, 228, 488],[350,100,500,500]]
    region_bboxes_list = []
    regional_masks=[]
    for mask in mask_xyxy:
       region_bboxes_list.append([float(n) for n in mask])#ast.literal_eval([mask_xyxy])
       x1, y1, x2, y2 = mask
       idmask = torch.zeros((height, width))
       idmask[y1:y2, x1:x2] = 1.0
       regional_masks.append(idmask)
       
    region_bboxes_list = adjust_and_normalize_bboxes(region_bboxes_list, width, height)
    region_bboxes_list = np.array(region_bboxes_list, dtype=np.float32)
    #region_bboxes_list=[]
    #region_bboxes_list.append(mask_xyxy)
    



    id_image_paths = ["./assets/musk.jpg","./assets/trump.jpg"]
    id_weights = [0.0,0.0]

    joint_attention_kwargs = {
        'id_image_paths': id_image_paths,
        'id_weights': id_weights,
        'id_masks': regional_masks[:len(id_image_paths)],  # use foreground mask as id mask
    }
    with torch.no_grad():
        images = pipe(prompt=global_caption * batch_size,
                      generator=torch.Generator(device=device).manual_seed(seed),
                      num_inference_steps=num_inference_steps,
                      bbox_phrases=region_caption_list,
                      bbox_raw=region_bboxes_list,
                      height=height,
                     width=width,
                      joint_attention_kwargs=joint_attention_kwargs
                      )
    image = images.images[0]
    filename = "test_pulid_creatilayout.png"
    image.save(os.path.join(img_save_root, filename))

    img_with_layout_save_name = os.path.join(img_with_layout_save_root, filename)

    white_image = Image.new('RGB', (width, height), color='rgb(256,256,256)')
    show_input = {"boxes": scale_boxes(region_bboxes_list, width, height), "labels": region_caption_list}

    bbox_visualization_img = bbox_visualization(white_image, show_input, font_size=15)
    image_with_bbox = bbox_visualization(image, show_input, font_size=15)

    total_width = width * 2
    total_height = height

    new_image = Image.new('RGB', (total_width, total_height))
    new_image.paste(bbox_visualization_img, (0, 0))
    new_image.paste(image_with_bbox, (width, 0))
    new_image.save(img_with_layout_save_name)

'''
    global_caption = "In a classroom during the afternoon, a man is practicing guitar by himself, with sunlight beautifully illuminating the room"
    region_caption_list = ["A man in a blue shirt and jeans, playing guitar"]
    mask_xyxy = [64, 20, 448, 512]
    region_bboxes_list = []
    region_bboxes_list.append([float(n) for n in mask_xyxy])#ast.literal_eval([mask_xyxy])
    region_bboxes_list = adjust_and_normalize_bboxes(region_bboxes_list, width, height)
    region_bboxes_list = np.array(region_bboxes_list, dtype=np.float32)
    #region_bboxes_list=[]
    #region_bboxes_list.append(mask_xyxy)
    
    background_mask = torch.ones((height, width))
    x1, y1, x2, y2 = mask_xyxy
    mask = torch.zeros((height, width))
    mask[y1:y2, x1:x2] = 1.0
    background_mask -= mask

    regional_masks=[]
    regional_masks.append(mask)


    id_image_paths = ["./assets/musk.jpg"]
    id_weights = [0.8]''' 

