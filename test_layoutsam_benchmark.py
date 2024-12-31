
import torch
import os 
from torch.utils.data import DataLoader
from tqdm import tqdm
from IPython.core.debugger import set_trace 
from utils.bbox_visualization import bbox_visualization,scale_boxes
from PIL import Image
from src.models.transformer_sd3_SiamLayout import SiamLayoutSD3Transformer2DModel
from src.pipeline.pipeline_CreatiLayout import CreatiLayoutSD3Pipeline
from dataset.layoutsam_benchmark import BboxDataset
from datasets import load_dataset

if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model_path = "stabilityai/stable-diffusion-3-medium-diffusers"
    ckpt_path = "HuiZhang0812/CreatiLayout"
    dataset_path = "HuiZhang0812/LayoutSAM-eval"
    transformer_additional_kwargs = dict(attention_type="layout",strict=True)
    transformer = SiamLayoutSD3Transformer2DModel.from_pretrained(
         ckpt_path, subfolder="transformer", torch_dtype=torch.float16,**transformer_additional_kwargs)
    pipe = CreatiLayoutSD3Pipeline.from_pretrained(model_path, transformer=transformer, torch_dtype=torch.float16)
    pipe = pipe.to("cuda")

    test_dataset = load_dataset(dataset_path, split='test')
    test_dataset = BboxDataset(test_dataset)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=1)

    seed = 42
    batch_size = 1
    num_inference_steps = 50
    height = 1024
    width = 1024

    save_root = "output/layoutSAM-eval"
    img_save_root = os.path.join(save_root,"images")
    os.makedirs(img_save_root,exist_ok=True)
    img_with_layout_save_root = os.path.join(save_root,"images_with_layout")
    os.makedirs(img_with_layout_save_root,exist_ok=True)

    #generation

    for i, batch in enumerate(tqdm(test_dataloader)):
        global_caption = batch["global_caption"]
        region_caption_list = [t[0] for t in batch["detail_region_caption_list"]]
        region_bboxes_list = batch["region_bboxes_list"][0]
        filename = batch["file_name"][0]
        with torch.no_grad():
            images = pipe(prompt = global_caption*batch_size,
                        generator = torch.Generator(device="cuda").manual_seed(seed),
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

        bbox_visualization_img = bbox_visualization(white_image,show_input)
        image_with_bbox = bbox_visualization(image ,show_input)

        total_width = width*2
        total_height = height

        new_image = Image.new('RGB', (total_width, total_height))
        new_image.paste(bbox_visualization_img, (0, 0))
        new_image.paste(image_with_bbox, (width, 0))
        new_image.save(img_with_layout_save_name)


