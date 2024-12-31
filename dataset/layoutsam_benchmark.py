from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import ast
import numpy as np
def adjust_and_normalize_bboxes(bboxes, orig_width, orig_height):
    normalized_bboxes = []
    for bbox in bboxes:
        x1, y1, x2, y2 = bbox
        x1_norm = round(x1 / orig_width,3)  
        y1_norm = round(y1 / orig_height,3)
        x2_norm = round(x2 / orig_width,3)
        y2_norm = round(y2 / orig_height,3)
        normalized_bboxes.append([x1_norm, y1_norm, x2_norm, y2_norm])
    
    return normalized_bboxes
class BboxDataset(Dataset):
    def __init__(self, dataset, resolution=1024):
        self.dataset = dataset
        self.resolution = resolution
        self.transform = transforms.Compose([
            transforms.Resize(
                (resolution,resolution), interpolation=transforms.InterpolationMode.BILINEAR 
            ),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        image = item['image']
        image = self.transform(image)
        height = int(item['height'])
        width = int(item['width'])
        global_caption = item['global_caption']
        region_bboxes_list = item['bbox_list']
        detail_region_caption_list = item['detail_region_captions']
        region_caption_list = item['region_captions']
        file_name = item['file_name']

        region_bboxes_list = ast.literal_eval(region_bboxes_list)
        region_bboxes_list = adjust_and_normalize_bboxes(region_bboxes_list,width,height)
        region_bboxes_list = np.array(region_bboxes_list, dtype=np.float32)
    
        region_caption_list = ast.literal_eval(region_caption_list)
        
        return {
            'image': image,
            'global_caption': global_caption,
            'detail_region_caption_list': detail_region_caption_list,
            'region_bboxes_list': region_bboxes_list,
            'region_caption_list': region_caption_list,
            'file_name': file_name,
            'height': height,
            'width': width
        }

