import os
import os.path as osp
from datetime import date

import mmcv
from mmengine.fileio import dump
from mmengine.utils import track_iter_progress

def convert_yolo_to_coco(image_dir, label_dir, out_file, categories):
    """
    image_dir: folder with .jpg/.jpeg/.png images
    label_dir: folder with .txt files (one per image, same basename)
    out_file:  output path for COCO JSON
    categories: list of dicts, e.g.
        [
          {'id':0, 'name':'Explosives',           'supercategory':''},
          {'id':1, 'name':'Anti-personnel mine',  'supercategory':''},
          {'id':2, 'name':'Anti-vehicle mine',    'supercategory':''},
        ]
    """
    images = []
    annotations = []
    ann_id = 0

    # gather all image files
    img_files = sorted(
        [f for f in os.listdir(image_dir)
         if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    )

    for img_id, img_name in enumerate(track_iter_progress(img_files)):
        img_path = osp.join(image_dir, img_name)
        # read to get size
        h, w = mmcv.imread(img_path).shape[:2]

        images.append({
            'id': img_id,
            'file_name': img_name,
            'height': h,
            'width': w
        })

        # corresponding label file
        lbl_name = osp.splitext(img_name)[0] + '.txt'
        lbl_path = osp.join(label_dir, lbl_name)
        if not osp.exists(lbl_path):
            continue

        with open(lbl_path, 'r') as f:
            lines = [ln.strip() for ln in f if ln.strip()]

        for ln in lines:
            parts = ln.split()
            if len(parts) != 5:
                continue
            cls_id = int(parts[0])
            x_c_n, y_c_n, w_n, h_n = map(float, parts[1:])

            # convert normalized to pixels
            x_c = x_c_n * w
            y_c = y_c_n * h
            bbox_w = w_n * w
            bbox_h = h_n * h

            x_min = max(0.0, x_c - bbox_w / 2)
            y_min = max(0.0, y_c - bbox_h / 2)
            # clip to image bounds
            bbox_w = min(bbox_w, w - x_min)
            bbox_h = min(bbox_h, h - y_min)

            ann = {
                'id': ann_id,
                'image_id': img_id,
                'category_id': cls_id,
                'bbox': [x_min, y_min, bbox_w, bbox_h],
                'area': bbox_w * bbox_h,
                'segmentation': [
                    [
                        x_min,         y_min,
                        x_min + bbox_w, y_min,
                        x_min + bbox_w, y_min + bbox_h,
                        x_min,         y_min + bbox_h
                    ]
                ],
                'iscrowd': 0
            }
            annotations.append(ann)
            ann_id += 1

    coco_dict = {
        'info': {
            'description': 'Fold_0 YOLO → COCO',
            'version': '1.0',
            'year': date.today().year,
            'date_created': date.today().isoformat()
        },
        'licenses': [],
        'images': images,
        'annotations': annotations,
        'categories': categories
    }

    dump(coco_dict, out_file)
    print(f'Wrote {len(images)} images and {len(annotations)} annotations to {out_file}')

if __name__ == '__main__':
    # adjust this root if needed
    data_root = 'data/temp/fold_4'

    # your three classes
    categories = [
        {'id': 0, 'name': 'Explosives',           'supercategory': ''},
        {'id': 1, 'name': 'Anti-personnel mine',  'supercategory': ''},
        {'id': 2, 'name': 'Anti-vehicle mine',    'supercategory': ''},
    ]

    # train split
    convert_yolo_to_coco(
        image_dir = osp.join(data_root, 'images/train'),
        label_dir = osp.join(data_root, 'labels/train'),
        out_file   = osp.join(data_root, 'coco_train.json'),
        categories = categories
    )

    # val split
    convert_yolo_to_coco(
        image_dir = osp.join(data_root, 'images/val'),
        label_dir = osp.join(data_root, 'labels/val'),
        out_file   = osp.join(data_root, 'coco_val.json'),
        categories = categories
    )
