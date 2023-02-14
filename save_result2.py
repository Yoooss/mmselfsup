import os
 
from mmdet.apis import init_detector, inference_detector
 
config_file = './configs/benchmarks/mmdetection/voc0712/faster_rcnn_swin_fpn_voc0712ls_iter copy.py'
checkpoint_file = './work_dirs/selfsup/densecl_resnet50_8xb32-coslr-200e_in1k_swin_pred_grid_num/20221213_161335_grid1/epoch_200.pth'
device = 'cuda:0'
model = init_detector(config_file, checkpoint_file, device=device)
imgPath = '/media/ls/disk1/NWPU VHR-10 dataset 3/VOCdevkit/VOC2007/JPEGImages'
imgList = os.listdir(imgPath)
outPath ='/media/ls/disk1/NWPU VHR-10 dataset 3/result'
if not os.path.exists(outPath):
    os.mkdir(outPath)
for img in imgList:
    image = os.path.join(imgPath,img)
    model.show_result(
        image,
        inference_detector(model, image),
        score_thr=0.3,
        show=False,
        wait_time=0,
        win_name="result",
        bbox_color=(255, 0, 0),
        text_color=(200, 200, 200),
        mask_color=None,
        out_file=os.path.join(outPath, img))