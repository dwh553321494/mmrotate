import os

# configs = [
#     'data/GZBY_X/train_instance10/logs/quadri_yolox_s_iter_5000/quadri_yolox_s_dota.py',
#     'data/GZBY_X/train_instance8/logs/quadri_yolox_s_iter_5000/quadri_yolox_s_dota.py',
#     'data/GZBY_X/train_instance6/logs/quadri_yolox_s_iter_5000/quadri_yolox_s_dota.py',
#     'data/GZBY_X/train_instance4/logs/quadri_yolox_s_iter_5000/quadri_yolox_s_dota.py',
#     'data/GZBY_X/train_instance2/logs/quadri_yolox_s_iter_5000/quadri_yolox_s_dota.py',
#     'data/GZBY_X/train_instance1/logs/quadri_yolox_s_iter_5000/quadri_yolox_s_dota.py',
# ]

# checkpoints =[
#     'data/GZBY_X/train_instance10/logs/quadri_yolox_s_iter_5000/best_dota_mAP_iter_3000.pth',
#     'data/GZBY_X/train_instance8/logs/quadri_yolox_s_iter_5000/best_dota_mAP_iter_1500.pth',
#     'data/GZBY_X/train_instance6/logs/quadri_yolox_s_iter_5000/best_dota_mAP_iter_3500.pth',
#     'data/GZBY_X/train_instance4/logs/quadri_yolox_s_iter_5000/best_dota_mAP_iter_4000.pth',
#     'data/GZBY_X/train_instance2/logs/quadri_yolox_s_iter_5000/best_dota_mAP_iter_500.pth',
#     'data/GZBY_X/train_instance1/logs/quadri_yolox_s_iter_5000/best_dota_mAP_iter_3000.pth',
# ]

configs = [
    'data/GZBY_X/train_instance10/logs/quadri_yolox_m_iter_5000/quadri_yolox_m_dota.py',
    'data/GZBY_X/train_instance10/logs/quadri_yolox_l_iter_5000/quadri_yolox_l_dota.py',
    'data/GZBY_X/train_instance10/logs/quadri_yolox_resnet18_iter_5000/quadri_yolox_resnet18_dota.py',
    'data/GZBY_X/train_instance10/logs/quadri_yolox_swim_tiny_22k_iter_5000/quadri_yolox_swim_tiny_dota.py',
]
checkpoints = [
    'data/GZBY_X/train_instance10/logs/quadri_yolox_m_iter_5000/best_dota_mAP_iter_3500.pth',
    'data/GZBY_X/train_instance10/logs/quadri_yolox_l_iter_5000/best_dota_mAP_iter_3000.pth',
    'data/GZBY_X/train_instance10/logs/quadri_yolox_resnet18_iter_5000/best_dota_mAP_iter_1000.pth',
    'data/GZBY_X/train_instance10/logs/quadri_yolox_swim_tiny_22k_iter_5000/best_dota_mAP_iter_3500.pth',
    ]

work_dir = 'data/GZBY_X/test_instance1200/logs/'

for config, checkpoint in zip(configs, checkpoints):
    print(config, checkpoint)
    test_name = os.path.basename(config).split('.')[0]
    work = os.path.join(work_dir, test_name)
    os.system(f'python tools/test.py --config {config} --checkpoint {checkpoint} --work-dir {work_dir} --out {test_name}.pkl --show-dir {work}')
