from ultralytics import YOLO
import ultralytics
from datetime import datetime
import os
import yaml

cfg = ultralytics.cfg.get_cfg(cfg='config.yaml')

cfg_dict = vars(cfg)
cfg_dict['project'] = cfg.project
cfg_dict['name'] = cfg.name

with open('temp_config.yaml', 'w') as f:
    yaml.dump(cfg_dict, f, allow_unicode=True)

model = YOLO(cfg.model)
results = model.train(cfg='temp_config.yaml')