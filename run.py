from util import create_folder
from config import ActiveConfig
from activesample import ActiveSample

import os


username = 'user00'
date = '2020-12-13'
config = ActiveConfig()
config.setup_info(username, date)

active_sample = ActiveSample(config)
create_folder(config.img_path, config.vis_path)
for i in range(0, 24):
    active_sample.run(os.path.join(config.video_path, str(i).zfill(2)), os.path.join(config.pred_path, str(i).zfill(2)))

