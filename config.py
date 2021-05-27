import os


class InfoConfig:
    username = None
    date = None


class ActiveConfig(InfoConfig):
    img_w = 800
    img_h = 600

    def __init__(self):
        super().__init__()
        self.root_path = 'E:/active_sample'

        self.seg_path = None
        self.function_seg = None

        self.video_path = None
        self.pred_path = None

        self.vis_path = None
        self.img_path = None

    def setup_info(self, username, date):
        self.username = username
        self.date = date

        # segmentation images
        self.seg_path = os.path.join(
            self.root_path, 'seg', self.username + '_seg.png')
        self.function_seg = os.path.join(
            self.root_path, 'seg', self.username + '_fun_seg.png')

        # input video and prediction
        self.video_path = os.path.join(
            self.root_path, self.username, self.date)
        self.pred_path = os.path.join(
            self.root_path, self.username, self.date + '_prediction')

        # output image and visualization
        self.vis_path = os.path.join(
            self.root_path, 'active_sample_output', self.username, self.date, 'visualization')
        self.img_path = os.path.join(
            self.root_path, 'active_sample_output', self.username, self.date, 'images')
