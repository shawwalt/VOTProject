from lib.train.admin.environment import env_settings


class Settings:
    """ Training settings, e.g. the paths to datasets and networks."""
    def __init__(self):
        self.set_default()

    def set_default(self):
        self.env = env_settings() # 数据集路径等设置
        self.use_gpu = True


