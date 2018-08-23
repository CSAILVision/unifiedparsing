# defines the configurations for this model
import os


class Config:

    # data paths config of net dissect dataset pipline. 

    # TODO(LYC):: replace this path with complete data set. 
    netdissect_data_root = "/home/yingchengliu/code/NetDissect/dataset_toy"
    ade_dir = os.path.join(netdissect_data_root, "ade20k")
    dtd_dir = os.path.join(netdissect_data_root, "dtd", "dtd-r1.0.1")
    os_dir = os.path.join(netdissect_data_root, "opensurfaces")
    pascal_dir = os.path.join(netdissect_data_root, "pascal")

    # broden_dir = os.path.join(netdissect_data_root, "broden_tmp")
    merged_data_info_dir = "/home/yingchengliu/code/NetDissect"

    # ade20k scene name label to places scene label.
    scene_ade2places_path = '/unsullied/sharefs/xtt/nfs-large/ade2places_revised.json'

    # adechallenger file index to scene label file.
    scene_label_path = '/home/yingchengliu/code/NetDissect/broden_lite/sceneCategories.txt'


config = Config()
