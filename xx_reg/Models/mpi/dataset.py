import numpy as np
import pandas as pd
import tensorflow as tf
from tfrecord_lite import decode_example
from torch.utils.data import Dataset,DataLoader
import os,sys
from tqdm import tqdm
sys.path.append("/data/lxg/workplace/xx_reg/")
# print(sys.path)
# is_train:'train','test','check'

def parse_npy(dd, key):
    dd = dd['features']['feature'][key]
    # ddd = dd[list(dd.keys())[0]]['value']
    return dd

# class xxDataset(Dataset):
#     def __init__(self, opt, mode='train'):
#         self.opt = opt
#         self.data_list = get_list(self.opt)
#         # print(len(self.data_list))
#         self.keywords = ['year', 'NIGHTLIGHTS', 'lat', 'tmmn', 'BLUE', 'GREEN', 'SWIR1', 'SWIR2', 'tmmx', 'NIR', 'pr',
#                          'lon', 'TEMP1', 'LON', 'RED', 'country', 'MPI_Easy4','LAT', 'DHSCLUST1', 'MPI3',
#                          'Number of Households2', 'system:index']
#         self.mode = mode
#         self.modes =  ["train","valid","test"]
#         self.mode_index = self.modes.index(self.mode)
#     def __len__(self):
#         return len(self.data_list[self.mode_index])

#     def __getitem__(self, idx):
#         # crt_data = tf.data.TFRecordDataset(self.data_list[idx]).numpy()
#         data = np.load(self.data_list[self.mode_index][idx], allow_pickle=True).item()
#         landsat = np.stack([
#             parse_npy(data, 'BLUE').reshape([255,255]),
#             parse_npy(data, 'GREEN').reshape([255,255]),
#             parse_npy(data, 'SWIR1').reshape([255, 255]),
#             parse_npy(data, 'SWIR2').reshape([255, 255]),
#             parse_npy(data, 'RED').reshape([255, 255]),
#             parse_npy(data, 'NIR').reshape([255, 255]),
#         ])
#         nl =  np.stack([parse_npy(data, 'NIGHTLIGHTS').reshape([255,255]),])
#         tmp = np.stack([
#             parse_npy(data, 'tmmn').reshape([255,255]),
#             parse_npy(data, 'tmmx').reshape([255,255]),
#             parse_npy(data, 'pr').reshape([255, 255]),
#         ])
#         other = np.stack([
#             parse_npy(data, 'Number of Households2'),
#         ])

#         # print(parse_npy(data, 'MPI_Easy4'))
#         return tuple([landsat,tmp,nl,other]), np.where(parse_npy(data, 'MPI3')==0,0.0001,parse_npy(data, 'MPI3'))

class xxDataset(Dataset):
    def __init__(self, opt, mode='train'):
        self.opt = opt
        self.data_list = [self.opt.train_list, self.opt.valid_list, self.opt.test_list]
        # print(len(self.data_list))
        self.keywords = ['year', 'NIGHTLIGHTS', 'lat', 'tmmn', 'BLUE', 'GREEN', 'SWIR1', 'SWIR2', 'tmmx', 'NIR', 'pr',
                         'lon', 'TEMP1', 'LON', 'RED', 'country', 'MPI_Easy4','LAT', 'DHSCLUST1', 'MPI3',
                         'Number of Households2', 'system:index']
        self.mode = mode
        self.modes =  ["train","valid","test"]
        self.mode_index = self.modes.index(self.mode)
    def __len__(self):
        return len(self.data_list[self.mode_index])

    def __getitem__(self, idx):
        # crt_data = tf.data.TFRecordDataset(self.data_list[idx]).numpy()
        data = np.load(self.data_list[self.mode_index][idx], allow_pickle=True).item()
        landsat = np.stack([
            parse_npy(data, 'BLUE'),
            parse_npy(data, 'GREEN'),
            parse_npy(data, 'SWIR1'),
            parse_npy(data, 'SWIR2'),
            parse_npy(data, 'RED'),
            parse_npy(data, 'NIR'),
        ])
        nl =  np.stack([parse_npy(data, 'NIGHTLIGHTS'),])
        tmp = np.stack([
            parse_npy(data, 'tmmn'),
            parse_npy(data, 'tmmx'),
            parse_npy(data, 'pr'),
        ])
        other = np.stack([
            parse_npy(data, 'Number of Households2'),
        ])
        geo = np.stack([
            parse_npy(data, 'lon'),
            parse_npy(data, 'lat'),
        ]).squeeze()
        # print(parse_npy(data, 'MPI_Easy4'))
        return tuple([landsat,tmp,nl,other]), np.where(parse_npy(data, 'MPI3')==0,0.0001,parse_npy(data, 'MPI3')), geo



# 不管输入如何，输出为三个list，表示训练、验证、测试的地址

if __name__=='__main__':
    # from Options import Opt
    # opt = Opt()
    # ds = xxDataset(opt)
    # loader = DataLoader(
    #             dataset=ds,
    #             batch_size=opt.batch_size,
    #             shuffle=True,
    #             num_workers=opt.num_workers,
    #             pin_memory=False,
    #             drop_last=False
    #     )
    # print(len(ds))
    # for batch_ndx, sample in enumerate(loader):
    #     print(batch_ndx,sample)

    # for raw_record in tf.data.TFRecordDataset(
    #         "/data/lxg/workplace/dhs_tfrecords/nigeria_2018_27_3_fold2.tfrecord.gz",
    #         compression_type='GZIP').take(1):
    #     example = tf.train.Example()
    #     example.ParseFromString(raw_record.numpy())
    #     print(example)
    # import json
    # from google.protobuf.json_format import MessageToJson
    # it = tf.python_io.tf_record_iterator(path)
    # decode_example(next(it))

    # keyword = None
    # dataset = tf.data.TFRecordDataset("/data/lxg/workplace/dhs_tfrecords/nigeria_2018_27_3_fold2.tfrecord.gz",compression_type='GZIP')
    # for rec in dataset.take(1):
    #     ex = tf.train.Example()
    #     ex.ParseFromString(rec.numpy())
    #     m = json.loads(MessageToJson(ex))
    #     keywords = m['features']['feature'].keys()
    #     print(keywords)
    #     for i in keywords:
    #         fea = m['features']['feature']
    #         print(len(fea[i]['floatList']['value']))


    '''
    从tfrecord转npy
    '''
    from Options import Opt
    opt = Opt()
    import json
    from google.protobuf.json_format import MessageToJson
    keywords = ['year', 'NIGHTLIGHTS', 'lat', 'tmmn', 'BLUE', 'GREEN', 'SWIR1', 'SWIR2',
               'tmmx', 'NIR', 'pr', 'lon', 'TEMP1', 'LON', 'RED', 'country', 'MPI_Easy4',
               'LAT', 'DHSCLUST1', 'MPI3', 'Number of Households2', 'system:index']

    remove_keywords = ['year', 'LAT','LON','country', 'DHSCLUST1', 'MPI_Easy4','system:index']

    select_keywords = ['year','NIGHTLIGHTS', 'lat', 'tmmn', 'BLUE', 'GREEN', 'SWIR1', 'SWIR2',
               'tmmx', 'NIR', 'pr', 'lon', 'RED',  'MPI3', 'Number of Households2',]


    data_dir = "/data/lxg/workplace/dhs_tfrecords/"
    save_dir = "/data/lxg/workplace/dhs_npy"

    for path in tqdm( os.listdir(data_dir)):
        if "tfrecord.gz" not in path:
            continue
        dataset = tf.data.TFRecordDataset(os.path.join(data_dir,path),
                                          compression_type='GZIP')
        for rec in dataset.take(1):
            ex = tf.train.Example()
            ex.ParseFromString(rec.numpy())
            m = json.loads(MessageToJson(ex))
            m['features']['feature'] = {key: val for key, val in m['features']['feature'].items() if key in select_keywords}
            for key in select_keywords:
                dd = m['features']['feature'][key]
                ddd = dd[list(dd.keys())[0]]['value']
                # print(key, len(ddd))
                if len(ddd)==65025:
                    m['features']['feature'][key] = np.array(ddd).reshape(255,255)
                else:
                    m['features']['feature'][key]  = np.array(ddd)
                # print(key, m['features']['feature'][key].shape)
            # m = m['features']['feature'][select_keywords]
            # print(m)
            # print(m['features']['feature']['MPI3'])
            np.save(os.path.join(save_dir,path+".npy"),m)
