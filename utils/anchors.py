import pickle

import matplotlib.pyplot as plt
import numpy as np


class PriorBox():
    def __init__(self, img_size, min_size, max_size=None, aspect_ratios=None,
                 flip=True, variances=[0.1], clip=True, **kwargs):

        self.waxis = 1
        self.haxis = 0

        self.img_size = img_size
        if min_size <= 0:
            raise Exception('min_size must be positive.')

        self.min_size = min_size
        self.max_size = max_size
        self.aspect_ratios = [1.0]
        if max_size:
            if max_size < min_size:
                raise Exception('max_size must be greater than min_size.')
            self.aspect_ratios.append(1.0)
        if aspect_ratios:
            for ar in aspect_ratios:
                if ar in self.aspect_ratios:
                    continue
                self.aspect_ratios.append(ar)
                if flip:
                    self.aspect_ratios.append(1.0 / ar)
        self.variances = np.array(variances)
        self.clip = True


    def call(self, input_shape, mask=None):
        # --------------------------------- #
        #   获取输入进来的特征层的宽和高
        #   比如38x38
        # --------------------------------- #
        layer_width = input_shape[self.waxis]
        layer_height = input_shape[self.haxis]

        # --------------------------------- #
        #   获取输入进来的图片的宽和高
        #   比如300x300
        # --------------------------------- #
        img_width = self.img_size[1]
        img_height = self.img_size[0]
        box_widths = []
        box_heights = []
        # --------------------------------- #
        #   self.aspect_ratios一般有两个值
        #   [1, 1, 2, 1/2]
        #   [1, 1, 2, 1/2, 3, 1/3]
        # --------------------------------- #
        for ar in self.aspect_ratios:
            # 首先添加一个较小的正方形
            if ar == 1 and len(box_widths) == 0:
                box_widths.append(self.min_size)
                box_heights.append(self.min_size)
            # 然后添加一个较大的正方形
            elif ar == 1 and len(box_widths) > 0:
                box_widths.append(np.sqrt(self.min_size * self.max_size))
                box_heights.append(np.sqrt(self.min_size * self.max_size))
            # 然后添加长方形
            elif ar != 1:
                box_widths.append(self.min_size * np.sqrt(ar))
                box_heights.append(self.min_size / np.sqrt(ar))

        # --------------------------------- #
        #   获得所有先验框的宽高1/2
        # --------------------------------- #
        box_widths = 0.5 * np.array(box_widths)
        box_heights = 0.5 * np.array(box_heights)

        # --------------------------------- #
        #   每一个特征层对应的步长
        # --------------------------------- #
        step_x = img_width / layer_width
        step_y = img_height / layer_height

        # --------------------------------- #
        #   生成网格中心
        # --------------------------------- #
        linx = np.linspace(0.5 * step_x, img_width - 0.5 * step_x,
                           layer_width)
        liny = np.linspace(0.5 * step_y, img_height - 0.5 * step_y,
                           layer_height)
        centers_x, centers_y = np.meshgrid(linx, liny)
        centers_x = centers_x.reshape(-1, 1)
        centers_y = centers_y.reshape(-1, 1)

        # 每一个先验框需要两个(centers_x, centers_y)，前一个用来计算左上角，后一个计算右下角
        num_priors_ = len(self.aspect_ratios)
        prior_boxes = np.concatenate((centers_x, centers_y), axis=1)
        prior_boxes = np.tile(prior_boxes, (1, 2 * num_priors_))
        
        # 获得先验框的左上角和右下角
        prior_boxes[:, ::4] -= box_widths
        prior_boxes[:, 1::4] -= box_heights
        prior_boxes[:, 2::4] += box_widths
        prior_boxes[:, 3::4] += box_heights

        # --------------------------------- #
        #   将先验框变成小数的形式
        #   归一化
        # --------------------------------- #
        prior_boxes[:, ::2] /= img_width
        prior_boxes[:, 1::2] /= img_height
        prior_boxes = prior_boxes.reshape(-1, 4)

        prior_boxes = np.minimum(np.maximum(prior_boxes, 0.0), 1.0)

        num_boxes = len(prior_boxes)
        
        if len(self.variances) == 1:
            variances = np.ones((num_boxes, 4)) * self.variances[0]
        elif len(self.variances) == 4:
            variances = np.tile(self.variances, (num_boxes, 1))
        else:
            raise Exception('Must provide one or four variances.')

        prior_boxes = np.concatenate((prior_boxes, variances), axis=1)
        return prior_boxes

def get_anchors(img_size = (300,300), anchors_size=[30,60,111,162,213,264,315]):
    if img_size == (512,512):
        features_map_length = [64,32,16,8,6,4]
    elif img_size == (300,300):
        features_map_length = [38,19,10,5,3,1]
    else:
        raise ValueError('Unsupported img_size - `{}`, Use (300,300), (512,512).'.format(img_size))

    net = {} 
    priorbox = PriorBox(img_size, anchors_size[0],max_size = anchors_size[1], aspect_ratios=[2],
                        variances=[0.1, 0.1, 0.2, 0.2],
                        name='conv4_3_norm_mbox_priorbox')
    net['conv4_3_norm_mbox_priorbox'] = priorbox.call([features_map_length[0],features_map_length[0]])


    priorbox = PriorBox(img_size, anchors_size[1], max_size=anchors_size[2], aspect_ratios=[2, 3],
                        variances=[0.1, 0.1, 0.2, 0.2],
                        name='fc7_mbox_priorbox')
    net['fc7_mbox_priorbox'] = priorbox.call([features_map_length[1],features_map_length[1]])


    priorbox = PriorBox(img_size, anchors_size[2], max_size=anchors_size[3], aspect_ratios=[2, 3],
                        variances=[0.1, 0.1, 0.2, 0.2],
                        name='conv6_2_mbox_priorbox')
    net['conv6_2_mbox_priorbox'] = priorbox.call([features_map_length[2],features_map_length[2]])


    priorbox = PriorBox(img_size, anchors_size[3], max_size=anchors_size[4], aspect_ratios=[2, 3],
                        variances=[0.1, 0.1, 0.2, 0.2],
                        name='conv7_2_mbox_priorbox')
    net['conv7_2_mbox_priorbox'] = priorbox.call([features_map_length[3],features_map_length[3]])


    priorbox = PriorBox(img_size, anchors_size[4], max_size=anchors_size[5], aspect_ratios=[2],
                        variances=[0.1, 0.1, 0.2, 0.2],
                        name='conv8_2_mbox_priorbox')
    net['conv8_2_mbox_priorbox'] = priorbox.call([features_map_length[4],features_map_length[4]])


    priorbox = PriorBox(img_size, anchors_size[5], max_size=anchors_size[6], aspect_ratios=[2],
                        variances=[0.1, 0.1, 0.2, 0.2],
                        name='pool6_mbox_priorbox')
                        

    net['pool6_mbox_priorbox'] = priorbox.call([features_map_length[5],features_map_length[5]])

    net['mbox_priorbox'] = np.concatenate([net['conv4_3_norm_mbox_priorbox'],
                                    net['fc7_mbox_priorbox'],
                                    net['conv6_2_mbox_priorbox'],
                                    net['conv7_2_mbox_priorbox'],
                                    net['conv8_2_mbox_priorbox'],
                                    net['pool6_mbox_priorbox']],
                                    axis=0)

    return net['mbox_priorbox']
