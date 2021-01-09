import keras.backend as K
from keras.layers import (Activation, Conv2D, Dense, Flatten,
                          GlobalAveragePooling2D, Input, MaxPooling2D, Reshape,
                          ZeroPadding2D, concatenate, merge)
from keras.models import Model

from nets.ssd_layers import Normalize, PriorBox
from nets.VGG16 import VGG16


def SSD300(input_shape, num_classes=21, anchors_size=[30,60,111,162,213,264,315]):
    #---------------------------------#
    #   典型的输入大小为[300,300,3]
    #---------------------------------#
    input_tensor = Input(shape=input_shape)
    
    # net变量里面包含了整个SSD的结构，通过层名可以找到对应的特征层
    net = VGG16(input_tensor)

    #-----------------------将提取到的主干特征进行处理---------------------------#
    # 对conv4_3的通道进行l2标准化处理 
    # 38,38,512
    net['conv4_3_norm'] = Normalize(20, name='conv4_3_norm')(net['conv4_3'])
    num_priors = 4
    # 预测框的处理
    # num_priors表示每个网格点先验框的数量，4是x,y,h,w的调整
    net['conv4_3_norm_mbox_loc'] = Conv2D(num_priors * 4, kernel_size=(3,3), padding='same', name='conv4_3_norm_mbox_loc')(net['conv4_3_norm'])
    net['conv4_3_norm_mbox_loc_flat'] = Flatten(name='conv4_3_norm_mbox_loc_flat')(net['conv4_3_norm_mbox_loc'])
    # num_priors表示每个网格点先验框的数量，num_classes是所分的类
    net['conv4_3_norm_mbox_conf'] = Conv2D(num_priors * num_classes, kernel_size=(3,3), padding='same',name='conv4_3_norm_mbox_conf')(net['conv4_3_norm'])
    net['conv4_3_norm_mbox_conf_flat'] = Flatten(name='conv4_3_norm_mbox_conf_flat')(net['conv4_3_norm_mbox_conf'])

    priorbox = PriorBox(input_shape, anchors_size[0], max_size=anchors_size[1], aspect_ratios=[2],
                        variances=[0.1, 0.1, 0.2, 0.2],
                        name='conv4_3_norm_mbox_priorbox')
    net['conv4_3_norm_mbox_priorbox'] = priorbox(net['conv4_3_norm'])
    
    # 对fc7层进行处理 
    # 19,19,1024
    num_priors = 6
    # 预测框的处理
    # num_priors表示每个网格点先验框的数量，4是x,y,h,w的调整
    net['fc7_mbox_loc'] = Conv2D(num_priors * 4, kernel_size=(3,3),padding='same',name='fc7_mbox_loc')(net['fc7'])
    net['fc7_mbox_loc_flat'] = Flatten(name='fc7_mbox_loc_flat')(net['fc7_mbox_loc'])
    # num_priors表示每个网格点先验框的数量，num_classes是所分的类
    net['fc7_mbox_conf'] = Conv2D(num_priors * num_classes, kernel_size=(3,3),padding='same',name='fc7_mbox_conf')(net['fc7'])
    net['fc7_mbox_conf_flat'] = Flatten(name='fc7_mbox_conf_flat')(net['fc7_mbox_conf'])

    priorbox = PriorBox(input_shape, anchors_size[1], max_size=anchors_size[2], aspect_ratios=[2, 3],
                        variances=[0.1, 0.1, 0.2, 0.2],
                        name='fc7_mbox_priorbox')
    net['fc7_mbox_priorbox'] = priorbox(net['fc7'])

    # 对conv6_2进行处理
    # 10,10,512
    num_priors = 6
    # 预测框的处理
    # num_priors表示每个网格点先验框的数量，4是x,y,h,w的调整
    x = Conv2D(num_priors * 4, kernel_size=(3,3), padding='same',name='conv6_2_mbox_loc')(net['conv6_2'])
    net['conv6_2_mbox_loc'] = x
    net['conv6_2_mbox_loc_flat'] = Flatten(name='conv6_2_mbox_loc_flat')(net['conv6_2_mbox_loc'])
    # num_priors表示每个网格点先验框的数量，num_classes是所分的类
    x = Conv2D(num_priors * num_classes, kernel_size=(3,3), padding='same',name='conv6_2_mbox_conf')(net['conv6_2'])
    net['conv6_2_mbox_conf'] = x
    net['conv6_2_mbox_conf_flat'] = Flatten(name='conv6_2_mbox_conf_flat')(net['conv6_2_mbox_conf'])

    priorbox = PriorBox(input_shape, anchors_size[2], max_size=anchors_size[3], aspect_ratios=[2, 3],
                        variances=[0.1, 0.1, 0.2, 0.2],
                        name='conv6_2_mbox_priorbox')
    net['conv6_2_mbox_priorbox'] = priorbox(net['conv6_2'])

    # 对conv7_2进行处理
    # 5,5,256
    num_priors = 6
    # 预测框的处理
    # num_priors表示每个网格点先验框的数量，4是x,y,h,w的调整
    x = Conv2D(num_priors * 4, kernel_size=(3,3), padding='same',name='conv7_2_mbox_loc')(net['conv7_2'])
    net['conv7_2_mbox_loc'] = x
    net['conv7_2_mbox_loc_flat'] = Flatten(name='conv7_2_mbox_loc_flat')(net['conv7_2_mbox_loc'])
    # num_priors表示每个网格点先验框的数量，num_classes是所分的类
    x = Conv2D(num_priors * num_classes, kernel_size=(3,3), padding='same',name='conv7_2_mbox_conf')(net['conv7_2'])
    net['conv7_2_mbox_conf'] = x
    net['conv7_2_mbox_conf_flat'] = Flatten(name='conv7_2_mbox_conf_flat')(net['conv7_2_mbox_conf'])

    priorbox = PriorBox(input_shape, anchors_size[3], max_size=anchors_size[4], aspect_ratios=[2, 3],
                        variances=[0.1, 0.1, 0.2, 0.2],
                        name='conv7_2_mbox_priorbox')
    net['conv7_2_mbox_priorbox'] = priorbox(net['conv7_2'])

    # 对conv8_2进行处理
    # 3,3,256
    num_priors = 4
    # 预测框的处理
    # num_priors表示每个网格点先验框的数量，4是x,y,h,w的调整
    x = Conv2D(num_priors * 4, kernel_size=(3,3), padding='same',name='conv8_2_mbox_loc')(net['conv8_2'])
    net['conv8_2_mbox_loc'] = x
    net['conv8_2_mbox_loc_flat'] = Flatten(name='conv8_2_mbox_loc_flat')(net['conv8_2_mbox_loc'])
    # num_priors表示每个网格点先验框的数量，num_classes是所分的类
    x = Conv2D(num_priors * num_classes, kernel_size=(3,3), padding='same',name='conv8_2_mbox_conf')(net['conv8_2'])
    net['conv8_2_mbox_conf'] = x
    net['conv8_2_mbox_conf_flat'] = Flatten(name='conv8_2_mbox_conf_flat')(net['conv8_2_mbox_conf'])

    priorbox = PriorBox(input_shape, anchors_size[4], max_size=anchors_size[5], aspect_ratios=[2],
                        variances=[0.1, 0.1, 0.2, 0.2],
                        name='conv8_2_mbox_priorbox')
    net['conv8_2_mbox_priorbox'] = priorbox(net['conv8_2'])

    # 对conv9_2进行处理
    # 1,1,256
    num_priors = 4
    # 预测框的处理
    # num_priors表示每个网格点先验框的数量，4是x,y,h,w的调整
    x = Conv2D(num_priors * 4, kernel_size=(3,3), padding='same',name='conv9_2_mbox_loc')(net['conv9_2'])
    net['conv9_2_mbox_loc'] = x
    net['conv9_2_mbox_loc_flat'] = Flatten(name='conv9_2_mbox_loc_flat')(net['conv9_2_mbox_loc'])
    # num_priors表示每个网格点先验框的数量，num_classes是所分的类
    x = Conv2D(num_priors * num_classes, kernel_size=(3,3), padding='same',name='conv9_2_mbox_conf')(net['conv9_2'])
    net['conv9_2_mbox_conf'] = x
    net['conv9_2_mbox_conf_flat'] = Flatten(name='conv9_2_mbox_conf_flat')(net['conv9_2_mbox_conf'])
    
    priorbox = PriorBox(input_shape, anchors_size[5], max_size=anchors_size[6], aspect_ratios=[2],
                        variances=[0.1, 0.1, 0.2, 0.2],
                        name='conv9_2_mbox_priorbox')

    net['conv9_2_mbox_priorbox'] = priorbox(net['conv9_2'])

    # 将所有结果进行堆叠
    net['mbox_loc'] = concatenate([net['conv4_3_norm_mbox_loc_flat'],
                             net['fc7_mbox_loc_flat'],
                             net['conv6_2_mbox_loc_flat'],
                             net['conv7_2_mbox_loc_flat'],
                             net['conv8_2_mbox_loc_flat'],
                             net['conv9_2_mbox_loc_flat']],
                            axis=1, name='mbox_loc')
    net['mbox_conf'] = concatenate([net['conv4_3_norm_mbox_conf_flat'],
                              net['fc7_mbox_conf_flat'],
                              net['conv6_2_mbox_conf_flat'],
                              net['conv7_2_mbox_conf_flat'],
                              net['conv8_2_mbox_conf_flat'],
                              net['conv9_2_mbox_conf_flat']],
                             axis=1, name='mbox_conf')
    net['mbox_priorbox'] = concatenate([net['conv4_3_norm_mbox_priorbox'],
                                  net['fc7_mbox_priorbox'],
                                  net['conv6_2_mbox_priorbox'],
                                  net['conv7_2_mbox_priorbox'],
                                  net['conv8_2_mbox_priorbox'],
                                  net['conv9_2_mbox_priorbox']],
                                  axis=1, name='mbox_priorbox')

    # 8732,4
    net['mbox_loc'] = Reshape((-1, 4),name='mbox_loc_final')(net['mbox_loc'])
    # 8732,21
    net['mbox_conf'] = Reshape((-1, num_classes),name='mbox_conf_logits')(net['mbox_conf'])
    # 8732,8
    net['mbox_conf'] = Activation('softmax',name='mbox_conf_final')(net['mbox_conf'])
    # 8732,33
    net['predictions'] = concatenate([net['mbox_loc'],
                                    net['mbox_conf'],
                                    net['mbox_priorbox']],
                                    axis=2, name='predictions')

    model = Model(net['input'], net['predictions'])
    return model
