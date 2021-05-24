import numpy as np
import tensorflow as tf
from PIL import Image

def letterbox_image(image, size):
    iw, ih = image.size
    w, h = size
    scale = min(w/iw, h/ih)
    nw = int(iw*scale)
    nh = int(ih*scale)

    image = image.resize((nw,nh), Image.BICUBIC)
    new_image = Image.new('RGB', size, (128,128,128))
    new_image.paste(image, ((w-nw)//2, (h-nh)//2))

    return new_image

def ssd_correct_boxes(top, left, bottom, right, input_shape, image_shape):
    new_shape = image_shape*np.min(input_shape/image_shape)

    offset = (input_shape-new_shape)/2./input_shape
    scale = input_shape/new_shape

    box_yx = np.concatenate(((top+bottom)/2,(left+right)/2),axis=-1)
    box_hw = np.concatenate((bottom-top,right-left),axis=-1)

    box_yx = (box_yx - offset) * scale
    box_hw *= scale

    box_mins = box_yx - (box_hw / 2.)
    box_maxes = box_yx + (box_hw / 2.)
    boxes =  np.concatenate([
        box_mins[:, 0:1],
        box_mins[:, 1:2],
        box_maxes[:, 0:1],
        box_maxes[:, 1:2]
    ],axis=-1)
    boxes *= np.concatenate([image_shape, image_shape],axis=-1)
    return boxes

class BBoxUtility(object):
    def __init__(self, num_classes, priors=None, overlap_threshold=0.5,
                 nms_thresh=0.45, top_k=400):
        self.num_classes = num_classes
        self.priors = priors
        self.num_priors = 0 if priors is None else len(priors)
        self.overlap_threshold = overlap_threshold
        self._nms_thresh = nms_thresh
        self._top_k = top_k
        self.boxes = tf.placeholder(dtype='float32', shape=(None, 4))
        self.scores = tf.placeholder(dtype='float32', shape=(None,))
        self.nms = tf.image.non_max_suppression(self.boxes, self.scores, self._top_k, iou_threshold=self._nms_thresh)
        self.sess = tf.Session(config=tf.ConfigProto(device_count={'GPU': 0}))

    def iou(self, box):
        # 计算出每个真实框与所有的先验框的iou
        # 判断真实框与先验框的重合情况
        inter_upleft = np.maximum(self.priors[:, :2], box[:2])
        inter_botright = np.minimum(self.priors[:, 2:4], box[2:])

        inter_wh = inter_botright - inter_upleft
        inter_wh = np.maximum(inter_wh, 0)
        inter = inter_wh[:, 0] * inter_wh[:, 1]
        # 真实框的面积
        area_true = (box[2] - box[0]) * (box[3] - box[1])
        # 先验框的面积
        area_gt = (self.priors[:, 2] - self.priors[:, 0])*(self.priors[:, 3] - self.priors[:, 1])
        # 计算iou
        union = area_true + area_gt - inter

        iou = inter / union
        return iou

    def encode_box(self, box, return_iou=True):
        # 计算当前真实框和先验框的重合情况
        iou = self.iou(box)
        encoded_box = np.zeros((self.num_priors, 4 + return_iou))
        
        # 找到每一个真实框，重合程度较高的先验框
        assign_mask = iou > self.overlap_threshold

        # 如果没有一个先验框重合度大于self.overlap_threshold
        # 则选择重合度最大的为正样本
        if not assign_mask.any():
            assign_mask[iou.argmax()] = True
        
        # 利用iou进行赋值 
        if return_iou:
            encoded_box[:, -1][assign_mask] = iou[assign_mask]
        
        # 找到对应的先验框
        assigned_priors = self.priors[assign_mask]

        #---------------------------------------------#
        #   逆向编码，将真实框转化为ssd预测结果的格式
        #   先计算真实框的中心与长宽
        #---------------------------------------------#
        box_center  = 0.5 * (box[:2] + box[2:])
        box_wh      = box[2:] - box[:2]
        #---------------------------------------------#
        #   再计算重合度较高的先验框的中心与长宽
        #---------------------------------------------#
        assigned_priors_center = 0.5 * (assigned_priors[:, :2] +
                                        assigned_priors[:, 2:4])
        assigned_priors_wh = (assigned_priors[:, 2:4] -
                              assigned_priors[:, :2])
        
        #------------------------------------------------#
        #   逆向求取ssd应该有的预测结果
        #   先求取中心的预测结果，再求取宽高的预测结果
        #   存在改变数量级的参数，默认为[0.1,0.1,0.2,0.2]
        #------------------------------------------------#
        encoded_box[:, :2][assign_mask] = box_center - assigned_priors_center
        encoded_box[:, :2][assign_mask] /= assigned_priors_wh
        encoded_box[:, :2][assign_mask] /= assigned_priors[:, -4:-2]

        encoded_box[:, 2:4][assign_mask] = np.log(box_wh / assigned_priors_wh)
        encoded_box[:, 2:4][assign_mask] /= assigned_priors[:, -2:]
        return encoded_box.ravel()

    def assign_boxes(self, boxes):
        #---------------------------------------------------#
        #   assignment分为3个部分
        #   :4      的内容为网络应该有的回归预测结果
        #   4:-8    的内容为先验框所对应的种类，默认为背景
        #   -8      的内容为当前先验框是否包含目标
        #   -7:     无意义
        #---------------------------------------------------#
        assignment = np.zeros((self.num_priors, 4 + self.num_classes + 8))
        assignment[:, 4] = 1.0
        if len(boxes) == 0:
            return assignment

        # 对每一个真实框都进行iou计算
        encoded_boxes = np.apply_along_axis(self.encode_box, 1, boxes[:, :4])
        #---------------------------------------------------#
        #   在reshape后，获得的encoded_boxes的shape为：
        #   [num_true_box, num_priors, 4+1]
        #   4是编码后的结果，1为iou
        #---------------------------------------------------#
        encoded_boxes = encoded_boxes.reshape(-1, self.num_priors, 5)
        
        #---------------------------------------------------#
        #   [num_priors]求取每一个先验框重合度最大的真实框
        #---------------------------------------------------#
        best_iou = encoded_boxes[:, :, -1].max(axis=0)
        best_iou_idx = encoded_boxes[:, :, -1].argmax(axis=0)
        best_iou_mask = best_iou > 0
        best_iou_idx = best_iou_idx[best_iou_mask]
        
        #---------------------------------------------------#
        #   计算一共有多少先验框满足需求
        #---------------------------------------------------#
        assign_num = len(best_iou_idx)

        # 将编码后的真实框取出
        encoded_boxes = encoded_boxes[:, best_iou_mask, :]
        assignment[:, :4][best_iou_mask] = encoded_boxes[best_iou_idx,np.arange(assign_num),:4]
        #----------------------------------------------------------#
        #   4代表为背景的概率，设定为0，因为这些先验框有对应的物体
        #----------------------------------------------------------#
        assignment[:, 4][best_iou_mask] = 0
        assignment[:, 5:-8][best_iou_mask] = boxes[best_iou_idx, 4:]
        #----------------------------------------------------------#
        #   -8表示先验框是否有对应的物体
        #----------------------------------------------------------#
        assignment[:, -8][best_iou_mask] = 1
        # 通过assign_boxes我们就获得了，输入进来的这张图片，应该有的预测结果是什么样子的
        return assignment

    def decode_boxes(self, mbox_loc, mbox_priorbox, variances):
        # 获得先验框的宽与高
        prior_width = mbox_priorbox[:, 2] - mbox_priorbox[:, 0]
        prior_height = mbox_priorbox[:, 3] - mbox_priorbox[:, 1]
        # 获得先验框的中心点
        prior_center_x = 0.5 * (mbox_priorbox[:, 2] + mbox_priorbox[:, 0])
        prior_center_y = 0.5 * (mbox_priorbox[:, 3] + mbox_priorbox[:, 1])

        # 真实框距离先验框中心的xy轴偏移情况
        decode_bbox_center_x = mbox_loc[:, 0] * prior_width * variances[:, 0]
        decode_bbox_center_x += prior_center_x
        decode_bbox_center_y = mbox_loc[:, 1] * prior_height * variances[:, 1]
        decode_bbox_center_y += prior_center_y
        
        # 真实框的宽与高的求取
        decode_bbox_width = np.exp(mbox_loc[:, 2] * variances[:, 2])
        decode_bbox_width *= prior_width
        decode_bbox_height = np.exp(mbox_loc[:, 3] * variances[:, 3])
        decode_bbox_height *= prior_height

        # 获取真实框的左上角与右下角
        decode_bbox_xmin = decode_bbox_center_x - 0.5 * decode_bbox_width
        decode_bbox_ymin = decode_bbox_center_y - 0.5 * decode_bbox_height
        decode_bbox_xmax = decode_bbox_center_x + 0.5 * decode_bbox_width
        decode_bbox_ymax = decode_bbox_center_y + 0.5 * decode_bbox_height

        # 真实框的左上角与右下角进行堆叠
        decode_bbox = np.concatenate((decode_bbox_xmin[:, None],
                                      decode_bbox_ymin[:, None],
                                      decode_bbox_xmax[:, None],
                                      decode_bbox_ymax[:, None]), axis=-1)
        # 防止超出0与1
        decode_bbox = np.minimum(np.maximum(decode_bbox, 0.0), 1.0)
        return decode_bbox

    def detection_out(self, predictions, background_label_id=0, keep_top_k=200, confidence_threshold=0.5):
        #---------------------------------------------------#
        #   :4是回归预测结果
        #---------------------------------------------------#
        mbox_loc        = predictions[:, :, :4]
        #---------------------------------------------------#
        #   获得种类的置信度
        #---------------------------------------------------#
        mbox_conf       = predictions[:, :, 4:-8]
        #---------------------------------------------------#
        #   获得网络的先验框
        #---------------------------------------------------#
        mbox_priorbox   = predictions[:, :, -8:-4]
        #---------------------------------------------------#
        #   variances是一个改变数量级的参数。
        #   所有variances全去了也可以训练，有了效果更好。
        #---------------------------------------------------#
        variances       = predictions[:, :, -4:]

        results = []

        # 对每一张图片进行处理，由于在predict.py的时候，我们只输入一张图片，所以for i in range(len(mbox_loc))只进行一次
        for i in range(len(mbox_loc)):
            results.append([])
            #--------------------------------#
            #   利用回归结果对先验框进行解码
            #--------------------------------#
            decode_bbox = self.decode_boxes(mbox_loc[i], mbox_priorbox[i],  variances[i])

            for c in range(self.num_classes):
                if c == background_label_id:
                    continue
                #--------------------------------#
                #   取出属于该类的所有框的置信度
                #   判断是否大于门限
                #--------------------------------#
                c_confs     = mbox_conf[i, :, c]
                c_confs_m   = c_confs > confidence_threshold
                if len(c_confs[c_confs_m]) > 0:
                    # 取出得分高于confidence_threshold的框
                    boxes_to_process = decode_bbox[c_confs_m]
                    confs_to_process = c_confs[c_confs_m]
                    # 进行iou的非极大抑制
                    feed_dict   = {self.boxes: boxes_to_process,
                                    self.scores: confs_to_process}
                    idx         = self.sess.run(self.nms, feed_dict=feed_dict)
                    # 取出在非极大抑制中效果较好的内容
                    good_boxes  = boxes_to_process[idx]
                    confs       = confs_to_process[idx][:, None]
                    # 将label、置信度、框的位置进行堆叠。
                    labels      = c * np.ones((len(idx), 1))
                    c_pred      = np.concatenate((labels, confs, good_boxes), axis=1)
                    # 添加进result里
                    results[-1].extend(c_pred)

            if len(results[-1]) > 0:
                # 按照置信度进行排序
                results[-1] = np.array(results[-1])
                argsort     = np.argsort(results[-1][:, 1])[::-1]
                results[-1] = results[-1][argsort]
                # 选出置信度最大的keep_top_k个
                results[-1] = results[-1][:keep_top_k]
        return results
