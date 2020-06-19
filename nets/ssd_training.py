import tensorflow as tf
from random import shuffle
import numpy as np
from PIL import Image
from keras.applications.imagenet_utils import preprocess_input
from keras import backend as K
from matplotlib.colors import rgb_to_hsv, hsv_to_rgb

class MultiboxLoss(object):
    def __init__(self, num_classes, alpha=1.0, neg_pos_ratio=3.0,
                 background_label_id=0, negatives_for_hard=100.0):
        self.num_classes = num_classes
        self.alpha = alpha
        self.neg_pos_ratio = neg_pos_ratio
        if background_label_id != 0:
            raise Exception('Only 0 as background label id is supported')
        self.background_label_id = background_label_id
        self.negatives_for_hard = negatives_for_hard

    def _l1_smooth_loss(self, y_true, y_pred):
        abs_loss = tf.abs(y_true - y_pred)
        sq_loss = 0.5 * (y_true - y_pred)**2
        l1_loss = tf.where(tf.less(abs_loss, 1.0), sq_loss, abs_loss - 0.5)
        return tf.reduce_sum(l1_loss, -1)

    def _softmax_loss(self, y_true, y_pred):
        y_pred = tf.maximum(y_pred, 1e-7)
        softmax_loss = -tf.reduce_sum(y_true * tf.log(y_pred),
                                      axis=-1)
        return softmax_loss

    def compute_loss(self, y_true, y_pred):
        batch_size = tf.shape(y_true)[0]
        num_boxes = tf.to_float(tf.shape(y_true)[1])

        # 计算所有的loss
        # 分类的loss
        # batch_size,8732,21 -> batch_size,8732
        conf_loss = self._softmax_loss(y_true[:, :, 4:-8],
                                       y_pred[:, :, 4:-8])
        # 框的位置的loss
        # batch_size,8732,4 -> batch_size,8732
        loc_loss = self._l1_smooth_loss(y_true[:, :, :4],
                                        y_pred[:, :, :4])

        # 获取所有的正标签的loss
        # 每一张图的pos的个数
        num_pos = tf.reduce_sum(y_true[:, :, -8], axis=-1)
        # 每一张图的pos_loc_loss
        pos_loc_loss = tf.reduce_sum(loc_loss * y_true[:, :, -8],
                                     axis=1)
        # 每一张图的pos_conf_loss
        pos_conf_loss = tf.reduce_sum(conf_loss * y_true[:, :, -8],
                                      axis=1)

        # 获取一定的负样本
        num_neg = tf.minimum(self.neg_pos_ratio * num_pos,
                             num_boxes - num_pos)

        # 找到了哪些值是大于0的
        pos_num_neg_mask = tf.greater(num_neg, 0)
        # 获得一个1.0
        has_min = tf.to_float(tf.reduce_any(pos_num_neg_mask))
        num_neg = tf.concat( axis=0,values=[num_neg,
                                [(1 - has_min) * self.negatives_for_hard]])
        # 求平均每个图片要取多少个负样本
        num_neg_batch = tf.reduce_mean(tf.boolean_mask(num_neg,
                                                      tf.greater(num_neg, 0)))
        num_neg_batch = tf.to_int32(num_neg_batch)

        # conf的起始
        confs_start = 4 + self.background_label_id + 1
        # conf的结束
        confs_end = confs_start + self.num_classes - 1

        # 找到实际上在该位置不应该有预测结果的框，求他们最大的置信度。
        max_confs = tf.reduce_max(y_pred[:, :, confs_start:confs_end],
                                  axis=2)

        # 取top_k个置信度，作为负样本
        _, indices = tf.nn.top_k(max_confs * (1 - y_true[:, :, -8]),
                                 k=num_neg_batch)

        # 找到其在1维上的索引
        batch_idx = tf.expand_dims(tf.range(0, batch_size), 1)
        batch_idx = tf.tile(batch_idx, (1, num_neg_batch))
        full_indices = (tf.reshape(batch_idx, [-1]) * tf.to_int32(num_boxes) +
                        tf.reshape(indices, [-1]))
        
        # full_indices = tf.concat(2, [tf.expand_dims(batch_idx, 2),
        #                              tf.expand_dims(indices, 2)])
        # neg_conf_loss = tf.gather_nd(conf_loss, full_indices)
        neg_conf_loss = tf.gather(tf.reshape(conf_loss, [-1]),
                                  full_indices)
        neg_conf_loss = tf.reshape(neg_conf_loss,
                                   [batch_size, num_neg_batch])
        neg_conf_loss = tf.reduce_sum(neg_conf_loss, axis=1)

        # loss is sum of positives and negatives
        
        num_pos = tf.where(tf.not_equal(num_pos, 0), num_pos,
                            tf.ones_like(num_pos))
        total_loss = tf.reduce_sum(pos_conf_loss) + tf.reduce_sum(neg_conf_loss)
        total_loss /= tf.reduce_sum(num_pos)
        total_loss += tf.reduce_sum(self.alpha * pos_loc_loss) / tf.reduce_sum(num_pos)

        return total_loss

def rand(a=0, b=1):
    return np.random.rand()*(b-a) + a


class Generator(object):
    def __init__(self, bbox_util,batch_size,
                 train_lines, val_lines, image_size,num_classes,
                 ):
        self.bbox_util = bbox_util
        self.batch_size = batch_size
        self.train_lines = train_lines
        self.val_lines = val_lines
        self.train_batches = len(train_lines)
        self.val_batches = len(val_lines)
        self.image_size = image_size
        self.num_classes = num_classes-1
        
    def get_random_data(self, annotation_line, input_shape, jitter=.1, hue=.1, sat=1.1, val=1.1):
        '''r实时数据增强的随机预处理'''
        line = annotation_line.split()
        image = Image.open(line[0])
        iw, ih = image.size
        h, w = input_shape
        box = np.array([np.array(list(map(int,box.split(',')))) for box in line[1:]])

        # resize image
        new_ar = w/h * rand(1-jitter,1+jitter)/rand(1-jitter,1+jitter)
        scale = rand(.25, 2)
        if new_ar < 1:
            nh = int(scale*h)
            nw = int(nh*new_ar)
        else:
            nw = int(scale*w)
            nh = int(nw/new_ar)
        image = image.resize((nw,nh), Image.BICUBIC)

        # place image
        dx = int(rand(0, w-nw))
        dy = int(rand(0, h-nh))
        new_image = Image.new('RGB', (w,h), (128,128,128))
        new_image.paste(image, (dx, dy))
        image = new_image

        # flip image or not
        flip = rand()<.5
        if flip: image = image.transpose(Image.FLIP_LEFT_RIGHT)

        # distort image
        hue = rand(-hue, hue)
        sat = rand(1, sat) if rand()<.5 else 1/rand(1, sat)
        val = rand(1, val) if rand()<.5 else 1/rand(1, val)
        x = rgb_to_hsv(np.array(image)/255.)
        x[..., 0] += hue
        x[..., 0][x[..., 0]>1] -= 1
        x[..., 0][x[..., 0]<0] += 1
        x[..., 1] *= sat
        x[..., 2] *= val
        x[x>1] = 1
        x[x<0] = 0
        image_data = hsv_to_rgb(x)*255 # numpy array, 0 to 1

        # correct boxes
        box_data = np.zeros((len(box),5))
        if len(box)>0:
            np.random.shuffle(box)
            box[:, [0,2]] = box[:, [0,2]]*nw/iw + dx
            box[:, [1,3]] = box[:, [1,3]]*nh/ih + dy
            if flip: box[:, [0,2]] = w - box[:, [2,0]]
            box[:, 0:2][box[:, 0:2]<0] = 0
            box[:, 2][box[:, 2]>w] = w
            box[:, 3][box[:, 3]>h] = h
            box_w = box[:, 2] - box[:, 0]
            box_h = box[:, 3] - box[:, 1]
            box = box[np.logical_and(box_w>1, box_h>1)] # discard invalid box
            box_data = np.zeros((len(box),5))
            box_data[:len(box)] = box
        if len(box) == 0:
            return image_data, []

        if (box_data[:,:4]>0).any():
            return image_data, box_data
        else:
            return image_data, []


    def generate(self, train=True):
        while True:
            if train:
                # 打乱
                shuffle(self.train_lines)
                lines = self.train_lines
            else:
                shuffle(self.val_lines)
                lines = self.val_lines
            inputs = []
            targets = []
            for annotation_line in lines:  
                img,y=self.get_random_data(annotation_line,self.image_size[0:2])
                if len(y)!=0:
                    boxes = np.array(y[:,:4],dtype=np.float32)
                    boxes[:,0] = boxes[:,0]/self.image_size[1]
                    boxes[:,1] = boxes[:,1]/self.image_size[0]
                    boxes[:,2] = boxes[:,2]/self.image_size[1]
                    boxes[:,3] = boxes[:,3]/self.image_size[0]
                    one_hot_label = np.eye(self.num_classes)[np.array(y[:,4],np.int32)]
                    if ((boxes[:,3]-boxes[:,1])<=0).any() and ((boxes[:,2]-boxes[:,0])<=0).any():
                        continue
                    
                    y = np.concatenate([boxes,one_hot_label],axis=-1)

                y = self.bbox_util.assign_boxes(y)
                inputs.append(img)               
                targets.append(y)
                if len(targets) == self.batch_size:
                    tmp_inp = np.array(inputs)
                    tmp_targets = np.array(targets)
                    inputs = []
                    targets = []
                    yield preprocess_input(tmp_inp), tmp_targets

