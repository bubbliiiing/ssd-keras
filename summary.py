#--------------------------------------------#
#   该部分代码用于看网络结构
#--------------------------------------------#
from nets.ssd import SSD300

if __name__ == "__main__":
    input_shape = [300, 300, 3]
    num_classes = 21

    model = SSD300(input_shape, num_classes)
    model.summary()

    for i,layer in enumerate(model.layers):
        print(i,layer.name)
