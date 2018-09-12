import numpy as np 
import sys
caffe_root = "/usr/src/app/ENet/caffe-enet/"
sys.path.insert(0, caffe_root + 'python')
import caffe
import cv2

image_path = "/usr/src/app/ENet/example_image/munich_000000_000019_leftImg8bit.png"
model = "/usr/src/app/ENet/prototxts/enet_deploy_final.prototxt"
weights = "/usr/src/app/cityscapes_weights.caffemodel"
color_map = "/usr/src/app/ENet/scripts/cityscapes19.png"
gpu_flag = '1'


# Caffe set mode
if gpu_flag == 0:
        caffe.set_mode_gpu()
else:
        caffe.set_mode_cpu()
        
# Load network
net = caffe.Net(model, weights, caffe.TEST)

input_shape = net.blobs['data'].data.shape
output_shape = net.blobs['deconv6_0_0'].data.shape


# Load and pre-proecess image
input_image = cv2.imread(image_path, 1).astype(np.float32)
label_colours = cv2.imread(color_map, 1).astype(np.uint8)
input_image = cv2.resize(input_image, (input_shape[3], input_shape[2]))
input_image = input_image.transpose((2, 0, 1))
input_image = np.asarray([input_image])

# Forward pass
net.forward_all(**{net.inputs[0]: input_image})

# Get the output of ENet
prediction = net.blobs['deconv6_0_0'].data[0].argmax(axis=0)
prediction = np.squeeze(prediction)
prediction = np.resize(prediction, (3, input_shape[2], input_shape[3]))
prediction = prediction.transpose(1, 2, 0).astype(np.uint8)
prediction_rgb = np.zeros(prediction.shape, dtype=np.uint8)

# Colouring the result
label_colours_bgr = label_colours[..., ::-1]
cv2.LUT(prediction, label_colours_bgr, prediction_rgb)