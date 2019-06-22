import keras
from keras.models import load_model
from skimage import io
from skimage import transform
from PIL import Image
import numpy as np
from keras.preprocessing.image import img_to_array
import os
import pickle
import cv2
import numpy as np
import matplotlib.pyplot as plt
import keras.backend as K

MODEL_PATH = './models'


def init():
    print('Loading model ...')
    model = load_model(os.path.join(MODEL_PATH, 'original_model.hdf5'))
    model.summary()
    print('Model loaded')

    print('Loading label dictionary ...')
    with open('./models/label_dict.hdf5', 'rb') as dt:
        labels_dict = pickle.load(dt)
    print(labels_dict)
    return model, labels_dict

def preprocess_image(image, target_size=(144,192)):
	'''
	Preprocess initial image, rescale and expand dim
	----------------------------------
	image
	target_size: size of image, i.e (w,h)
	----------------------------------
	return:
		preprocessed image
	'''
	image = transform.resize(image, (144,192))
	# image = image.resize(target_size)
	# image = img_to_array(image)
	# Rescale
	# if np.max(image) > 1:
	# 	image /= 255
	image = np.expand_dims(image, axis=0)
	return image

def predict(model, labels_dict, image):
    # global model, labels_dict

    image = preprocess_image(image)
    preds = model.predict(image, steps=1, verbose=1)
	# Get predictions
    predictions = np.argmax(preds, axis=1)
	# Get probabilities
    probs = np.max(preds, axis=1)
   
    predict_labels = [labels_dict[k] for k in predictions]
    return predict_labels, probs

def resize_image(image, target_shape=(144,192)):
    return transform.resize(image, target_shape)

def visualize_cam(model, image, last_conv_layer_index=-5, learning_phase=0, show=False, path_to_save=None):
    '''
    visualize class activation map function
    ----------------------------------------
    arguments:
    - model: CNN model with average pooling layer
    - image: image to apply cam
    - last_conv_layer_index: index of last convolution layer
    - learning_phase: integer, if 1 then mode='learing', else if 0 then mode='testing'
    - show: boolean, True then show image using matplotlib
    - path_to_save: path to save the image
    -----------------------------------------
    return:
        predictions: softmax vector
    '''

    '''Get weights of dense output layer'''
    class_weights = model.layers[-3].get_weights()[0]

    '''Create the function to get last conv layer output and model output'''
    last_conv_layer = model.layers[last_conv_layer_index].output
    get_output = K.function([model.input, K.learning_phase()], [last_conv_layer, model.output])

    img = np.array([np.transpose(np.float32(image), (0, 1, 2))])

    [conv_outputs, predictions] = get_output([img, learning_phase])
    conv_outputs = conv_outputs[0,:,:,:]

    '''Create the class activation map'''
    class_num = np.argmax(predictions)
    cam = np.zeros(dtype=np.float32, shape=conv_outputs.shape[:2])

    for i,w in enumerate(class_weights[:,class_num]):
        cam += w*conv_outputs[:,:,i]
    cam /= np.max(cam)
    cam = cv2.resize(cam, (image.shape[1], image.shape[0]))
    heatmap = cv2.applyColorMap(np.uint8(255*cam), cv2.COLORMAP_JET)
    heatmap[np.where(cam < 0.2)] = 0
    in_heatmap = 255 - heatmap
    
    # plt.imshow(image)
    plt.imshow(in_heatmap, alpha=0.6)
    plt.axis('off')
    if not path_to_save == None:
        plt.savefig(path_to_save, dpi=100)
    if show == True:
        plt.show()
    
    #   return predictions
    return in_heatmap



# if __name__ == '__main__':
#     image = io.imread("./upload/30_AE_A4_08_94_64_38569_5a0dbe1aa925d.png")
#     model, labels_dict = init()
#     true_label, prob = predict(model, labels_dict, image)
#     print(true_label, prob)
