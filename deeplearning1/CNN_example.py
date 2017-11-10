'''
A brief introduction into finetuning deep CNNs using MNIST or Cifar??.
Will conver data preprocessing, data augmentation, finetuning and regularization
'''

from keras.datasets import cifar10
import keras as ks
import numpy as np
import os
import matplotlib.pyplot as plt


cifar10_lookup = {1: 'airplane', 2: 'automobile', 3: 'bird', 4: 'cat',
                  5: 'deer', 6: 'dog', 7: 'frog', 8: 'horse', 9: 'ship',
                  10: 'truck'}


def createNetwork(input_shape, num_classes, print_summary=True):
    model = ks.models.Sequential()
    model.add(ks.layers.Conv2D(32, (3, 3), padding='same',
                               input_shape=input_shape))
    model.add(ks.layers.Activation('relu'))
    model.add(ks.layers.Conv2D(32, (3, 3)))
    model.add(ks.layers.Activation('relu'))
    model.add(ks.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(ks.layers.Dropout(0.25))

    model.add(ks.layers.Conv2D(64, (3, 3), padding='same'))
    model.add(ks.layers.Activation('relu'))
    model.add(ks.layers.Conv2D(64, (3, 3)))
    model.add(ks.layers.Activation('relu'))
    model.add(ks.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(ks.layers.Dropout(0.25))

    model.add(ks.layers.Flatten())
    model.add(ks.layers.Dense(512))
    model.add(ks.layers.Activation('relu'))
    model.add(ks.layers.Dropout(0.5))
    model.add(ks.layers.Dense(num_classes))
    model.add(ks.layers.Activation('softmax'))

    if print_summary:
        print '\nCreated network: '
        print model.summary()
        print '\n'

    return model


def disp_data(img_arr, lbl_arr):
    '''
    Display a subplot with a few of the train and/or testing images.
    Let's limit this to 4 images + labels
    '''
    cifar10_lookup = {1: 'airplane', 2: 'automobile', 3: 'bird', 4: 'cat',
                      5: 'deer', 6: 'dog', 7: 'frog', 8: 'horse', 9: 'ship',
                      10: 'truck'}
    assert img_arr.shape[0] <= 4, 'Too many images. {} given'.format(img_arr)
    assert img_arr.shape[0] == lbl_arr.shape[0], (
        'mismatch between number of images and labels given')
    lblstr = []
    for lbl in range(lbl_arr.shape[0]):
        if len(lbl_arr[lbl, ...]) > 1:
            idx = np.where(lbl_arr[lbl, :] == 1)[0][0]
            lblstr.append(cifar10_lookup[idx+1])
        else:
            lblstr.append(cifar10_lookup[lbl_arr[lbl, :]+1])
    f, axarr = plt.subplots(2,2)
    axarr[0,0].imshow(img_arr[0, ...])
    axarr[0,0].set_title(lblstr[0])
    axarr[0,1].imshow(img_arr[1, ...])
    axarr[0,1].set_title(lblstr[1])
    axarr[1,0].imshow(img_arr[2, ...])
    axarr[1,0].set_title(lblstr[2])
    axarr[1,1].imshow(img_arr[3, ...])
    axarr[1,1].set_title(lblstr[3])
    plt.show()

batch_size = 10
num_classes = 10
epochs = 5

save_dir = os.path.join(os.getcwd(), 'brisbaneai', 'cifar_models',
                        'cifar_train')
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

print 'training set:    {} images'.format(x_train.shape)
print 'training labels: {}'.format(y_train.shape)
print 'testing set:     {} images'.format(x_test.shape)
print 'testing labels:  {}'.format(y_test.shape)

# create one hot vectors
y_train = ks.utils.to_categorical(y_train, num_classes)
y_test = ks.utils.to_categorical(y_test, num_classes)
print 'training labels (one-hot encoded): {}'.format(y_train.shape)
print 'testing labels (one-hot encoded):  {}'.format(y_test.shape)

idx = np.random.randint(0, y_train.shape[0], 4)
disp_data(x_train[idx, ...], y_train[idx, ...])

# model = createNetwork(x_train.shape[1:], num_classes, print_summary=False)
