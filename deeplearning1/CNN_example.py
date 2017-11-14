'''
A brief introduction into finetuning deep CNNs using Cifa10
Will conver data preprocessing, data augmentation, finetunig and regularization.

Based on the Keras Cifar10 CNN example code at:
https://github.com/fchollet/keras/blob/master/examples/cifar10_cnn.py

----------------------------------------------------------
Look at the function "main" first, line 288 or something.
"def main(train_params, verbose=False, train=True):"
----------------------------------------------------------

Created by Sean McMahon on the 10th Nov 2017.
'''

from keras.datasets import cifar10
import keras
import numpy as np
import os
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator


def createNetwork_noDropout(input_shape, num_classes, print_summary=True):
    model = keras.models.Sequential()
    model.add(keras.layers.Conv2D(32, (3, 3), padding='same',
                                  input_shape=input_shape))
    model.add(keras.layers.Activation('relu'))
    model.add(keras.layers.Conv2D(32, (3, 3)))
    model.add(keras.layers.Activation('relu'))
    model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))

    model.add(keras.layers.Conv2D(64, (3, 3), padding='same'))
    model.add(keras.layers.Activation('relu'))
    model.add(keras.layers.Conv2D(64, (3, 3)))
    model.add(keras.layers.Activation('relu'))
    model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))

    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(512))
    model.add(keras.layers.Activation('relu'))
    model.add(keras.layers.Dense(num_classes))
    model.add(keras.layers.Activation('softmax'))

    if print_summary:
        print '\nCreated network: '
        print model.summary()
        print '\n'

    return model


def createNetwork(input_shape, num_classes, print_summary=True):
    '''
    The base network given in the Cifar10 example by Keras.
    '''
    model = keras.models.Sequential()
    model.add(keras.layers.Conv2D(32, (3, 3), padding='same',
                                  input_shape=input_shape))
    model.add(keras.layers.Activation('relu'))
    model.add(keras.layers.Conv2D(32, (3, 3)))
    model.add(keras.layers.Activation('relu'))
    model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(keras.layers.Dropout(0.25))

    model.add(keras.layers.Conv2D(64, (3, 3), padding='same'))
    model.add(keras.layers.Activation('relu'))
    model.add(keras.layers.Conv2D(64, (3, 3)))
    model.add(keras.layers.Activation('relu'))
    model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(keras.layers.Dropout(0.25))

    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(512))
    model.add(keras.layers.Activation('relu'))
    model.add(keras.layers.Dropout(0.5))
    model.add(keras.layers.Dense(num_classes))
    model.add(keras.layers.Activation('softmax'))

    if print_summary:
        print '\nCreated network: '
        print model.summary()
        print '\n'

    return model


def createNetworkBN(input_shape, num_classes, print_summary=True):
    '''
    Put BN before or after activation?
    https://github.com/ducha-aiki/caffenet-benchmark/blob/master/batchnorm.md#bn----before-or-after-relu
    Appears afterwards is better, despite the authors suggesting before:
    https://github.com/fchollet/keras/issues/1802#issuecomment-187966878
    '''
    model = keras.models.Sequential()
    model.add(keras.layers.Conv2D(32, (3, 3), padding='same',
                                  input_shape=input_shape))
    model.add(keras.layers.normalization.BatchNormalization(axis=1))
    model.add(keras.layers.Activation('relu'))
    model.add(keras.layers.Conv2D(32, (3, 3)))
    model.add(keras.layers.normalization.BatchNormalization(axis=1))
    model.add(keras.layers.Activation('relu'))
    model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(keras.layers.Dropout(0.25))

    model.add(keras.layers.Conv2D(64, (3, 3), padding='same'))
    model.add(keras.layers.normalization.BatchNormalization(axis=1))
    model.add(keras.layers.Activation('relu'))
    model.add(keras.layers.Conv2D(64, (3, 3)))
    model.add(keras.layers.normalization.BatchNormalization(axis=1))
    model.add(keras.layers.Activation('relu'))
    model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(keras.layers.Dropout(0.25))

    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(512))
    model.add(keras.layers.normalization.BatchNormalization())
    model.add(keras.layers.Activation('relu'))
    model.add(keras.layers.Dropout(0.5))
    model.add(keras.layers.Dense(num_classes))
    model.add(keras.layers.Activation('softmax'))

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
            lblstr.append(cifar10_lookup[idx + 1])
        else:
            lblstr.append(cifar10_lookup[lbl_arr[lbl, :][0] + 1])
    f, axarr = plt.subplots(2, 2)
    axarr[0, 0].imshow(img_arr[0, ...])
    axarr[0, 0].set_title(lblstr[0])
    axarr[0, 1].imshow(img_arr[1, ...])
    axarr[0, 1].set_title(lblstr[1])
    axarr[1, 0].imshow(img_arr[2, ...])
    axarr[1, 0].set_title(lblstr[2])
    axarr[1, 1].imshow(img_arr[3, ...])
    axarr[1, 1].set_title(lblstr[3])
    plt.show()


def printDataInfo(x_train, y_train, x_test, y_test, plotting=True):
    print 'Dataset Statistics (by no means comprehensive):\n', '=' * 40
    print 'training set:    {} images'.format(x_train.shape)
    print 'training labels: {}'.format(y_train.shape)
    print 'testing set:     {} images'.format(x_test.shape)
    print 'testing labels:  {}'.format(y_test.shape)
    print '\n', '-' * 30, '\nValues of Data:'
    print 'Range of train images: {}-{}, datatype: {}'.format(x_train.min(), x_train.max(), x_train.dtype)
    print 'Range of train labels: {}-{}, datatype: {}'.format(y_train.min(), y_train.max(), y_train.dtype)
    print 'Range of test images: {}-{}, datatype: {}'.format(x_test.min(), x_test.max(), x_test.dtype)
    print 'Range of test labels: {}-{}, datatype: {}'.format(y_test.min(), y_test.max(), y_test.dtype)
    if plotting:
        for _ in range(1):
            idx = np.random.randint(0, y_train.shape[0], 4)
            disp_data(x_train[idx, ...], y_train[idx, ...])
    print '=' * 40


def create_model(train_params, input_shape, print_summary=True):
    '''
    A couple of different networks for experimenting with.
    This would typicall be VGG for most applications, but as this is just a toy
    example, we're using a custom network trained from scratch.
    '''
    if train_params['batch_norm']:
        print 'Batch norm and Dropout'
        model = createNetworkBN(
            input_shape, train_params['num_classes'], print_summary=print_summary)
    elif train_params['no_dropout']:
        print 'no Batch Norm or Dropout'
        model = createNetwork_noDropout(
            input_shape, train_params['num_classes'], print_summary=print_summary)
    else:
        print 'Dropout, no batch norm'
        model = createNetwork(
            input_shape, train_params['num_classes'], print_summary=print_summary)
    return model


def train_model(model, x_train, y_train, x_test, y_test, train_params):
    '''
    Setup the solver/optimizer you want to use.
    Link it to the network/model/graph.
    (optional) Setup data augmention, here a class already exists.
    Begin training.
    Save the trained weights.
    '''
    model_path = os.path.join(
        train_params['save_dir'], train_params['model_name'])

    # initiate RMSprop optimizer an improvement on SGD, similar to Adam
    opt = keras.optimizers.rmsprop(
        lr=train_params.get('lr', 0.0001), decay=1e-6)

    # Let's train the model using SGD
    model.compile(loss='categorical_crossentropy',
                  optimizer=opt,
                  metrics=['accuracy'])

    # Augment Data and Train!
    if not train_params['data_augmentation']:
        print('Not using data augmentation.')
        model.fit(x_train, y_train,
                  batch_size=train_params['batch_size'],
                  epochs=train_params['epochs'],
                  validation_data=(x_test, y_test),
                  shuffle=True)
    else:
        print('Using real-time data augmentation.')
        # This will do preprocessing and realtime data augmentation:
        datagen = ImageDataGenerator(
            featurewise_center=False,  # set input mean to 0 over the dataset
            samplewise_center=False,  # set each sample mean to 0
            featurewise_std_normalization=False,  # divide inputs by std of the dataset
            samplewise_std_normalization=False,  # divide each input by its std
            zca_whitening=False,  # apply ZCA whitening
            # randomly rotate images in the range (degrees, 0 to 180)
            rotation_range=5,
            # randomly shift images horizontally (fraction of total width)
            width_shift_range=0.1,
            # randomly shift images vertically (fraction of total height)
            height_shift_range=0.1,
            horizontal_flip=True,  # randomly flip images
            vertical_flip=False)  # randomly flip images

        # Compute quantities required for feature-wise normalization
        # (std, mean, and principal components if ZCA whitening is applied).
        datagen.fit(x_train)

        # Fit the model on the batches generated by datagen.flow().
        model.fit_generator(datagen.flow(x_train, y_train,
                                         batch_size=train_params['batch_size']),
                            steps_per_epoch=int(
                                np.ceil(x_train.shape[0] / float(train_params['batch_size']))),
                            epochs=train_params['epochs'],
                            validation_data=(x_test, y_test),
                            workers=4)

    model.save(model_path)
    print 'Saved trained model to %s' % model_path

    scores = model.evaluate(x_test, y_test, verbose=1)
    print 'Test loss {}'.format(scores[0])
    print 'Test accuracy {}'.format(scores[1])


def main(train_params, verbose=False, train=True):
    '''
    Load the data then visualise it.
    Once the data has been checked do the preprocessing.
    Create the model
        - usually initialise a pre-trained network, but that's not done here.
    Setup Data augmentation and trainining paramerts.
    '''
    save_dir = os.path.join(os.getcwd(), 'brisbaneai', 'cifar_models',
                            'cifar_train')
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    train_params['save_dir'] = save_dir

    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    if verbose:
        printDataInfo(x_train, y_train, x_test, y_test, plotting=True)

    # uncomment to enable debugging
    # import pdb; pdb.set_trace()

    # create one hot vectors
    y_train = keras.utils.to_categorical(y_train, train_params['num_classes'])
    y_test = keras.utils.to_categorical(y_test, train_params['num_classes'])
    if verbose:
        print 'training labels (one-hot encoded): {}'.format(y_train.shape)
        print 'testing labels (one-hot encoded):  {}'.format(y_test.shape)

    # convert to float32's and normalise images
    # Only convert Data to float32, keep labels as uint8's!!!!
    x_train.astype(np.float32)
    x_test.astype(np.float32)
    x_train /= 255
    x_test /= 255

    model = create_model(train_params, x_train.shape[1:], print_summary=True)

    if train:
        train_model(model, x_train, y_train, x_test, y_test, train_params)
    else:
        print 'network not training, set train flag' + \
            ' within "main" function to True to train'
#
if __name__ == '__main__':
    '''
    This block is run when you call "python CNN_example.py", it in turn calls
    all the other functions.
    '''
    no_aug = {'data_augmentation': False,
              'batch_norm': False,
              'no_dropout': False,
              'batch_size': 32,
              'num_classes': 10,
              'epochs': 5,
              'model_name': 'keras_cifar10_no_aug.h5'}

    t_aug = {'data_augmentation': True,
             'batch_norm': False,
             'no_dropout': False,
             'batch_size': no_aug['batch_size'],
             'num_classes': no_aug['num_classes'],
             'epochs': no_aug['epochs'],
             'model_name': 'keras_cifar10_aug.h5'}

    t_noDrop = {'data_augmentation': True,
                'batch_norm': False,
                'no_dropout': True,
                'batch_size': no_aug['batch_size'],
                'num_classes': no_aug['num_classes'],
                'epochs': no_aug['epochs'],
                'model_name': 'keras_cifar10_aug_nodropout.h5'}

    t_bn = {'data_augmentation': True,
            'batch_norm': True,
            'no_dropout': False,
            'batch_size': no_aug['batch_size'],
            'num_classes': no_aug['num_classes'],
            'epochs': no_aug['epochs'],
            'model_name': 'keras_cifar10_aug_BN.h5'}

    main(t_bn, verbose=True, train=True)

    # params = [no_aug, t_aug, t_bn, t_noDrop]
    # for count, p in enumerate(params, 1):
    #     print '=*=' * 40
    #     print 'Running params {}/{}'.format(count, len(params))
    #     for key, value in p.iteritems():
    #         print 'Set "{}" to {}'.format(key, value)
    #     main(p)
    #     print '\n' * 5
