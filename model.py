# -*- coding: utf-8 -*-
""" 
   Residual Attention Network for Brain Age Estimation (RAN4BAE)

   This code is modified from the residual attention network (RAN) skeleton.
   (Wang, Fei, et.al., Residual attention network for image classification, CVPR, 2017) 
   
   You can find relevant demo about RAN in the below website.
   https://colab.research.google.com/drive/0Byp2051RIziCYVNMNmpwb0hyTVdVM2lueXFWcXZHcmdoLXpN
    
   Author: allard wen.shi 2019.08
    
"""

from __future__ import print_function
import os
import keras 
import pickle
import numpy as np
import tensorflow as tf
import keras.backend as K
from keras.models import Model
from keras.layers import Layer
from keras.optimizers import SGD
from keras.initializers import glorot_normal
from distutils.version import LooseVersion
from keras.initializers import glorot_uniform
from keras.callbacks import ReduceLROnPlateau, Callback
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.layers import (Input, Multiply,Add, Dense, Activation, 
                          BatchNormalization, Flatten, Conv2D, 
                          AveragePooling2D, MaxPooling2D,Lambda)

# version requirement
assert LooseVersion(tf.__version__) >= LooseVersion("1.10.0")
assert LooseVersion(keras.__version__) >= LooseVersion('2.2.4')

## utils

def res_conv(X, filters, base, s):
    
    name_base = base + '/branch'
    f1, f2, f3 = filters    
    X_shortcut = X
    
    # First component of Branch1 
    X = BatchNormalization(axis=-1, name=name_base + '1/bn_1')(X)
    X= Activation('relu', name=name_base + '1/relu_1')(X)
    X = Conv2D(filters=f1, kernel_size=1, strides=1, padding='valid', name=name_base + '1/conv_1', kernel_initializer=glorot_uniform())(X)

    # Second component of Branch1
    X = BatchNormalization(axis=-1, name=name_base + '1/bn_2')(X)
    X = Activation('relu', name=name_base + '1/relu_2')(X)
    X = Conv2D(filters=f2, kernel_size=3, strides=s, padding='same', name=name_base + '1/conv_2', kernel_initializer=glorot_uniform())(X)
    
    # Third component of Branch1
    X = BatchNormalization(axis=-1, name=name_base + '1/bn_3')(X)
    X = Activation('relu', name=name_base + '1/relu_3')(X)
    X = Conv2D(filters=f3, kernel_size=1, strides=1, padding='valid', name=name_base + '1/conv_3', kernel_initializer=glorot_uniform())(X)
    
    X_shortcut = BatchNormalization(axis=-1, name=name_base + '2/bn_1')(X_shortcut)
    X_shortcut= Activation('relu', name=name_base + '2/relu_1')(X_shortcut)
    X_shortcut = Conv2D(filters=f3, kernel_size=1, strides=s, padding='valid', name=name_base + '2/conv_1', kernel_initializer=glorot_uniform())(X_shortcut)
    
    # Final step: Add Branch1 and Branch2
    X = Add(name=base + '/Add')([X, X_shortcut])

    return X

def res_identity(X, filters, base):
    """ 
    Stacks of (1 x 1)-(3 x 3)-(1 x 1) Conv2D, so-called bottleneck layer.

    """
    
    name_base = base + '/branch'
    f1, f2, f3 = filters

    X_shortcut = X
    
    # First component of Branch1 
    X = BatchNormalization(axis=-1, name=name_base + '1/bn_1')(X)
    Shortcut= Activation('relu', name=name_base + '1/relu_1')(X)
    X = Conv2D(filters=f1, kernel_size=(1,1), strides=(1,1), padding='valid', name=name_base + '1/conv_1', kernel_initializer=glorot_uniform())(Shortcut)

    # Second component of Branch1
    X = BatchNormalization(axis=-1, name=name_base + '1/bn_2')(X)
    X = Activation('relu', name=name_base + '1/relu_2')(X)
    X = Conv2D(filters=f2, kernel_size=(3,3), strides=(1,1), padding='same', name=name_base + '1/conv_2', kernel_initializer=glorot_uniform())(X)
    
    # Third component of Branch1
    X = BatchNormalization(axis=-1, name=name_base + '1/bn_3')(X)
    X = Activation('relu', name=name_base + '1/relu_3')(X)
    X = Conv2D(filters=f3, kernel_size=(1,1), strides=(1,1), padding='valid', name=name_base + '1/conv_3', kernel_initializer=glorot_uniform())(X)    
    
    # Final step: Add Branch1 and the original Input itself
    X = Add(name=base + '/Add')([X, X_shortcut])

    return X

def Trunk_block(x, F, name_base):
        
    x = res_identity(x, F, name_base + '/Residual_id_1')
    x = res_identity(x, F, name_base + '/Residual_id_2')
    
    return x

def interpolation(input_tensor, ref_tensor,name): 
    """
    resize the tensor 
    """
    
    H, W = ref_tensor.get_shape()[1], ref_tensor.get_shape()[2]
    return tf.image.resize_bilinear(input_tensor, [H.value, W.value],name=name)

def select_layer(input_tensor,name):
    
    brunch1 = input_tensor
    brunch2 = input_tensor
    
    return brunch1, brunch2

def softplus(input_tensor,name):
    
    return K.log(1 + K.exp(input_tensor)) + 1e-6  
    

def attention_module_1(X, filters, base):
    
    F1, F2, F3 = filters
    
    name_base = base
    
    X = res_identity(X, filters, name_base+ '/Pre_Residual_id')
    X_Trunk = Trunk_block(X, filters, name_base+ '/Trunk')
    
    X = MaxPooling2D((3,3), strides=(2,2), padding='same', name=name_base+ '/Mask/pool_3')(X)
    X = res_identity(X, filters, name_base+ '/Mask/Residual_id_3_Down')
    Residual_id_3_Down_shortcut = X
    Residual_id_3_Down_branched = res_identity(X, filters, name_base+ '/Mask/Residual_id_3_Down_branched')
    
    X = MaxPooling2D((3,3), strides=(2,2), padding='same', name=name_base+ '/Mask/pool_2')(X)
    X = res_identity(X, filters, name_base+ '/Mask/Residual_id_2_Down')
    Residual_id_2_Down_shortcut = X
    Residual_id_2_Down_branched = res_identity(X, filters, name_base+ '/Mask/Residual_id_2_Down_branched')
    
    X = MaxPooling2D((3,3), strides=(2,2), padding='same', name=name_base+ '/Mask/pool_1')(X)
    X = res_identity(X, filters, name_base+ '/Mask/Residual_id_1_Down')
    X = res_identity(X, filters, name_base+ '/Mask/Residual_id_1_Up')
    temp_name1 = name_base+ "/Mask/Interpool_1"
    
    X = Lambda(interpolation, arguments={'ref_tensor': Residual_id_2_Down_shortcut,'name':temp_name1})(X)                                 
    X = Add(name=base + '/Mask/Add_after_Interpool_1')([X, Residual_id_2_Down_branched])
                                          
    X = res_identity(X, filters, name_base+ '/Mask/Residual_id_2_Up')
    
    temp_name2 = name_base+ "/Mask/Interpool_2"
    X = Lambda(interpolation, arguments={'ref_tensor': Residual_id_3_Down_shortcut,'name':temp_name2})(X)                                     
    X = Add(name=base + '/Mask/Add_after_Interpool_2')([X, Residual_id_3_Down_branched])
                                     
    X = res_identity(X, filters, name_base+ '/Mask/Residual_id_3_Up')
    
    temp_name3 = name_base+ "/Mask/Interpool_3" 
    X = Lambda(interpolation, arguments={'ref_tensor': X_Trunk,'name':temp_name3})(X)                                     
    X = BatchNormalization(axis=-1, name=name_base + '/Mask/Interpool_3/bn_1')(X)                                 
    X = Activation('relu', name=name_base + '/Mask/Interpool_3/relu_1')(X)                                
    
    X = Conv2D(F3, kernel_size=(1,1), strides=(1,1), padding='valid', name=name_base + '/Mask/Interpool_3/conv_1', kernel_initializer=glorot_uniform())(X)
    X = BatchNormalization(axis=-1, name=name_base + '/Mask/Interpool_3/bn_2')(X)                                   
    X = Activation('relu', name=name_base + '/Mask/Interpool_3/relu_2')(X)                                   
    
    X = Conv2D(F3, kernel_size=(1,1), strides=(1,1), padding='valid', name=name_base + '/Mask/Interpool_3/conv_2', kernel_initializer=glorot_uniform())(X)
    X = Activation('sigmoid', name=name_base+'/Mask/sigmoid')(X)
    X = Multiply(name=name_base+'/Multiply')([X_Trunk,X])
    X = Add(name=name_base+'/Add')([X_Trunk,X])

    X = res_identity(X, filters, name_base+ '/Post_Residual_id')
    
    return X

def attention_module_2(X, filters, base):
    
    # Define filter channel
    F1, F2, F3 = filters
    
    # Define name
    name_base = base
    
    X = res_identity(X, filters, name_base+ '/Pre_Residual_id')
    X_Trunk = Trunk_block(X, filters, name_base+ '/Trunk')
    X = MaxPooling2D((3,3), strides=(2,2), padding='same', name=name_base+ '/Mask/pool_2')(X)
    X = res_identity(X, filters, name_base+ '/Mask/Residual_id_2_Down')
    
    Residual_id_2_Down_shortcut = X
    Residual_id_2_Down_branched = res_identity(X, filters, name_base+ '/Mask/Residual_id_2_Down_branched')
    
    X = MaxPooling2D((3,3), strides=(2,2), padding='same', name=name_base+ '/Mask/pool_1')(X)
    X = res_identity(X, filters, name_base+ '/Mask/Residual_id_1_Down')                                     
    X = res_identity(X, filters, name_base+ '/Mask/Residual_id_1_Up')
    
    temp_name1 = name_base+ "/Mask/Interpool_1"
    X = Lambda(interpolation, arguments={'ref_tensor': Residual_id_2_Down_shortcut,'name':temp_name1})(X)                                      
    X = Add(name=base + '/Mask/Add_after_Interpool_1')([X, Residual_id_2_Down_branched])                                 
    X = res_identity(X, filters, name_base+ '/Mask/Residual_id_2_Up')
    
    temp_name2 = name_base+ "/Mask/Interpool_2"
    X = Lambda(interpolation, arguments={'ref_tensor': X_Trunk,'name':temp_name2})(X)                                     
    X = BatchNormalization(axis=-1, name=name_base + '/Mask/Interpool_2/bn_1')(X)                                   
    X = Activation('relu', name=name_base + '/Mask/Interpool_2/relu_1')(X)                                      
   
    X = Conv2D(F3, kernel_size=(1,1), strides=(1,1), padding='valid', name=name_base + '/Mask/Interpool_2/conv_1', kernel_initializer=glorot_uniform())(X)
    X = BatchNormalization(axis=-1, name=name_base + '/Mask/Interpool_2/bn_2')(X)                                     
    X = Activation('relu', name=name_base + '/Mask/Interpool_2/relu_2')(X)                                     
   
    X = Conv2D(F3, kernel_size=(1,1), strides=(1,1), padding='valid', name=name_base + '/Mask/Interpool_2/conv_2', kernel_initializer=glorot_uniform())(X)
    X = Activation('sigmoid', name=name_base+'/Mask/sigmoid')(X) 
    X = Multiply(name=name_base+'/Multiply')([X_Trunk,X])
    X = Add(name=name_base+'/Add')([X_Trunk,X])

    X = res_identity(X, filters, name_base+ '/Post_Residual_id')
    
    return X

def attention_module_3(X, filters, base):
    
    f1, f2, f3 = filters
    name_base = base
    
    X = res_identity(X, filters, name_base+ '/Pre_Residual_id') 
    
    # The mask branch has the same architecture of truck block in attention module 1 and 2
    X_Mask = Trunk_block(X, filters, name_base+ '/Mask')
    X_Mask = Activation('sigmoid', name=name_base+'/Mask/sigmoid')(X_Mask)
    
    X = BatchNormalization(axis=-1, name=name_base + '/Mask/Interpool_2/bn_1')(X)                                     
    X = Activation('relu', name=name_base + '/Mask/Interpool_2/relu_1')(X)                                      
    X = Conv2D(f3, kernel_size=(1,1), strides=(1,1), padding='valid', name=name_base + '/Trunk/Interpool_2/conv_1', kernel_initializer=glorot_uniform())(X)
    X = BatchNormalization(axis=-1, name=name_base + '/Trunk/Interpool_2/bn_2')(X)                                     
    X = Activation('relu', name=name_base + '/Trunk/Interpool_2/relu_2')(X)                                
    X = Conv2D(f3, kernel_size=(1,1), strides=(1,1), padding='valid', name=name_base + '/Trunk/Interpool_2/conv_2', kernel_initializer=glorot_uniform())(X)
    
    X_Mask = Multiply(name=name_base+'/Multiply')([X_Mask,X])
    X = Add(name=name_base+'/Add')([X_Mask,X])
        
    X = res_identity(X, filters, name_base+ '/Post_Residual_id')
    
    return X

def custom_loss(sigma):
    def gaussian_loss(y_true, y_pred):
        # Calculate uncertainty
        return tf.reduce_mean(0.5*tf.log(sigma) + 0.5*tf.div(tf.square(y_true - y_pred), sigma)) + 1e-6
    return gaussian_loss

class UncertaintyLayer(Layer):
    """
    Custom layer
    
    """
    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        super(UncertaintyLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.weight_1 = self.add_weight(name='weight_1', 
                                      shape=(input_shape[1], self.output_dim),
                                      initializer=glorot_normal(),
                                      trainable=True)
        self.weight_2 = self.add_weight(name='weight_2', 
                                      shape=(input_shape[1], self.output_dim),
                                      initializer=glorot_normal(),
                                      trainable=True)
        self.bias_1 = self.add_weight(name='bias_1',
                                    shape=(self.output_dim, ),
                                    initializer=glorot_normal(),
                                    trainable=True)
        self.bias_2 = self.add_weight(name='bias_2',
                                    shape=(self.output_dim, ),
                                    initializer=glorot_normal(),
                                    trainable=True)
        super(UncertaintyLayer, self).build(input_shape) 

    def call(self, x):
        output_mu  = K.dot(x, self.weight_1) + self.bias_1
        output_sig = K.dot(x, self.weight_2) + self.bias_2
        output_sig_pos = K.log(1 + K.exp(output_sig)) + 1e-06  
        return [output_mu, output_sig_pos]

    def compute_output_shape(self, input_shape):
        return [(input_shape[0], self.output_dim), (input_shape[0], self.output_dim)]


def RANet(input_shape, fil = [32,64,128], uncertainty = False):
    """
    # Inputs
        - input_shape (tensor): shape of input image tensor
        
        - fil (list): list of filter channel 
        
        - uncertainty (bool): whether to estimate uncertainty 
        
    # Returns
        - model (Model): Keras model instance  
        
        - sigma: for aleatoric unceratiny calculation 
      
     """
    
    x_input = Input(input_shape)
    
    x = Conv2D(32, 3, strides=2, padding='same', name='conv_1', kernel_initializer=glorot_uniform())(x_input)
    x = BatchNormalization(axis=-1, name='bn_1')(x)
    x = Activation('relu', name='relu_1')(x)
    x = MaxPooling2D((3,3), strides=(2,2), padding='same' ,name='pool_1')(x)
    x = res_conv(x, fil, 'Residual_conv_1', 1)
  
    x = attention_module_1(x, fil, 'attention_layer1')
    x = res_conv(x, [x*2 for x in fil], 'Residual_conv_2', 2)
    
    x = attention_module_2(x, [x*2 for x in fil], 'attention_layer2')
    x = res_conv(x, [x*4 for x in fil], 'Residual_conv_3', 2)
    
    x = attention_module_3(x, [x*4 for x in fil], 'attention_layer3')
 
    x = res_identity(x, [x*4 for x in fil], 'Residual_id_1')
    x = res_identity(x, [x*4 for x in fil], 'Residual_id_2')
   
    x = BatchNormalization(axis=-1, name='bn_2')(x)
    x = Activation('relu', name='relu_2')(x)
    x = AveragePooling2D(pool_size=(4,4), name='avg_pool')(x)
    x = Flatten()(x) 
    
    if uncertainty:
        mu, sigma = UncertaintyLayer(1, name='uncertainty_output')(x)
        model = Model(inputs=x_input, outputs=mu)
        
        return model, sigma
    
    else:
        outputs = Dense(1, kernel_initializer='he_normal')(x)
        model = Model(inputs=x_input, outputs=outputs)
        
        return model 
     
## Some bugs in it, please wait for the next version
#class SGDLearningRateTracker(Callback):
#    def on_epoch_end(self, epoch, logs={}):
#        optimizer = self.model.optimizer
#        lr = optimizer.lr * (1. / (1. + optimizer.decay * optimizer.iterations))
#      
#        print('\nLR: {:.6f}\n'.format(lr))

def lr_schedule(epoch):
    """Learning Rate Schedule

    Learning rate is rescheduled to be reduced after each epochs.

    # Arguments
        epoch (int): The number of epochs

    # Returns
        lr (float32): learning rate
        
    """
    decay = 0.05
    lr = 1e-2
    
    if epoch > 20:
        lr = 1e-2/(1+decay*epoch)

    
    print('Learning rate: ', lr)
    return lr

def ran4bae_load(input_shape, weight_path, uncertainty = False):
    """ Load RAN4BAE weights

    # Inputs
       - input_shape: The number of epochs
        
       - weight_path: saved weight path
        
       - uncertainty(bool): select whether to use uncertainty 

    # Returns
       - model: RAN4BAE Model
        
    """

    model, _ = RANet(input_shape=input_shape,   
                    uncertainty=uncertainty)
    model.load_weights(weight_path)
    
    return model
       
def ran4bae_run(opt):
    """  Train the network
         
         The model weights are automatically saved in the path: saved_models      
         
    """
        
    x_train = pickle.load(open(opt.x_train, 'rb' ) )
    x_validation = pickle.load(open(opt.x_validation, 'rb' ) ) 
    
    y_train = pickle.load(open(opt.y_train, 'rb' ) )
    y_validation = pickle.load(open(opt.y_validation, 'rb' ) )

    input_shape = x_train.shape[1:]
    nb_channel = x_train.shape[3]
    
    print('='*20+' data info '+'='*20)
    print('number of channels: ', nb_channel)
    print('input shape: ', opt.input_shape)
    print(x_train.shape[0], ' train samples')
    print(x_validation.shape[0], ' validation samples')
    
    # set GPU usage
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
        
    input_shape = x_train.shape[1:]
    
    # select proper optimizer
    optimizer_select = SGD(lr=opt.init_lr, decay=0.05, momentum=0.9, nesterov=False, clipnorm=1.)
    #optimizer_select = Adam(lr = lr_schedule(epochs))
    
    if opt.choose_uncertainty:
        model, sigma = RANet(input_shape=input_shape, 
                             uncertainty=opt.choose_uncertainty)
        model.compile(loss=custom_loss(sigma),
                          optimizer=optimizer_select,
                          metrics=['mse','mae'])
    else:
        model = RANet(input_shape=input_shape,
                      uncertainty=opt.choose_uncertainty)
        model.compile(loss='mse',
                      optimizer=optimizer_select,
                      metrics=['mse','mae'])
        
    model.summary()
    print(opt.model_name)
    
    # prepare model saving directory.
    save_dir = os.path.join(os.getcwd(), 'saved_models')
    model_name = '%s_model.{epoch:03d}.h5' % opt.model_name
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    filepath = os.path.join(save_dir, model_name)
    
    # prepare callbacks for model saving and for learning rate adjustment.
    checkpoint = ModelCheckpoint(filepath=filepath,
                                 monitor='val_mean_squared_error',
                                 verbose=1,
                                 save_best_only=True,
                                 save_weights_only=True)
    
    # set learning rate reducer     
    lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1),
                                   cooldown=0,
                                   patience=5,
                                   min_lr=0.5e-4)
    
    # manual design lr with callback
    lr_scheduler = LearningRateScheduler(lr_schedule)
    
    
    # Run training, with or without data augmentation.
    if not opt.data_augmentation:
        print('Not using data augmentation.')
        history = model.fit(x_train, y_train,
                  batch_size=opt.batch_size,
                  epochs=opt.epochs,
                  validation_data=(x_validation, y_validation),
                  shuffle=True,
                  callbacks=[checkpoint, lr_reducer])
    else:
        print('Using real-time data augmentation.')
        # This will do preprocessing and realtime data augmentation:
        datagen = ImageDataGenerator(
            # set input mean to 0 over the dataset
            featurewise_center=False,
            # set each sample mean to 0
            samplewise_center=False,
            # divide inputs by std of dataset
            featurewise_std_normalization=False,
            # divide each input by its std
            samplewise_std_normalization=False,
            # apply ZCA whitening
            zca_whitening=False,
            # epsilon for ZCA whitening
            zca_epsilon=0,
            # randomly rotate images in the range (deg 0 to 180)
            rotation_range=22,
            # randomly shift images horizontally
            width_shift_range=0.02,
            # randomly shift images vertically
            height_shift_range=0.02,
            # set range for random shear
            shear_range=0.01,
            # set range for random zoom
            zoom_range=0.,
            # set range for random channel shifts
            channel_shift_range=0.,
            # set mode for filling points outside the input boundaries
            fill_mode='constant',
            # value used for fill_mode = "constant"
            cval=0.,
            # randomly flip images
            horizontal_flip=True,
            # randomly flip images
            vertical_flip=True,
            # set rescaling factor (applied before any other transformation)
            rescale=None,
            # set function that will be applied on each input
            preprocessing_function=None,
            # image data format, either "channels_first" or "channels_last"
            data_format=None,
            # fraction of images reserved for validation (strictly between 0 and 1)
            validation_split=0.0)
    
        datagen.fit(x_train)
    
        # Fit the model on the batches generated by datagen.flow().
        history = model.fit_generator(datagen.flow(x_train, y_train,batch_size=opt.batch_size),
                                      validation_data=(x_validation, y_validation),
                                      epochs=opt.epochs, verbose=1, workers=4, shuffle=True,
                                      callbacks=[checkpoint, lr_reducer, lr_scheduler])
    
    # score trained model.
    scores = model.evaluate(x_validation, y_validation, verbose=1)
    print('Validation loss:', scores[0])
    print('Validation accuracy:', scores[1])
    

    # prepare history saving directory.
    if not os.path.isdir('history/'):
        os.makedirs('history')
    pickle.dump(history.history, open('history/'+opt.model_name+'_history.p', 'wb' ) )
