from keras.models import Model
from keras.layers import Input, BatchNormalization, Activation
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.merge import concatenate, Add


def conv2d_block(input_tensor, n_filters, kernel_size=3, batchnorm=True):
    x = Conv2D(filters=n_filters, kernel_size=kernel_size, kernel_initializer="he_normal", padding="same", kernel_regularizer='l2')(input_tensor)
    if batchnorm:
        x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Conv2D(filters=n_filters, kernel_size=kernel_size, kernel_initializer="he_normal", padding="same", kernel_regularizer='l2')(x)
    if batchnorm:
        x = BatchNormalization()(x)
    y = Activation("relu")(x)

    out = y
    
    return out


def get_unet(im_height, im_width, im_depth, n_filters=1, batchnorm=True):
    # contracting path
    input_img = Input((im_height, im_width, im_depth), name='img')
    c1 = conv2d_block(input_img, n_filters=n_filters*1, kernel_size=3, batchnorm=batchnorm, resblock=resblock)
    p1 = Conv2D(filters=n_filters*2, kernel_size=3, kernel_initializer="he_normal", padding="same", kernel_regularizer='l2', strides=(2,2))(c1)

    c2 = conv2d_block(p1, n_filters=n_filters*2, kernel_size=3, batchnorm=batchnorm, resblock=resblock)
    p2 = Conv2D(filters=n_filters*4, kernel_size=3, kernel_initializer="he_normal", padding="same", kernel_regularizer='l2', strides=(2,2))(c2)

    c3 = conv2d_block(p2, n_filters=n_filters*4, kernel_size=3, batchnorm=batchnorm, resblock=resblock)
    p3 = Conv2D(filters=n_filters*8, kernel_size=3, kernel_initializer="he_normal", padding="same", kernel_regularizer='l2', strides=(2,2))(c3)

    c4 = conv2d_block(p3, n_filters=n_filters*8, kernel_size=3, batchnorm=batchnorm, resblock=resblock)
    p4 = Conv2D(filters=n_filters*16, kernel_size=3, kernel_initializer="he_normal", padding="same", kernel_regularizer='l2', strides=(2,2))(c4)

    c5 = conv2d_block(p4, n_filters=n_filters*16, kernel_size=3, batchnorm=batchnorm, resblock=resblock)
    
    # Covid 2D expansive path
    u6 = Conv2DTranspose(n_filters*8, (3, 3), strides=(2, 2), padding='same', kernel_regularizer='l2') (c5)
    u6 = concatenate([u6, c4])
    c6 = conv2d_block(u6, n_filters=n_filters*8, kernel_size=3, batchnorm=batchnorm, resblock=resblock)
        
    u7 = Conv2DTranspose(n_filters*4, (3, 3), strides=(2, 2), padding='same', kernel_regularizer='l2') (c6)
    u7 = concatenate([u7, c3])
    c7 = conv2d_block(u7, n_filters=n_filters*4, kernel_size=3, batchnorm=batchnorm, resblock=resblock)
        
    u8 = Conv2DTranspose(n_filters*2, (3, 3), strides=(2, 2), padding='same', kernel_regularizer='l2') (c7)
    u8 = concatenate([u8, c2])
    c8 = conv2d_block(u8, n_filters=n_filters*2, kernel_size=3, batchnorm=batchnorm, resblock=resblock)

    u9 = Conv2DTranspose(n_filters*1, (3, 3), strides=(2, 2), padding='same', kernel_regularizer='l2') (c8)
    u9 = concatenate([u9, c1])
    c9 = conv2d_block(u9, n_filters=n_filters*1, kernel_size=3, batchnorm=batchnorm, resblock=resblock)
    
    covid_output = Conv2D(1, (1,1), activation='sigmoid', name = 'covid_output', kernel_regularizer='l2')(c9)
        
    # Lung 2D expansive path
    u6 = Conv2DTranspose(n_filters*8, (3, 3), strides=(2, 2), padding='same', kernel_regularizer='l2') (c5)
    u6 = concatenate([u6, c4])
    c6 = conv2d_block(u6, n_filters=n_filters*8, kernel_size=3, batchnorm=batchnorm, resblock=resblock)

    u7 = Conv2DTranspose(n_filters*8, (3, 3), strides=(2, 2), padding='same', kernel_regularizer='l2') (c6)
    u7 = concatenate([u7, c3])
    c7 = conv2d_block(u7, n_filters=n_filters*4, kernel_size=3, batchnorm=batchnorm, resblock=resblock)

    u8 = Conv2DTranspose(n_filters*8, (3, 3), strides=(2, 2), padding='same', kernel_regularizer='l2') (c7)
    u8 = concatenate([u8, c2])
    c8 = conv2d_block(u8, n_filters=n_filters*2, kernel_size=3, batchnorm=batchnorm, resblock=resblock)

    u9 = Conv2DTranspose(n_filters*8, (3, 3), strides=(2, 2), padding='same', kernel_regularizer='l2') (c8)
    u9 = concatenate([u9, c1])
    c9 = conv2d_block(u9, n_filters=n_filters*1, kernel_size=3, batchnorm=batchnorm, resblock=resblock)
    
    lung_output = Conv2D(1, (1,1), activation='sigmoid', name = 'lung_output', kernel_regularizer='l2')(c9)

    model = Model(inputs=[input_img], outputs=[covid_output, lung_output])
        
    return model

##Features can be extracted from the segmented COVID-19 and lung volumes using numpy package and commented code below:
#volume_ratio = np.sum(covid_vol)/np.sum(lung_vol)
#as well as other features such as np.mean(), np.std(), np.percentile(,75), np.amin(), np.amax(), np.sum(image_volume[seg==1])
