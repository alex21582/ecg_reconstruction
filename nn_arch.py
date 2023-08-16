from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, UpSampling1D, concatenate, Conv1DTranspose, \
    BatchNormalization, Permute, Cropping1D, ZeroPadding1D
from tensorflow.keras.models import Model


# Base architecture
def ann128(input_size=(128, 1)):
    inputs = Input(input_size, name='input')  # (None, 128, 1)

    # Encoder
    conv0 = Conv1D(32, 5, activation='relu', padding='same', name='encoder_conv00')(inputs)  # (None, 128, 32)
    conv0 = Conv1D(32, 5, activation='relu', padding='same', name='encoder_conv01')(conv0)  # (None, 128, 32)
    pool0 = MaxPooling1D(pool_size=2, name='encoder_maxpool0')(conv0)  # (None, 64, 32)

    conv1 = Conv1D(64, 5, activation='relu', padding='same', name='encoder_conv10')(pool0)  # (None, 64, 64)
    conv1 = Conv1D(64, 5, activation='relu', padding='same', name='encoder_conv11')(conv1)  # (None, 64, 64)
    pool1 = MaxPooling1D(pool_size=2, name='encoder_maxpool1')(conv1)  # (None, 32, 64)

    conv2 = Conv1D(128, 5, activation='relu', padding='same', name='encoder_conv20')(pool1)  # (None, 32, 128)
    conv2 = Conv1D(128, 5, activation='relu', padding='same', name='encoder_conv21')(conv2)  # (None, 32, 128)
    pool2 = MaxPooling1D(pool_size=2, name='encoder_maxpool2')(conv2)  # (None, 16, 128)

    conv3 = Conv1D(256, 5, activation='relu', padding='same', name='encoder_conv30')(pool2)  # (None, 16, 256)
    conv3 = Conv1D(256, 5, activation='relu', padding='same', name='encoder_conv31')(conv3)  # (None, 16, 256)
    pool3 = MaxPooling1D(pool_size=2, name='encoder_maxpool3')(conv3)  # (None, 8, 256)

    conv4 = Conv1D(512, 5, activation='relu', padding='same', name='encoder_conv40')(pool3)  # (None, 8, 512)
    conv4 = Conv1D(512, 5, activation='relu', padding='same', name='encoder_conv41')(conv4)  # (None, 8, 512)

    # Decoder
    up5 = UpSampling1D(size=2, name='decoder_up5')(conv4)  # (None, 16, 512)
    merge5 = concatenate([conv3, up5], axis=2, name='decoder_merge5')  # (None, 16, 768)
    conv5 = Conv1D(256, 5, activation='relu', padding='same', name='decoder_conv50')(merge5)  # (None, 64, 256)
    conv5 = Conv1D(256, 5, activation='relu', padding='same', name='decoder_conv51')(conv5)  # (None, 64, 256)

    up6 = UpSampling1D(size=2, name='decoder_up6')(conv5)  # (None, 32, 256)
    merge6 = concatenate([conv2, up6], axis=2, name='decoder_merge6')  # (None, 32, 384)
    conv6 = Conv1D(128, 5, activation='relu', padding='same', name='decoder_conv60')(merge6)  # (None, 128, 128)
    conv6 = Conv1D(128, 5, activation='relu', padding='same', name='decoder_conv61')(conv6)  # (None, 128, 128)

    up7 = UpSampling1D(size=2, name='decoder_up7')(conv6)  # (None, 64, 128)
    merge7 = concatenate([conv1, up7], axis=2, name='decoder_merge7')  # (None, 64, 192)
    conv7 = Conv1D(64, 5, activation='relu', padding='same', name='decoder_conv70')(merge7)  # (None, 256, 64)
    conv7 = Conv1D(64, 5, activation='relu', padding='same', name='decoder_conv71')(conv7)  # (None, 256, 64)

    up8 = UpSampling1D(size=2, name='decoder_up8')(conv7)  # (None, 128, 64)
    merge8 = concatenate([conv0, up8], axis=2, name='decoder_merge8')  # (None, 128, 96)
    conv8 = Conv1D(32, 5, activation='relu', padding='same', name='decoder_conv80')(merge8)  # (None, 128, 32)
    conv8 = Conv1D(32, 5, activation='relu', padding='same', name='decoder_conv81')(conv8)  # (None, 128, 32)

    outputs = Conv1D(1, 1, activation='linear')(conv8)

    model = Model(inputs=[inputs], outputs=[outputs])
    return model


# Architecture with Conv1DTranspose layers
def ann128_tr(input_size=(128, 1)):
    inputs = Input(input_size, name='input')  # (None, 128, 1)

    # Encoder
    conv0 = Conv1D(32, 5, activation='relu', padding='same', name='encoder_conv00')(inputs)  # (None, 128, 32)
    conv0 = Conv1D(32, 5, activation='relu', padding='same', name='encoder_conv01')(conv0)  # (None, 128, 32)
    pool0 = MaxPooling1D(pool_size=2, name='encoder_maxpool0')(conv0)  # (None, 64, 32)

    conv1 = Conv1D(64, 5, activation='relu', padding='same', name='encoder_conv10')(pool0)  # (None, 64, 64)
    conv1 = Conv1D(64, 5, activation='relu', padding='same', name='encoder_conv11')(conv1)  # (None, 64, 64)
    pool1 = MaxPooling1D(pool_size=2, name='encoder_maxpool1')(conv1)  # (None, 32, 64)

    conv2 = Conv1D(128, 5, activation='relu', padding='same', name='encoder_conv20')(pool1)  # (None, 32, 128)
    conv2 = Conv1D(128, 5, activation='relu', padding='same', name='encoder_conv21')(conv2)  # (None, 32, 128)
    pool2 = MaxPooling1D(pool_size=2, name='encoder_maxpool2')(conv2)  # (None, 16, 128)

    conv3 = Conv1D(256, 5, activation='relu', padding='same', name='encoder_conv30')(pool2)  # (None, 16, 256)
    conv3 = Conv1D(256, 5, activation='relu', padding='same', name='encoder_conv31')(conv3)  # (None, 16, 256)
    pool3 = MaxPooling1D(pool_size=2, name='encoder_maxpool3')(conv3)  # (None, 8, 256)

    conv4 = Conv1D(512, 5, activation='relu', padding='same', name='encoder_conv40')(pool3)  # (None, 8, 512)
    conv4 = Conv1D(512, 5, activation='relu', padding='same', name='encoder_conv41')(conv4)  # (None, 8, 256)

    # Decoder
    up5 = Conv1DTranspose(256, 5, dilation_rate=2, name='decoder_up5')(conv4)  # (None, 16, 256)
    merge5 = concatenate([conv3, up5], axis=2, name='decoder_merge5')  # (None, 16, 512)
    conv5 = Conv1D(256, 5, activation='relu', padding='same', name='decoder_conv50')(merge5)  # (None, 16, 256)
    conv5 = Conv1D(256, 5, activation='relu', padding='same', name='decoder_conv51')(conv5)  # (None, 16, 256)

    up6 = Conv1DTranspose(128, 5, dilation_rate=4, name='decoder_up6')(conv5)  # (None, 32, 128)
    merge6 = concatenate([conv2, up6], axis=2, name='decoder_merge6')  # (None, 32, 256)
    conv6 = Conv1D(128, 5, activation='relu', padding='same', name='decoder_conv60')(merge6)  # (None, 32, 128)
    conv6 = Conv1D(128, 5, activation='relu', padding='same', name='decoder_conv61')(conv6)  # (None, 32, 128)

    up7 = Conv1DTranspose(64, 5, dilation_rate=8, name='decoder_up7')(conv6)  # (None, 64, 64)
    merge7 = concatenate([conv1, up7], axis=2, name='decoder_merge7')  # (None, 64, 128)
    conv7 = Conv1D(64, 5, activation='relu', padding='same', name='decoder_conv70')(merge7)  # (None, 64, 64)
    conv7 = Conv1D(64, 5, activation='relu', padding='same', name='decoder_conv71')(conv7)  # (None, 64, 64)

    up8 = Conv1DTranspose(32, 5, dilation_rate=16, name='decoder_up8')(conv7)  # (None, 128, 32)
    merge8 = concatenate([conv0, up8], axis=2, name='decoder_merge8')  # (None, 256, 32)
    conv8 = Conv1D(32, 5, activation='relu', padding='same', name='decoder_conv80')(merge8)  # (None, 128, 32)
    conv8 = Conv1D(32, 5, activation='relu', padding='same', name='decoder_conv81')(conv8)  # (None, 128, 32)

    outputs = Conv1D(1, 1, activation='linear', name='decoder_conv90')(conv8)  # (None, 128, 1)

    model = Model(inputs=[inputs], outputs=[outputs])
    return model


# Architecture with Conv1DTranspose and BatchNormalization
def ann128_trb(input_size=(128, 1)):
    inputs = Input(input_size, name='input')  # (None, 128, 1)

    # Encoder
    conv0 = Conv1D(32, 5, activation='relu', padding='same', name='encoder_conv00')(inputs)  # (None, 128, 32)
    batch0 = BatchNormalization(name='encoder_batch00')(conv0)
    conv0 = Conv1D(32, 5, activation='relu', padding='same', name='encoder_conv01')(batch0)  # (None, 128, 32)
    batch0 = BatchNormalization(name='encoder_batch01')(conv0)
    pool0 = MaxPooling1D(pool_size=2, name='encoder_maxpool0')(batch0)  # (None, 64, 32)

    conv1 = Conv1D(64, 5, activation='relu', padding='same', name='encoder_conv10')(pool0)  # (None, 64, 64)
    batch1 = BatchNormalization(name='encoder_batch10')(conv1)
    conv1 = Conv1D(64, 5, activation='relu', padding='same', name='encoder_conv11')(batch1)  # (None, 64, 64)
    batch1 = BatchNormalization(name='encoder_batch11')(conv1)
    pool1 = MaxPooling1D(pool_size=2, name='encoder_maxpool1')(batch1)  # (None, 32, 64)

    conv2 = Conv1D(128, 5, activation='relu', padding='same', name='encoder_conv20')(pool1)  # (None, 32, 128)
    batch2 = BatchNormalization(name='encoder_batch20')(conv2)
    conv2 = Conv1D(128, 5, activation='relu', padding='same', name='encoder_conv21')(batch2)  # (None, 32, 128)
    batch2 = BatchNormalization(name='encoder_batch21')(conv2)
    pool2 = MaxPooling1D(pool_size=2, name='encoder_maxpool2')(batch2)  # (None, 16, 128)

    conv3 = Conv1D(256, 5, activation='relu', padding='same', name='encoder_conv30')(pool2)  # (None, 16, 256)
    batch3 = BatchNormalization(name='encoder_batch30')(conv3)
    conv3 = Conv1D(256, 5, activation='relu', padding='same', name='encoder_conv31')(batch3)  # (None, 16, 256)
    batch3 = BatchNormalization(name='encoder_batch31')(conv3)
    pool3 = MaxPooling1D(pool_size=2, name='encoder_maxpool3')(batch3)  # (None, 8, 256)

    conv4 = Conv1D(512, 5, activation='relu', padding='same', name='encoder_conv40')(pool3)  # (None, 8, 512)
    batch4 = BatchNormalization(name='encoder_batch40')(conv4)
    conv4 = Conv1D(512, 5, activation='relu', padding='same', name='encoder_conv41')(batch4)  # (None, 8, 256)
    batch4 = BatchNormalization(name='encoder_batch41')(conv4)

    # Decoder
    up5 = Conv1DTranspose(256, 5, dilation_rate=2, name='decoder_up5')(batch4)  # (None, 16, 256)
    merge5 = concatenate([conv3, up5], axis=2, name='decoder_merge5')  # (None, 16, 512)
    conv5 = Conv1D(256, 5, activation='relu', padding='same', name='decoder_conv50')(merge5)  # (None, 16, 256)
    batch5 = BatchNormalization(name='decoder_batch50')(conv5)
    conv5 = Conv1D(256, 5, activation='relu', padding='same', name='decoder_conv51')(batch5)  # (None, 16, 256)
    batch5 = BatchNormalization(name='decoder_batch51')(conv5)

    up6 = Conv1DTranspose(128, 5, dilation_rate=4, name='decoder_up6')(batch5)  # (None, 32, 128)
    merge6 = concatenate([conv2, up6], axis=2, name='decoder_merge6')  # (None, 32, 256)
    conv6 = Conv1D(128, 5, activation='relu', padding='same', name='decoder_conv60')(merge6)  # (None, 32, 128)
    batch6 = BatchNormalization(name='decoder_batch60')(conv6)
    conv6 = Conv1D(128, 5, activation='relu', padding='same', name='decoder_conv61')(batch6)  # (None, 32, 128)
    batch6 = BatchNormalization(name='decoder_batch61')(conv6)

    up7 = Conv1DTranspose(64, 5, dilation_rate=8, name='decoder_up7')(batch6)  # (None, 64, 64)
    merge7 = concatenate([conv1, up7], axis=2, name='decoder_merge7')  # (None, 64, 128)
    conv7 = Conv1D(64, 5, activation='relu', padding='same', name='decoder_conv70')(merge7)  # (None, 64, 64)
    batch7 = BatchNormalization(name='decoder_batch70')(conv7)
    conv7 = Conv1D(64, 5, activation='relu', padding='same', name='decoder_conv71')(batch7)  # (None, 64, 64)
    batch7 = BatchNormalization(name='decoder_batch71')(conv7)

    up8 = Conv1DTranspose(32, 5, dilation_rate=16, name='decoder_up8')(batch7)  # (None, 128, 32)
    merge8 = concatenate([conv0, up8], axis=2, name='decoder_merge8')  # (None, 256, 32)
    conv8 = Conv1D(32, 5, activation='relu', padding='same', name='decoder_conv80')(merge8)  # (None, 128, 32)
    batch8 = BatchNormalization(name='decoder_batch80')(conv8)
    conv8 = Conv1D(32, 5, activation='relu', padding='same', name='decoder_conv81')(batch8)  # (None, 128, 32)
    batch8 = BatchNormalization(name='decoder_batch81')(conv8)

    outputs = Conv1D(1, 1, activation='linear', name='decoder_conv90')(batch8)  # (None, 128, 1)

    model = Model(inputs=[inputs], outputs=[outputs])
    return model


# Architecture with heart axis
def ann136():
    input_ecg = Input((128, 1), name='inputs_ecg')  # (None, 128, 1)
    input_ecg_permute = Permute((2, 1), input_shape=(128, 1))(input_ecg)
    input_axis = Input((8, 1), name='input_axis')
    input_axis_permute = Permute((2, 1), input_shape=(9, 1))(input_axis)
    concatenate_inputs = concatenate([input_ecg_permute, input_axis_permute], name='concatenate_inputs')
    inputs = Permute((2, 1), input_shape=(1, 137))(concatenate_inputs)

    # Encoder
    conv0 = Conv1D(32, 5, activation='relu', padding='same', name='encoder_conv00')(inputs)  # (None, 128, 32)
    conv0 = Conv1D(32, 5, activation='relu', padding='same', name='encoder_conv01')(conv0)  # (None, 128, 32)
    pool0 = MaxPooling1D(pool_size=2, name='encoder_maxpool0')(conv0)  # (None, 64, 32)

    conv1 = Conv1D(64, 5, activation='relu', padding='same', name='encoder_conv10')(pool0)  # (None, 64, 64)
    conv1 = Conv1D(64, 5, activation='relu', padding='same', name='encoder_conv11')(conv1)  # (None, 64, 64)
    pool1 = MaxPooling1D(pool_size=2, name='encoder_maxpool1')(conv1)  # (None, 32, 64)

    conv2 = Conv1D(128, 5, activation='relu', padding='same', name='encoder_conv20')(pool1)  # (None, 32, 128)
    conv2 = Conv1D(128, 5, activation='relu', padding='same', name='encoder_conv21')(conv2)  # (None, 32, 128)
    pool2 = MaxPooling1D(pool_size=2, name='encoder_maxpool2')(conv2)  # (None, 16, 128)

    conv3 = Conv1D(256, 5, activation='relu', padding='same', name='encoder_conv30')(pool2)  # (None, 16, 256)
    conv3 = Conv1D(256, 5, activation='relu', padding='same', name='encoder_conv31')(conv3)  # (None, 16, 256)
    pool3 = MaxPooling1D(pool_size=2, name='encoder_maxpool3')(conv3)  # (None, 8, 256)

    conv4 = Conv1D(512, 5, activation='relu', padding='same', name='encoder_conv40')(pool3)  # (None, 8, 512)
    conv4 = Conv1D(512, 5, activation='relu', padding='same', name='encoder_conv41')(conv4)  # (None, 8, 512)

    # Decoder
    up5 = UpSampling1D(size=2, name='decoder_up5')(conv4)  # (None, 16, 512)
    zp5 = ZeroPadding1D((0, 1), name='zero_padding_up5')(up5)
    merge5 = concatenate([conv3, zp5], axis=2, name='decoder_merge5')  # (None, 16, 768)
    conv5 = Conv1D(256, 5, activation='relu', padding='same', name='decoder_conv50')(merge5)  # (None, 16, 256)
    conv5 = Conv1D(256, 5, activation='relu', padding='same', name='decoder_conv51')(conv5)  # (None, 16, 256)

    up6 = UpSampling1D(size=2, name='decoder_up6')(conv5)  # (None, 32, 256)
    merge6 = concatenate([conv2, up6], axis=2, name='decoder_merge6')  # (None, 32, 384)
    conv6 = Conv1D(128, 5, activation='relu', padding='same', name='decoder_conv60')(merge6)  # (None, 32, 128)
    conv6 = Conv1D(128, 5, activation='relu', padding='same', name='decoder_conv61')(conv6)  # (None, 32, 128)

    up7 = UpSampling1D(size=2, name='decoder_up7')(conv6)  # (None, 64, 128)
    merge7 = concatenate([conv1, up7], axis=2, name='decoder_merge7')  # (None, 64, 192)
    conv7 = Conv1D(64, 5, activation='relu', padding='same', name='decoder_conv70')(merge7)  # (None, 64, 64)
    conv7 = Conv1D(64, 5, activation='relu', padding='same', name='decoder_conv71')(conv7)  # (None, 64, 64)

    up8 = UpSampling1D(size=2, name='decoder_up8')(conv7)  # (None, 128, 64)
    merge8 = concatenate([conv0, up8], axis=2, name='decoder_merge8')  # (None, 128, 96)
    conv8 = Conv1D(32, 5, activation='relu', padding='same', name='decoder_conv80')(merge8)  # (None, 128, 32)
    conv8 = Conv1D(32, 5, activation='relu', padding='same', name='decoder_conv81')(conv8)  # (None, 128, 32)

    outputs = Conv1D(1, 1, activation='linear')(conv8)  # (None, 137, 1)
    crop_outputs = Cropping1D((0, 8), name='crop_outputs')(outputs)

    model = Model(inputs=[input_ecg, input_axis], outputs=[crop_outputs])
    return model
