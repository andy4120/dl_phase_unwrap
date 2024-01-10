import os
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.optimizers.optimizer_v2 import adam as adam_v2
from keras.layers import Input, Conv2D, BatchNormalization, Activation, MaxPooling2D, Conv2DTranspose, concatenate, Reshape, Permute, Bidirectional, LSTM
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import losses
from keras.models import Model
from sklearn.model_selection import train_test_split
from utils import *
from results import *

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
tf.compat.v1.enable_eager_execution()

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_virtual_device_configuration(gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=19000)])
    except RuntimeError as e:
        print(e)


def cnn_sqd_lstm_model():
    '''
    Defines the joint convolutional and spatial quad-directional LSTM network
    '''

    # input to the network
    input = Input((512, 512, 1))  # Input size changed to 512x512

    # encoder network
    c1 = Conv2D(filters=16, kernel_size=(3,3), kernel_initializer='he_normal', padding='same')(input)
    c1 = BatchNormalization()(c1)
    c1 = Activation('relu')(c1)
    p1 = MaxPooling2D()(c1)
    #p1 = AveragePooling2D()(c1)

    c2 = Conv2D(filters=32, kernel_size=(3,3), kernel_initializer='he_normal', padding='same')(p1)
    c2 = BatchNormalization()(c2)
    c2 = Activation('relu')(c2)
    p2 = MaxPooling2D()(c2)
    #p2 = AveragePooling2D()(c2)

    c3 = Conv2D(filters=64, kernel_size=(3,3), kernel_initializer='he_normal', padding='same')(p2)
    c3 = BatchNormalization()(c3)
    c3 = Activation('relu')(c3)
    p3 = MaxPooling2D()(c3)
    #p3 = AveragePooling2D()(c3)

    c4 = Conv2D(filters=128, kernel_size=(3,3), kernel_initializer='he_normal', padding='same')(p3)
    c4 = BatchNormalization()(c4)
    c4 = Activation('relu')(c4)
    p4 = MaxPooling2D()(c4)
    #p4 = AveragePooling2D()(c4)

    c5 = Conv2D(filters=256, kernel_size=(3,3), kernel_initializer='he_normal', padding='same')(p4)
    c5 = BatchNormalization()(c5)
    c5 = Activation('relu')(c5)
    p5 = MaxPooling2D()(c5)
    #p5 = AveragePooling2D()(c5)

    c6 = Conv2D(filters=512, kernel_size=(3,3), kernel_initializer='he_normal', padding='same')(p5)
    c6 = BatchNormalization()(c6)
    c6 = Activation('relu')(c6)
    p6 = MaxPooling2D()(c6)
    #p6 = AveragePooling2D()(c6)


    # SQD-LSTM Block
    x_hor_1 = Reshape((8 * 8, 512))(p6)
    x_ver_1 = Reshape((8 * 8, 512))(Permute((2, 1, 3))(p6))

    h_hor_1 = Bidirectional(LSTM(units=128, activation='tanh', return_sequences=True, go_backwards=False))(x_hor_1)
    h_ver_1 = Bidirectional(LSTM(units=128, activation='tanh', return_sequences=True, go_backwards=False))(x_ver_1)

    H_hor_1 = Reshape((8, 8, 256))(h_hor_1)
    H_ver_1 = Permute((2, 1, 3))(Reshape((8, 8, 256))(h_ver_1))

    c_hor_1 = Conv2D(filters=64, kernel_size=(3, 3),
                     kernel_initializer='he_normal', padding='same')(H_hor_1)
    c_ver_1 = Conv2D(filters=64, kernel_size=(3, 3),
                     kernel_initializer='he_normal', padding='same')(H_ver_1)

    H = concatenate([c_hor_1, c_ver_1])

    # decoder network
    u7 = Conv2DTranspose(512, (3, 3), strides=(2, 2), padding='same')(H)
    u7 = concatenate([u7, c6])
    c7 = Conv2D(filters=512, kernel_size=(3,3), kernel_initializer='he_normal', padding='same')(u7)
    c7 = BatchNormalization()(c7)
    c7 = Activation('relu')(c7)

    u8 = Conv2DTranspose(256, (3, 3), strides=(2, 2), padding='same')(c7)
    u8 = concatenate([u8, c5])
    c8 = Conv2D(filters=256, kernel_size=(3,3), kernel_initializer='he_normal', padding='same')(u8)
    c8 = BatchNormalization()(c8)
    c8 = Activation('relu')(c8)

    u9 = Conv2DTranspose(128, (3, 3), strides=(2, 2), padding='same')(c8)
    u9 = concatenate([u9, c4])
    c9 = Conv2D(filters=128, kernel_size=(3,3), kernel_initializer='he_normal', padding='same')(u9)
    c9 = BatchNormalization()(c9)
    c9 = Activation('relu')(c9)

    u10 = Conv2DTranspose(64, (3, 3), strides=(2, 2), padding='same')(c9)
    u10 = concatenate([u10, c3])
    c10 = Conv2D(filters=64, kernel_size=(3,3), kernel_initializer='he_normal', padding='same')(u10)
    c10 = BatchNormalization()(c10)
    c10 = Activation('relu')(c10)

    u11 = Conv2DTranspose(32, (3, 3), strides=(2, 2), padding='same')(c10)
    u11 = concatenate([u11, c2])
    c11 = Conv2D(filters=32, kernel_size=(3,3), kernel_initializer='he_normal', padding='same')(u11)
    c11 = BatchNormalization()(c11)
    c11 = Activation('relu')(c11)

    u12 = Conv2DTranspose(16, (3, 3), strides=(2, 2), padding='same')(c11)
    u12 = concatenate([u12, c1])
    c12 = Conv2D(filters=16, kernel_size=(3,3), kernel_initializer='he_normal', padding='same')(u12)
    #c12 = Conv2D(filters=32, kernel_size=(3,3), kernel_initializer='he_normal', padding='same')(u12)
    c12 = BatchNormalization()(c12)
    c12 = Activation('relu')(c12)

    ## output layer
    output = Conv2D(filters=1, kernel_size=(1, 1), padding='same', name='out1')(c12)
    output = Activation('linear')(output)

    model = Model(inputs=[input], outputs=[output])

    return model

def model_train_setup():
    model = cnn_sqd_lstm_model() # load neural network layer
    model.summary() # print out summary

    # compile model
    model.compile(
        optimizer=adam_v2.Adam(learning_rate=1e-3),
        loss=losses.mean_squared_error,
        metrics=['accuracy']
    )

    return model

def model_train(model_name: str, model):
    model_filepath = './model/' + model_name + '.h5' # model save path

    earlystopper = EarlyStopping(
        monitor='loss',
        patience=100,
        verbose=1
    )

    model_checkpoint = ModelCheckpoint(
        filepath=model_filepath,
        monitor='val_loss',  # Use validation loss for saving the best model
        verbose=1,
        save_best_only=True
    )

    history = model.fit(
        x=X_train.reshape(X_train.shape[0], 512, 512, 1),
        y=Y_train.reshape(Y_train.shape[0], 512, 512, 1),
        batch_size=6,
        epochs=500, #20000
        verbose=True,
        validation_data=(X_val.reshape(X_val.shape[0], 512, 512, 1), Y_val.reshape(Y_val.shape[0], 512, 512, 1)),
        callbacks=[model_checkpoint, earlystopper]
    )

    return model, history

def training_data_img(data_folder: str, input_filename: str, output_filename: str):
    # TODO: 1000 dataset splitted into Train:Validation:Test = 8:1:1

    X, Y = img_loader(data_folder, input_filename, output_filename)

    # Split the data into train, validation, and test sets
    X_train, X_temp, Y_train, Y_temp = train_test_split(X, Y, test_size=0.1, random_state=42)
    X_val, X_test, Y_val, Y_test = train_test_split(X_temp, Y_temp, test_size=0.1, random_state=42)

    return X_train, X_val, X_test, Y_train, Y_val, Y_test

def training_data_numpy(data_folder: str, numpy_filename: str, single_shot_filename: str):
    # TODO: 1000 dataset splitted into Train:Validation:Test = 8:1:1

    single_shot_data_numpy(data_folder, single_shot_filename) # convert single-shot input data into numpy first
    X, Y = numpy_loader(data_folder, numpy_filename) # load saved numpy arrays

    # Split the data into train, validation, and test sets
    X_train, X_temp, Y_train, Y_temp = train_test_split(X, Y, test_size=0.1, random_state=42)
    X_val, X_test, Y_val, Y_test = train_test_split(X_temp, Y_temp, test_size=0.1, random_state=42)

    return X_train, X_val, X_test, Y_train, Y_val, Y_test

def train_validation_acc_loss_plot(model_history, model_name: str) -> None:
    '''
    Training and validation accuracy and loss plot
    @model_history: history of the model from the training, to load accuracy and loss values
    @model_name: model name to save plots based on this label
    '''

    # extract the training and validation accuracy and loss
    train_acc = model_history.history['accuracy']
    val_acc = model_history.history['val_accuracy']
    train_loss = model_history.history['loss']
    val_loss = model_history.history['val_loss']

    # plot training and validation accuracy
    plt.figure(figsize=(8, 6))
    plt.plot(train_acc, label='Train Accuracy', color='blue', alpha=0.7)
    plt.plot(val_acc, label='Validation Accuracy', color='orange', alpha=0.7)
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig('./model/' + model_name + '_accuracy.png')

    # plot training and validation loss
    plt.figure(figsize=(8, 6))
    plt.plot(train_loss, label='Train Loss', color='blue', alpha=0.7)
    plt.plot(val_loss, label='Validation Loss', color='orange', alpha=0.7)
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('./model/' + model_name + '_loss.png')

    # plot training and validation loss in log scale
    plt.figure(figsize=(8, 6))
    plt.plot(train_loss, label='Train Loss', color='blue', alpha=0.7)
    plt.plot(val_loss, label='Validation Loss', color='orange', alpha=0.7)
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.yscale('log') # set y-axis to logarithmic scale
    plt.legend()
    plt.savefig('./model/' + model_name + '_loss_log.png')

def test_accuracy_loss(model, X_test: np.array, Y_test: np.array) -> None:
    '''
    Print test set accuracy and loss
    @model: trained model
    @X_test: test set input
    @Y_test: test set output (GT)
    '''

    # evaluate on the test set
    test_score = model.evaluate(X_test.reshape(X_test.shape[0], 512, 512, 1), Y_test.reshape(Y_test.shape[0], 512, 512, 1), batch_size=6)
    
    # print the test accuracy from the evaluation results
    print(f'Test Loss: {test_score[0]:.4f}')
    print(f'Test Accuracy: {test_score[1]:.4f}')

def predict_single_test_set(model_name: str, X_test: np.array, Y_test: np.array, set_index: int, \
                            crop: bool, crop_x: int, crop_y: int, pixel_length: int, direction: str) -> None:
    '''
    Predict one of the images from the test set
    @model_name: model name label for the location of the pre-trained model
    @X_test: input test set
    @Y_test: output (GT) test set
    @set_index: n-th image will be analyzed from the test set
    @plot_path: 3 images (input, GT, predicted) with colorbar (2pi) will be plotted in a single plot
    @crop: whether to crop the inner area of the phase or not for further analysis
    @crop_x: x-coordinate crop starting position
    @crop_y: y-coordinate crop starting position
    @pixel_length: width and length of the cropped area
    @direction: phase direction, either 'horizontal' or 'vertical'
    '''

    input_image = X_test[set_index].reshape(512, 512) # n-th X test set image as input
    ground_truth = Y_test[set_index].reshape(512, 512) # n-th Y test set image as ground truth

    input_for_prediction = input_image.reshape(1, 512, 512, 1) # Reshape the input image for prediction

    model_path = './model/' + model_name + '.h5' # pre-trained model path
    model = tf.keras.models.load_model(model_path) # load pre-trained model for prediction
    predicted_output = model.predict(input_for_prediction).reshape(512, 512) # Predict using the loaded model

    if crop: # if checking zoom-in version (small square portion within the profile)
        input_image = input_image[crop_y : (crop_y + pixel_length), crop_x : (crop_x + pixel_length)]
        ground_truth = ground_truth[crop_y : (crop_y + pixel_length), crop_x : (crop_x + pixel_length)]
        predicted_output = predicted_output[crop_y : (crop_y + pixel_length), crop_x : (crop_x + pixel_length)]

        crop_plot_path = './model/' + model_name + '_test_single_crop.png' # cropped profile plot save path
        gt_pred_profile_plot(gt_img=ground_truth, pred_img=predicted_output, direction=direction, index=(pixel_length / 2), plot_save_path=crop_plot_path)

    # Plot and save the images
    fig, axs = plt.subplots(1, 3, figsize=(12, 5))

    # Input Image
    img = axs[0].imshow(input_image, cmap='gray')
    fig.colorbar(img, ax=axs[0])
    axs[0].set_title('Input Image')
    axs[0].axis('off')

    # Ground Truth
    img = axs[1].imshow(ground_truth, cmap='gray')
    fig.colorbar(img, ax=axs[1])
    axs[1].set_title('Ground Truth')
    axs[1].axis('off')

    # Predicted Output
    img = axs[2].imshow(predicted_output, cmap='gray')
    fig.colorbar(img, ax=axs[2])
    axs[2].set_title('Predicted Output')
    axs[2].axis('off')

    fig.tight_layout()
    fig.savefig('./model/' + model_name + '_test_single.png')

def predict_single_real_data(model_name: str, real_img_path: str, real_crop_x: int, real_crop_y: int, direction: str, index: int) -> None:
    '''
    Predict single real image data as an input
    @model_name: model name label for the location of the pre-trained model
    @real_img_path: real input image
    @real_crop_x: real img crop starting x-coordinate position
    @real_crop_y: real img crop starting y-coordinate position
    @direction: phase direction, either 'horizontal' or 'vertical'
    @index: profile pixel location
    @plot_save_path: profile plot save location
    '''

    input_image = img_read_crop(real_img_path, real_crop_x, real_crop_y) # real image input data
    model_path = './model/' + model_name + '.h5' # pre-trained model path
    model = tf.keras.models.load_model(model_path) # load pre-trained model for prediction
    predicted_output = model.predict(input_image).reshape(512, 512) # predict using the pre-trained model
    pred_profile_plot(pred_img=predicted_output, direction=direction, index=index, plot_save_path='./model/' + model_name + '_pred_real.png') # plot profile

if __name__ == '__main__':
    '''
    Dataset Info
    Raw (0-2pi)
    - (1) Vertical: img_9.png
    - (2) Horizontal: img_10.png
    Normalized (0-255)
    - (1) Vertical: img_9_norm.png
    - (2) Horizontal: img_10_norm.png
    '''
    
    ''' Data Loading '''
    # load img data, split into Train/Validation/Test set
    data_folder = './dl_data_set/dl_deflec_eye/'
    input_filename = 'img_8.png'
    # output_filename = 'img_10.png'
    # X_train, X_val, X_test, Y_train, Y_val, Y_test = training_data_img(data_folder, input_filename, output_filename)
    numpy_filename = 'horizontal_norm.npy'
    X_train, X_val, X_test, Y_train, Y_val, Y_test = training_data_numpy(data_folder, numpy_filename, input_filename)

    ''' Training '''
    # model initialization & training
    model_name = 'horizontal_raw' # model name
    model = model_train_setup() # create and load the model
    model, model_history = model_train(model_name, model) # train the model
    
    # training and validation statistics
    train_validation_acc_loss_plot(model_history, model_name) # training and validation accuracy and loss plots

    # test statistics
    test_accuracy_loss(model, X_test, Y_test) # test accuracy and loss

    ''' Prediction '''
    model_name = 'horizontal_raw' # model name
    direction = 'horizontal'

    # predict one of the test set image and plot for visualization
    set_index = 0
    plot_path = './model/horizontal_raw_test_single.png'
    predict_single_test_set(model_name=model_name, X_test=X_test, Y_test=Y_test, set_index=set_index, \
                            crop=True, crop_x=50, crop_y=50, pixel_length=50, direction=direction)
    
    # predict real image as input (no GT)
    real_img = 'C1.png'
    predict_single_real_data(model_name=model_name, real_img_path=real_img, real_crop_x=250, real_crop_y=300, direction=direction, index=150)