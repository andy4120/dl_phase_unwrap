from phase_to_gaze_model import *

class CNN_SQD_LSTM(PhaseGazeModel):
    def __init__(self, model_name, epochs=10, batch_size=32, learn_rate=0.001, lr_type="fixed", early_stop=True):
        """
        Constructor for model class
        @model_name: string; name of the model -either 'resnet' or 'vgg'
        @batch_size: int
        @epochs: int
        @learn_rate: float
        @lr_type: learning rates can be: 'fixed', 'cosine', 'plateau'
        @early_stop: boolean; whether to set early stopping or not
        """

        # call superclass' constructor
        PhaseGazeModel.__init__(self, model_name, epochs, batch_size, learn_rate, lr_type, early_stop)

    def _cnn_sqd_lstm_layer(self):
        input = Input((512, 512, 2))  # Input size changed to 512x512

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

        flattened = Flatten()(c12)  # Flatten the output of the last convolutional layer
        dense1 = Dense(128, activation='relu')(flattened)
        dense2 = Dense(64, activation='relu')(dense1)
        # dense3 = Dense(32, activation='relu')(dense2)
        # dense4 = Dense(16, activation='relu')(dense3)
        
        # Output vector (size depends on the task, here an example with size 10)
        output_vector = Dense(3, activation='linear')(dense2)  # Adjust the size of the output vector as needed

        self.model = models.Model(inputs=[input], outputs=[output_vector])
        
    def model_train(self):
        self._cnn_sqd_lstm_layer() # construct neural network layer
        self.model.summary() # print out summary

        # adam_v2.Adam(learning_rate=1e-1, clipvalue=1.0),
        self.model.compile(optimizer='adam', loss=self._vector_angle_loss, metrics=['accuracy'])

        self.model_history = self.model.fit(self.X_train, self.Y_train, epochs=self.epochs, verbose=True, \
                                            validation_data=(self.X_val, self.Y_val), callbacks=self._callbacks())
        
        self._train_validation_acc_loss_plot() # training and validation accuracy and loss plots
        self._test_accuracy_loss() # test accuracy and loss

if __name__ == '__main__':
    set_gpus()

    ''' Model object '''
    cnn_model = CNN_SQD_LSTM(model_name='phase_to_gaze_sqd_lstm_300', epochs=300)

    ''' Data load '''
    # load img data, split into Train/Validation/Test set
    data_folder = './dl_data_set/dl_deflec_eye/'
    input_filename_1 = 'img_9_norm.png'
    input_filename_2 = 'img_10_norm.png'
    cnn_model.training_data_img(data_folder, input_filename_1, input_filename_2)

    ''' Train '''
    cnn_model.model_train()

    ''' Real Dataset Prediction '''
    cnn_model._load_real_data(folder_path='./DL_data', data_length=20, degree=[0, 2, 4, 8, 6])
    cnn_model._predict_real_data()