from phase_to_gaze_model import *

class UNet(PhaseGazeModel):
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

    def _u_net_layer(self):
        inputs = Input(shape=(512, 512, 2))

        # Contracting Path (Encoder)
        c1 = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
        c1 = Conv2D(32, (3, 3), activation='relu', padding='same')(c1)
        p1 = MaxPooling2D((2, 2))(c1)

        c2 = Conv2D(64, (3, 3), activation='relu', padding='same')(p1)
        c2 = Conv2D(64, (3, 3), activation='relu', padding='same')(c2)
        p2 = MaxPooling2D((2, 2))(c2)

        c3 = Conv2D(128, (3, 3), activation='relu', padding='same')(p2)
        c3 = Conv2D(128, (3, 3), activation='relu', padding='same')(c3)
        p3 = MaxPooling2D((2, 2))(c3)

        c4 = Conv2D(256, (3, 3), activation='relu', padding='same')(p3)
        c4 = Conv2D(256, (3, 3), activation='relu', padding='same')(c4)
        p4 = MaxPooling2D((2, 2))(c4)

        # Bottleneck
        c5 = Conv2D(512, (3, 3), activation='relu', padding='same')(p4)
        c5 = Conv2D(512, (3, 3), activation='relu', padding='same')(c5)

        # Expansive Path (Decoder) - omitting the up-convolutions and concatenations since we are predicting a vector
        # Flatten and pass through dense layers to predict the surface normal vector
        f6 = Flatten()(c5)
        d6 = Dense(256, activation='relu')(f6)
        d7 = Dense(128, activation='relu')(d6)
        
        # Output layer - 3 units for the surface normal vector
        outputs = Dense(3, activation='linear')(d7)  # 'linear' activation for regression

        self.model = models.Model(inputs=[inputs], outputs=[outputs])
        
    def model_train(self):
        self._u_net_layer() # construct neural network layer
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
    cnn_model = UNet(model_name='phase_to_gaze_unet_300', epochs=300)

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