from phase_to_gaze_model import *

class CNN(PhaseGazeModel):
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

    def _cnn_layer(self):
        self.model = models.Sequential([
            # Input layer - specify input shape (height, width, channels)
            Input(shape=(512, 512, 2)),
            
            # Convolutional layers
            Conv2D(32, (3, 3), activation='relu', padding='same'),
            MaxPooling2D((2, 2)),
            Conv2D(64, (3, 3), activation='relu', padding='same'),
            MaxPooling2D((2, 2)),
            Conv2D(128, (3, 3), activation='relu', padding='same'),
            MaxPooling2D((2, 2)),
            
            # Flattening the output of the conv layers to feed into the dense layers
            Flatten(),
            
            # Dense layers for prediction
            Dense(128, activation='relu'),
            Dense(64, activation='relu'),

            # dropout layer
            # Dropout(0.3),
            
            # Output layer - 3 units for the surface normal vector
            Dense(3, activation='linear')  # 'linear' activation for regression
        ])
        
    def model_train(self):
        self._cnn_layer() # construct neural network layer
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
    cnn_model = CNN(model_name='phase_to_gaze_cnn_no_200', epochs=200)

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