import os
import matplotlib.pyplot as plt
import numpy as np
import cv2
import tensorflow as tf
from keras import models
from keras.layers import *
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, CSVLogger
# from keras.optimizers.optimizer_v2 import adam as adam_v2 # Linux
from keras.optimizer_v2 import adam as adam_v2 # Windows
from sklearn.model_selection import train_test_split
from utils import *
from results import *
import json

def set_gpus():
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    tf.compat.v1.enable_eager_execution()

    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            tf.config.experimental.set_virtual_device_configuration(gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=19000)])
        except RuntimeError as e:
            print(e)

class PhaseGazeModel:
    def __init__(self, model_name, epochs=10, batch_size=32, learn_rate=0.001, lr_type="fixed", early_stop=True):
        """
        Constructor for model class
        @model_name: string; name of the model
        @batch_size: int, by default 32
        @epochs: int
        @learn_rate: double
        @lr_type: learning rates can be: 'fixed', 'cosine', 'plateau'
        @early_stop: boolean; whether to set early stopping or not
        @notes: string; notes that should be used when saving array and plots
        """

        self.model_name = model_name
        self.epochs = epochs
        self.batch_size = batch_size
        self.learn_rate = learn_rate
        self.lr_type = lr_type
        self.early_stop = early_stop

        self.model = None
        self.model_history = None
        self.X_train, self.X_val, self.X_test, self.Y_train, self.Y_val, self.Y_test = None, None, None, None, None, None
        self.X_real = None

    def training_data_img(self, data_folder: str, input_filename_1: str, input_filename_2: str):
        '''
        1000 dataset splitted into Train:Validation:Test = 8:1:1
        '''
        # TODO: file loading depends on the file format

        with open(os.path.join(data_folder, 'config.json'), "r") as handler: data_dict = json.load(handler)

        file_path_real_gaze_pairs = []
        for frame in data_dict['frames']:
            # file_path = frame['file_path']
            real_gaze = frame['real_gaze'][0]
            file_path_real_gaze_pairs.append(real_gaze)
            # file_path_real_gaze_pairs.append((file_path, real_gaze))

        Y = np.array(file_path_real_gaze_pairs)
        X = phase_pair_img_loader(os.path.join(data_folder, 'imgs'), input_filename_1, input_filename_2)

        # Split the data into train, validation, and test sets
        self.X_train, X_temp, self.Y_train, Y_temp = train_test_split(X, Y, test_size=0.1, random_state=42)
        self.X_val, self.X_test, self.Y_val, self.Y_test = train_test_split(X_temp, Y_temp, test_size=1/9, random_state=42)

    def _callbacks(self):
        model_checkpoint = ModelCheckpoint(
            filepath='./model/' + self.model_name + '.h5',
            monitor='val_loss',  # Use validation loss for saving the best model
            verbose=1,
            save_weights_only=False,
            save_best_only=True
        )
        
        csv_logger = CSVLogger('./model/' + self.model_name + '.csv')  # csv logger

        callbacks = [model_checkpoint, csv_logger]

        if (self.lr_type == 'plateau'):
            reduce_lr = ReduceLROnPlateau(monitor='val_accuracy', factor=.01, patience=3, min_lr=1e-5)  # reduce lr
            callbacks.append(reduce_lr)

        if (self.early_stop == True):
            early_stop = EarlyStopping(monitor='val_loss', patience=50, verbose=1)
            callbacks.append(early_stop)

        return callbacks

    def _vector_angle_loss(self, y_true, y_pred):
        # Normalize the vectors to unit vectors
        y_true_normalized = tf.nn.l2_normalize(y_true, axis=-1)
        y_pred_normalized = tf.nn.l2_normalize(y_pred, axis=-1)
        
        # Calculate the cosine similarity
        cosine_similarity = tf.reduce_sum(tf.multiply(y_true_normalized, y_pred_normalized), axis=-1)
        
        # Clip the cosine similarity values to avoid NaNs during backpropagation
        epsilon = 1e-7
        cosine_similarity = tf.clip_by_value(cosine_similarity, -1.0 + epsilon, 1.0 - epsilon)
        cosine_similarity_checked = tf.debugging.check_numerics(cosine_similarity, 'cosine_similarity') # NaN occurrence debug

        # Calculate the angle in radians and then convert to degrees
        angle_radians = tf.acos(cosine_similarity_checked)
        angle_degrees = angle_radians * 180 / tf.constant(np.pi)
        
        # Return the mean angle error in degrees
        return tf.reduce_mean(angle_degrees)
    
    def _train_validation_acc_loss_plot(self) -> None:
        '''
        Training and validation Mean Absolute Error and loss plot
        '''

        # extract the training and validation Mean Absolute Error and loss
        train_acc = self.model_history.history['accuracy']
        val_acc = self.model_history.history['val_accuracy']
        train_loss = self.model_history.history['loss']
        val_loss = self.model_history.history['val_loss']

        # plot training and validation accuracy
        plt.figure(figsize=(8, 6))
        plt.plot(train_acc, label='Train Accuracy', color='blue', alpha=0.7)
        plt.plot(val_acc, label='Validation Accuracy', color='orange', alpha=0.7)
        plt.title('Training and Validation Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.savefig('./model/' + self.model_name + '_accuracy.png')

        # plot training and validation loss
        plt.figure(figsize=(8, 6))
        plt.plot(train_loss, label='Train Loss', color='blue', alpha=0.7)
        plt.plot(val_loss, label='Validation Loss', color='orange', alpha=0.7)
        plt.title('Training and Validation Loss (Raw)')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig('./model/' + self.model_name + '_loss.png')

        # plot training and validation loss in log scale
        plt.figure(figsize=(8, 6))
        plt.plot(train_loss, label='Train Loss', color='blue', alpha=0.7)
        plt.plot(val_loss, label='Validation Loss', color='orange', alpha=0.7)
        plt.title('Training and Validation Loss (log-scale)')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.yscale('log') # set y-axis to logarithmic scale
        plt.legend()
        plt.savefig('./model/' + self.model_name + '_loss_log.png')

    def _test_accuracy_loss(self) -> None:
        '''
        Print test set accuracy and loss
        '''

        # evaluate on the test set
        test_score = self.model.evaluate(self.X_test.reshape(self.X_test.shape[0], 512, 512, 2), self.Y_test.reshape(self.Y_test.shape[0], 3, 1))
        
        # print the test accuracy from the evaluation results
        print(f'Test Loss: {test_score[0]:.4f}')
        print(f'Test Accuracy: {test_score[1]:.4f}')

        content = "Test Loss, Test Accuracy\n" + str(test_score[0]) + ', ' + str(test_score[1])
        with open('./model/' + self.model_name + '_test_stat.csv', 'w') as file: file.write(content)

    def _load_real_data(self, folder_path: str, data_length: int, degree: list=[0, 2, 4, 8, 6]) -> None:
        im_list = []
        for i in range(data_length):
            for d in degree:
                data_folder = os.path.join(folder_path, str(d) + '_degree')
                vertical_path = os.path.join(data_folder, 'phase_v')
                horizontal_path = os.path.join(data_folder, 'phase_h')
                vertical_path = os.path.join(vertical_path, str(i + 1) + '.png')
                horizontal_path = os.path.join(horizontal_path, str(i + 1) + '.png')
                im_list.append([vertical_path, horizontal_path])

        X = []
        for i in im_list:
            input_image_1 = img_read_crop(i[0], 450, 300, False)
            input_image_2 = img_read_crop(i[1], 450, 300, False)

            # Stack the two images along the third dimension (channel dimension) and append to X
            X.append(np.dstack((input_image_1, input_image_2)))

        self.X_real = np.array(X) # Convert the list of image data arrays to a single NumPy array

    def _predict_real_data(self, model_path=None, degree: list=[0, 2, 4, 8, 6]) -> None:
        if model_path is not None:
            self.model = tf.keras.models.load_model(model_path, custom_objects={'_vector_angle_loss': self._vector_angle_loss}) # load pre-trained model

        predictions = self.model.predict(self.X_real)

        # TODO: index automate
        content = "Data Index, angle_neg_4, angle_neg_2, angle_0, angle_pos_2, angle_pos_4, ERR_neg_4, ERR_neg_2, ERR_0, ERR_pos_2, ERR_pos_4, RMSE\n"
        relative_degrees = [-4., -2., 0., 2., 4.]
        
        data_idx = 1
        for i in range(predictions.shape[0] // len(degree)):
            sub_prediction = predictions[i : i + len(degree), :]
            middle_pred = sub_prediction[len(sub_prediction) // 2, :]
            content += str(data_idx) + ', '
            
            before_zero = True
            angle_pred = []

            for j in sub_prediction:
                angle = self._tan_vector_angle(middle_pred, j)
                if angle == 0: before_zero = False
                if before_zero: angle *= -1
                content += str(angle) + ', '

                angle_pred.append(angle)

            err = [a - b for a, b in zip(relative_degrees, angle_pred)]
            for j in err:
                content += str(j) + ', '
            
            mse = np.mean((np.array(relative_degrees) - np.array(angle_pred)) ** 2)
            rmse = np.sqrt(mse)
            content += str(rmse) + '\n'

            data_idx += 1

        with open('./model/' + self.model_name + '_real_data_prediction.csv', 'w') as file: file.write(content)

    def _tan_vector_angle(self, vector1, vector2):
        # Calculate the angle in radians between the two vectors
        angle_rad = np.arctan2(np.linalg.norm(np.cross(vector1, vector2)), np.dot(vector1, vector2))

        # Convert the angle to degrees
        angle_deg = np.degrees(angle_rad)

        return angle_deg
    
    def _cos_vector_angle(self, vector1, vector2):
        dot_product = np.dot(vector1, vector2) # dot product of vectors

        # Calculate the magnitudes of each vector
        magnitude1 = np.linalg.norm(vector1)
        magnitude2 = np.linalg.norm(vector2)

        cosine_similarity = dot_product / (magnitude1 * magnitude2) # cos angle between vectors

        angle_radians = np.arccos(cosine_similarity) # angle between vectors in radians
        angle_degrees = np.degrees(angle_radians) # radian to degrees

        return angle_degrees