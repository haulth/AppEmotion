from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import glob
import os
import glob
import numpy as np
from PIL import Image
from tensorflow.keras.utils import Sequence
from imgaug import augmenters as iaa
import numpy as np
from PIL import Image
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.layers import Input, Flatten, Dense, Dropout
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.metrics import SparseCategoricalAccuracy
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from tensorflow import keras
import tensorflow as tf
import numpy as np
from tensorflow.compat.v1.keras import backend



list_label = ['Angry', 'Disgust','Fear','Happy','Neutral','Sad','Surprise']
label_map = {item: idx for idx, item in enumerate(list_label)}
config = tf.compat.v1.ConfigProto(
device_count={'GPU': 1},
intra_op_parallelism_threads=1,
allow_soft_placement=True
)

config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 0.6

session = tf.compat.v1.Session(config=config)

backend.set_session(session)
class DataLoader(Sequence):
    def __init__(self, batch_size, dataset_path, label_map):
      self.batch_size = batch_size
      self.list_filenames = glob.glob(os.path.join(dataset_path, '*/*.*'))
      self.label_map = label_map

      self.indices = np.random.permutation(len(self.list_filenames))

    def __len__(self):
      return int(len(self.list_filenames) / self.batch_size)

    def __getitem__(self, idx):
      list_np_image = []
      list_labels = []
      # for tu 0 den 32
      for idx_batch in range(self.batch_size):
        idx_dataset = idx * self.batch_size + idx_batch
        idx_filename = self.indices[idx_dataset]
        filename = self.list_filenames[idx_filename]

        
        np_image = np.array(Image.open(filename).convert('RGB').resize([96, 96]))
        list_np_image.append(np_image)

        original_label = filename.split('/')[-2]
        label = self.label_map[original_label]
        
        list_labels.append(label)
      
      batch_images = np.array(list_np_image)
        
      batch_labels = np.array(list_labels)

      return batch_images, batch_labels

    def on_epoch_end(self):
        self.indices = np.random.permutation(len(self.list_filenames))
class Classifier:
    def __init__(self):
        self.model = None
        self.list_labels = ['Angry', 'Disgust','Fear','Happy','Neutral','Sad','Surprise']
        self.label_map = {item: idx for idx, item in enumerate(self.list_labels)}

    def build_model(self):
        input_layer = Input(shape=[96, 96, 3])
        preprocess_layer = preprocess_input(input_layer)

        backbone = ResNet50(input_shape=[96, 96, 3], include_top=False, weights = 'imagenet')
        backbone_output_layer = backbone(preprocess_layer)

        for layer in backbone.layers[:-3]:
          layer.trainable = False

        flatten_layer = Flatten()(backbone_output_layer)
        output_layer = Dense(10, activation='softmax')(flatten_layer)

        model = Model(input_layer, output_layer)
        model.summary()

        loss = SparseCategoricalCrossentropy()
        optimizer = Adam(learning_rate=0.0001)
        metric = SparseCategoricalAccuracy()
        
        model.compile(loss=loss, optimizer=optimizer, metrics=[metric])
        self.model = model

    def load_model(self):

        self.model = load_model('models/emotionbest.h5')
        self.model._make_predict_function() # fix bug

    def save_model(self):
        pass

    def train(self, train_path, valid_path):
        train_generator = DataLoader(64, train_path, self.label_map)
        valid_generator = DataLoader(64, train_path, self.label_map)

        tensorboard = TensorBoard(log_dir="/content/drive/MyDrive/xongxoa/Emotion-ResNet50/Graph")

        call_backs = [
            EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10),
            ModelCheckpoint('models/emotion.h5',monitor = 'val_loss',save_best_only = True, verbose = 1),
            TensorBoard(log_dir='/content/drive/MyDrive/xongxoa/Emotion-ResNet50/Graph', histogram_freq=0, write_graph=True, write_images=True)
        ]
        self.model.fit(train_generator, validation_data = valid_generator, epochs = 200, callbacks=[call_backs, tensorboard])

    def predict(self, image):
        """
        :param image: a PIL image with arbitrary size
        :return: a string ('Angry', 'Disgust','Fear','Happy','Neutral','Sad','Surprise')
        """
        try:
            with session.as_default():
                with session.graph.as_default():
                    #image = image.resize([96, 96])
                    np_image = np.array(image)
                    batch_images = np.array([np_image])
                    y_prob_batch = self.model.predict(batch_images)
                    y_pred_batch = np.argmax(y_prob_batch, axis = 1)
                    y_predict = y_pred_batch[0]
                    # print("Kết quả mô hình dự đoán là:",self.list_labels[y_predict])
                    
                    return self.list_labels[y_predict]
        except :
            pass
