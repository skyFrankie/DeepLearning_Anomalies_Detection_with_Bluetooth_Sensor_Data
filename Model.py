import datetime
from config import *
from ScrapHolidayDate import getHoliday
from __future__ import print_function
import os
import pandas as pd
import numpy as np
import seaborn as sns
sns.set(color_codes=True)
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GroupShuffleSplit
import joblib
from keras.layers import Input, Dropout,Dense, LSTM, TimeDistributed,RepeatVector
from keras.models import Model
from keras import regularizers
from __future__ import print_function
from keras.utils.vis_utils import plot_model
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import Embedding
from keras.layers import Conv1D, GlobalMaxPooling1D,GlobalAveragePooling1D
from keras.layers import MaxPooling1D
from keras.datasets import imdb
from keras.layers import Reshape
from keras.layers import Input
from keras.layers import Flatten
from keras.layers import UpSampling1D
from keras.layers import Dropout
from keras.models import load_model
from keras.layers import concatenate


class Convolution_training():
    def __init__(self, route_num,weekday):
        self.route_num = route_num
        self.weekday = weekday
        self.training_data_processor = preprocess_training_data(route_num,weekday)
        tf.random.set_seed(31)
        self.autoencoder = self.Convolution_model()
        self.transformer = MinMaxScaler()

        self.input_1 = self.training_data_processor.train_set['Travel_time']
        self.input_2 = self.training_data_processor.train_set['Holiday']
        self.X_train_1 = self.transformer.fit_transform(np.array(self.input_1).reshape(-1, 1))
        self.X_train_1 = self.X_train_1.reshape(-1, 180, 1)
        self.X_train_2 = np.array([self.input_2.loc[self.input_2.index.to_list()[i]] for i in range(0,len(self.input_2.index),180)]).reshape(-1, 1, 1)

        self.input_3 = self.training_data_processor.test_set['Travel_time']
        self.input_4 = self.training_data_processor.test_set['Holiday']
        self.X_test_1 = self.transformer.transform(np.array(self.input_3).reshape(-1, 1))
        self.X_test_1 = self.X_test_1.reshape(-1, 180, 1)
        self.X_test_2 = np.array([self.input_4.loc[self.input_4.index.to_list()[i]] for i in range(0,len(self.input_4.index),180)]).reshape(-1, 1, 1)

    def Convolution_model(self):
        # ENCODER
        input_sig = Input(batch_shape=(None,180, 1))
        holiday = Input(batch_shape=(None,1,1))
        x = Conv1D(45, 3, padding='valid')(input_sig)
        x = Dropout(0.2)(x)
        x = MaxPooling1D(2)(x)
        x = Conv1D(23, 3, padding='valid')(x)
        x = Dropout(0.2)(x)
        x = MaxPooling1D(2)(x)
        flat = Flatten()(x)
        encoded = Dense(12)(flat)
        encoded = concatenate([encoded, Dense(1)(Flatten()(holiday))])
        encoded = Dense(10)(encoded)
        x_shape = tf.keras.backend.int_shape(x)[1:]
        decoder_input = Dense(np.prod(x_shape), activation='relu')(encoded)
        decoder_input = Reshape(x_shape)(decoder_input)

        #DECODER
        x2 = Conv1D(23,3, activation='relu',padding='valid')(decoder_input)
        x2 = Dropout(0.2)(x2)
        x2 = UpSampling1D(2)(x2)
        x2 = Conv1D(45,3, padding='valid')(x2)
        x2 = Dropout(0.2)(x2)
        upsamp = UpSampling1D(2)(x2)
        flat = Flatten()(upsamp)
        decoded = Dense(180)(flat)
        decoded = Reshape((180,1))(decoded)

        # Compile
        autoencoder = Model(inputs=[input_sig,holiday], outputs=decoded)
        autoencoder.compile(optimizer='adam', loss='mse', metrics=['accuracy'])
        #print(plot_model(autoencoder, show_shapes=True, to_file='1dconv_autoencoder_1.png'))
        return autoencoder

    def Train(self):
        nb_epochs = 80
        batch_size = 1
        callback = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=15)
        history = self.autoencoder.fit([self.X_train_1,self.X_train_2], self.X_train_1, epochs=nb_epochs, batch_size=batch_size,callbacks=[callback],validation_split=0.2).history
        fig, ax = plt.subplots(figsize=(14, 6), dpi=80)
        ax.plot(history['loss'], 'b', label='Train', linewidth=2)
        ax.plot(history['val_loss'], 'r', label='Validation', linewidth=2)
        ax.set_title('Model loss', fontsize=16)
        ax.set_ylabel('Loss(mse)')
        ax.set_xlabel('Epoch')
        ax.legend(loc='upper right')
        plt.show()


class preprocess_training_data(Convolution_training):
    """
    Gather all the target route data from different folder and add Holiday feature to the dataset
    """
    def __init__(self,route_num,weekday):
        self.route_num = route_num
        self.weekday = weekday
        self.target = self.gather_route_data()
        self.add_Holiday()
        self.train_set, self.test_set = self.train_test_split()

    def gather_route_data(self):
        route_data = []
#        for weekday in ["%s" % (datetime.date.today() + datetime.timedelta(days=x)).strftime("%a") for x in range(7)]:
        try:
            route_data.append(pd.read_csv(TRAINING_DATA_PATH/self.weekday/rf'route_{self.route_num}_{self.weekday}.csv'))
            return pd.concat(route_data, ignore_index=True)
        except:
            print('File not found!')
            print(route_data)
            return None

    def add_Holiday(self):
        """
        Use function getHoliday in ScrapHolidayDate.py to scrape Holiday date
        Add to target dataframe
        :return:
        """
        holiday_list = getHoliday()
        self.target['Date_temp'] =pd.to_datetime(self.target['Date'], dayfirst=True)
        self.target.sort_values(['Date_temp'],inplace=True)
        self.target['Date_temp'] = self.target['Date_temp'].dt.strftime('%d/%m/%Y')
        self.target['Holiday'] = self.target.apply(lambda x: 1 if x['Date_temp'] in holiday_list['Date_2017'].to_list() else 0, axis=1)
        self.target['Date_temp'] = self.target['Date_temp'].replace(self.target['Date_temp'].unique(),range(0,self.target['Date_temp'].nunique()))

    def shuffle_train_test_split(self):
        groups = np.array(self.target.Date_temp.to_list())
        gss = GroupShuffleSplit(n_splits=2, train_size=.75, random_state=42)
        set1, set2 = gss.split(self.target,groups = groups)
        train_set = self.target.loc[set1[0],:]
        test_set = self.target.loc[set1[1],:]
        return train_set.drop(['Date_temp','Day_of_week'],axis=1).set_index('Date'), test_set.drop(['Date_temp','Day_of_week'],axis=1).set_index('Date')

    def train_test_split(self):
        train_set = self.target[:5040]
        test_set = self.target[5040:]
        return train_set.drop(['Date_temp','Day_of_week'],axis=1).set_index('Date'), test_set.drop(['Date_temp','Day_of_week'],axis=1).set_index('Date')


    def get_target_data(self):
        return self.target

##########################################################################################################
Test = Convolution_training(2,'Sat')
Test.Train()
################################################################################################
X_pred =Test.autoencoder.predict([Test.X_train_1,Test.X_train_2])
X_pred = X_pred.reshape(len(Test.input_1),1)
X_pred = Test.transformer.inverse_transform(X_pred)
X_pred = pd.DataFrame(X_pred,columns= ['Travel_time_pred'])
X_pred.index = Test.input_1.index
compare = pd.concat([Test.input_1.rename('Travel_time_origin'),X_pred],axis=1)
compare.plot(subplots=True, figsize=(6, 6)); plt.legend(loc='best')
compare.plot(figsize=(6, 6)); plt.legend(loc='best')
#plt.savefig(str('/content/drive/My Drive/Model/Route'+str(jj)+'/'+Weekday[p]+'/'+str(p)+'/Reconstruct_result_'+str(jj)+'_'+str(p)+'.png'))
plt.show()

#########################################################################################################
scored = pd.DataFrame(index=Test.input_1.index)
scored['Loss_mse'] = np.mean(np.abs(np.array(X_pred) - np.array(Test.input_1).reshape(len(Test.input_1))), axis=1)
plt.figure(figsize=(15, 9), dpi=80)
plt.title('Loss Distribution', fontsize=16)
sns.distplot(scored['Loss_mse'], bins=1, kde=True, color='blue')
#plt.xlim([0, .2])
plt.show()
########################################################################################################



