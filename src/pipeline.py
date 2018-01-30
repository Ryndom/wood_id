import numpy as np
import pandas as pd
from tempfile import TemporaryFile
from PIL import Image
import glob
import matplotlib.pyplot as plt
# ---------------------------
# ------ MODELING -----------
from keras.preprocessing.image import array_to_img, \
    img_to_array, load_img

from keras.utils import np_utils
from sklearn.model_selection import train_test_split
tts = train_test_split
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers.convolutional import Conv2D

from keras import backend as K
# reporting
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn import metrics
cm = confusion_matrix
cr = classification_report

# Gets lists of files in the image folder

def get_file_list(location = '../images/*.*'):
    return sorted(glob.glob(location))
# files in /images folder
filelist_images = get_file_list()

# save filelist_images to done.csv
def save_done(filelist):
    done_files = np.array(filelist_images)
    np.savetxt('../images/done/done.csv',
           done_files,
           delimiter=",",
           fmt='%s')

# open done.csv file and save as done list
def open_done(location = '../images/done/done.csv'):
    completed = pd.read_csv(location, header = None)
    is_done = np.array(completed)
    done = []
    length = len(is_done)
    for i in range(length):
        done.append(is_done[i][0])
    return done



# Function called to compare files to those that have already been processed
def check_for_new():
    '''
    function to look at /images folder and see
    if new files have been added
    '''
    if filelist_images == done:
        return 'no action needed'
    else:
        print('need to update database')
        pass

# Converts to gray scale
def convert_to_gs(array):
    # define RGB conversion
    red = .299
    green = .587
    blue = round(1-(red+green), 3)
    gs = np.array([red, green, blue]).T
    return array.dot(gs)

'''
create a dictionary that has the following info:
dictionary is created during loop through all files
after file is converted to npy
file: example  red_oak_v07 - unique name
height: example array = np.array(img); array.shape[0]
width: example array = np.array(img); array.shape[1]
'''


def make_blanks(filelist_images, size = 180, border = 5):
    # returns zero matrix for X and y and file_indexing
    # ------------ FIXED FOR ALL FILES -------------
    size = 180 # size of image
    border = 5 # remove any weirdness at egde

    # Loop through each file
    total_rows = 0
    file_indexing = [[]]
    images_dict = {}
    for file in filelist_images:
        image = Image.open(file)
        array = np.array(image)
        array = convert_to_gs(array)
        # array in now grey scale
        height = array.shape[0]
        width = array.shape[1]
        steps_h = int((height-2*border)/(size/2))
        steps_w = int((width-2*border)/(size/2))
        num_slices = (steps_w-1) * (steps_h-1)
        num_rows = num_slices * 8
        file.split('ges/')[1]
        stuff = [file.split('ges/')[1], file, num_rows, file.split('ges/')[1].split('_v')[0]]
        file_indexing.append(stuff)
        total_rows += num_rows

        if [] in file_indexing:
            file_indexing.remove([])



    X = np.zeros((total_rows, size, size))
    y = np.zeros((total_rows, 1))
    print('X shape: ', X.shape)
    print('y shape: ', y.shape)
    return X, y, file_indexing

def get_catagories(file_indexing):
    # =======================================================
    '''
    get number of catagories and
    list of woods being looked at
    '''
    # returns wood_index_map, maps number to woods
    # return wood_names_map, maps wood to number
    names = []
    for row in range(len(file_indexing)):
        names.append(file_indexing[row][3])
    unique_names = set(names)
    unique_names = list(unique_names)
    wood_names_map = {}

    for i in range(len(unique_names)):
        # print(unique_names[i])
        wood_names_map[unique_names[i]] = i

    wood_index_map = {v: k for k, v in wood_names_map.items()}
    return wood_index_map, wood_names_map, unique_names

def build_X_and_y(X, y, file_indexing, wood_names_map, size = 180, border = 5):
    # =======================================================
    # walk through file indexing files, and add files to X and y
    at_row = 0
    for i in range(len(file_indexing)):
        file_name = file_indexing[i][0]
        file = file_indexing[i][1]
        file_num_rows = file_indexing[i][2]
        wood_name = file_indexing[i][3]
        wood_number = int(wood_names_map[file_indexing[i][3]])
        image = Image.open(file)
        array = np.array(image)
        array = convert_to_gs(array)

        # array in now grey scale
        height = array.shape[0]
        width = array.shape[1]
        steps_h = int((height-2*border)/(size/2))
        steps_w = int((width-2*border)/(size/2))

        for w in range(steps_w):
            for h in range(steps_h):
                begin_w = int(w*(size/2)) + border
                end_w = begin_w + size
                begin_h = int(h*(size/2)) + border
                end_h = begin_h + size
                if end_w <= width and end_h <= height and at_row < X.shape[0]:
                    area = (begin_w, begin_h, end_w, end_h)
                    X[at_row] = array[begin_h:end_h, begin_w:end_w]

                    cropped_imgA = X[at_row]
                    y[at_row] = wood_number

                    X[at_row + 1] = np.rot90(cropped_imgA)
                    cropped_imgB = np.rot90(cropped_imgA)
                    y[at_row + 1] = wood_number

                    X[at_row + 2] = np.rot90(cropped_imgB)
                    cropped_imgC = np.rot90(cropped_imgB)
                    y[at_row + 2] = wood_number

                    X[at_row + 3] = np.rot90(cropped_imgC)
                    cropped_imgD = np.rot90(cropped_imgC)
                    y[at_row + 3] = wood_number

                    X[at_row + 4] = np.fliplr(cropped_imgA)
                    y[at_row + 4] = wood_number
                    X[at_row + 5] = np.fliplr(cropped_imgB)
                    y[at_row + 5] = wood_number
                    X[at_row + 6] = np.fliplr(cropped_imgC)
                    y[at_row + 6] = wood_number
                    X[at_row + 7] = np.fliplr(cropped_imgD)
                    y[at_row + 7] = wood_number
                    at_row = at_row + 8


                print("I'm making X and y")
    return X, y

def build_model(X, y, size, un):
    # returns X_train, X_test, y_train, y_test and model
    # also returns normalized Y_train and Y_test
    X_train, X_test, y_train, y_test = tts(
        X, y, test_size = .2)
    nb_classes = len(un)
    # convert class vectors
    Y_train = np_utils.to_categorical(y_train, nb_classes)
    Y_test = np_utils.to_categorical(y_test, nb_classes)

    # set up initial parameters of model

    batch_size = 80 # number of training samples used at a time to update the weights
    nb_epoch = 2
    # number of passes, now at 2 only for testing, reset for 5 or 6

    # input image dimensions
    img_rows, img_cols = size, size
    input_shape = (img_rows, img_cols, 1)
    # number of convolutional filters to use
    nb_filters = 12
    # size of pooling area for max pooling
    pool_size = (2, 2) # decreases image size, and helps to avoid overfitting
    # convolution kernel size
    kernel_size = (4, 4) # slides over image to learn features
    # reshape image for Keras, note that image_dim_ordering set in ~.keras/keras.json

    if K.image_dim_ordering() == 'th':
        X_train = X_train.reshape(X_train.shape[0], 1, img_rows, img_cols)
        X_test = X_test.reshape(X_test.shape[0], 1, img_rows, img_cols)
        input_shape = (1, img_rows, img_cols)
    else:
        X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)
        X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)
        input_shape = (img_rows, img_cols, 1)

    #
    # Normalization
    X_train = X_train.astype('float32') # data was uint8 [0-255]
    X_test = X_test.astype('float32')  # data was uint8 [0-255]
    X_train /= 255 # normalizing (scaling from 0 to 1)
    X_test /= 255  # normalizing (scaling from 0 to 1)

    # build model..........



    model = Sequential()
    model.add(Conv2D(nb_filters, (kernel_size[0], kernel_size[1]),
                        padding='valid',
                        input_shape=input_shape))
    #first conv. layer


    model.add(Activation('tanh'))
    print('Original Activation Level hyperbolic tan', )
    model.add(Conv2D(nb_filters, (kernel_size[0], kernel_size[1])))

    model.add(Activation('tanh'))
    print('Model after first Convulution ', model.output_shape)

    # decreases size, helps prevent overfitting
    model.add(MaxPooling2D(pool_size=pool_size))
    # zeros out some fraction of inputs, helps prevent overfitting
    model.add(Dropout(0.5))
    print('Model after first MaxPooling ', model.output_shape)

    # 2nd conv layer
    model.add(Activation('relu'))
    print('2nd Convulution Level relu', )
    model.add(Conv2D(nb_filters, (kernel_size[0], kernel_size[1])))
    model.add(Activation('relu'))
    print('Model after second Convulution ', model.output_shape)


    model.add(MaxPooling2D(pool_size=pool_size))
    model.add(Dropout(0.5))
    print('Model after first MaxPooling ', model.output_shape)



    # Flatten
    model.add(Flatten())
    print('Model flattened out to ', model.output_shape)

    # Layer 1
    model.add(Dense(32))
    model.add(Activation('relu'))
    print('First Dense Layer 32 ', model.output_shape)
    # Layer 2
    model.add(Dense(64))
    model.add(Activation('relu'))
    print('2nd Dense Layer, relu ', model.output_shape)
    # Layer 3
    model.add(Dense(128))
    model.add(Activation('relu'))
    print('3nd Dense Layer, relu ', model.output_shape)

    # Dropout at .5
    model.add(Dropout(0.5))

    # Dense layer matching number catagories
    model.add(Dense(nb_classes))
    model.add(Activation('softmax'))
    print('Last Dense Layer - number of classes, softmax ', model.output_shape)

    model.compile(loss='categorical_crossentropy',
                optimizer='adam',
                metrics=['accuracy'])
    return model, X_train, X_test, y_train, y_test, Y_train, Y_test, batch_size, nb_epoch


def get_fit_pred(model, X_train, X_test, Y_train, Y_test, batch_size, nb_epoch):
    # during fit process watch train and test error simultaneously
    cnn_fit = model.fit(X_train, Y_train, batch_size=batch_size, epochs=nb_epoch,
              verbose=1, validation_data=(X_test, Y_test))
    cnn_pred = model.predict(X_test)
    return cnn_fit, cnn_pred


def get_corrected_prediction(cnn_pred):
    cnn_pred_1 = cnn_pred.argmax(axis=-1)
    return cnn_pred_1

def reports(model, wood_index_map, X_test, Y_test, cnn_pred_1):
    score = model.evaluate(X_test, Y_test, verbose=0)

    print('classification report \n')
    print(cr(y_test, cnn_pred_1))
    print('confusion matrix\n')
    print(cm(y_test, cnn_pred_1))
    for i in range(len(wood_index_map)):
        print(i,'--', wood_index_map[i])


    print('\n')
    print('Test score:', score[0])
    print('Test accuracy:', score[1])

if __name__ == '__main__':
    # filelist_images = l
    l = get_file_list()
    # define global variables, size and border
    size = 180 # square size of all processed images
    border = 5 # border to remove weirdness on outside of image
    blank_X, blank_y, fi = make_blanks(l, size, border)
    # fi is file_indexing file
    # un is unique_names
    wood_index_map, wood_names_map, un = get_catagories(fi)
    X, y = build_X_and_y(blank_X, blank_y, fi, wood_names_map, size, border)
    model, X_train, X_test, y_train, y_test, Y_train, Y_test, batch_size, nb_epoch = build_model(X, y, size, un)

    cnn_fit, cnn_pred = get_fit_pred(model, X_train, X_test, Y_train, Y_test, batch_size, nb_epoch)
    model.save(filepath = '../data/cnn.model')
    cnn_pred_1 = get_corrected_prediction(cnn_pred)
    reports(model, wood_index_map, X_test, Y_test, cnn_pred_1)

    # export model to json file
    from keras.models import model_from_json
    model_json = model.to_json()
    with open("model.json", "w") as json_file:
        json_file.write(model_json)

    model.save_weights('model.h5')
