from keras.models import Sequential
from keras.layers import Dense
from keras.models import model_from_json
import coremltools
import numpy
import os

# load json models

json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model.h5")
print("Loaded model from disk")


loaded_model.save(filepath='../data/cnn.model')
coreml_model = coremltools.converters.keras.convert('../data/cnn.model')
