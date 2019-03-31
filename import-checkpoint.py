import simplejson

import keras.backend as K
import keras.losses
from keras.models import load_model

keras.losses.lenient_cross_entropy = lambda x, y: K.mean(y)

net = load_model('checkpoint.hdf5')

with open('model.json', 'w') as file:
    file.write(simplejson.dumps(simplejson.loads(net.to_json()), indent=4))

net.save_weights('weights.h5')
