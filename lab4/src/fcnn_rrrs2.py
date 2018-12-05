import mxnet as mx
import numpy as np
import logging
import time
from autoencoder import pretrain, build_network

logging.getLogger().setLevel(logging.DEBUG)

batch_s = 8

layer_sizes = [7500, 2500, 1000, 250]
pretrain_types=['relu','relu','relu','sigmoid']
train_types=['relu','relu','relu','softmax']
print(layer_sizes)
print(pretrain_types)
print(train_types)

pretrain_data = mx.io.ImageRecordIter(
  path_imgrec="data.rec",
  data_shape=(3, 64, 64),
  batch_size=batch_s)

x_train=[]
for batch in pretrain_data:
	for item in batch.data[0]:
		img=item.asnumpy()/255
		x_train.append(img)
X_train=mx.nd.array(x_train)

params = pretrain(layer_sizes, pretrain_types, X_train, batch_size=batch_s)
layers = build_network(layer_sizes, train_types)

train_data = mx.io.ImageRecordIter(
  path_imgrec="data.rec",
  data_shape=(3, 64, 64),
  batch_size=batch_s)

test_data = mx.io.ImageRecordIter(
  path_imgrec="test_data.rec",
  data_shape=(3, 64, 64),
  batch_size=batch_s)

model = mx.mod.Module(symbol=layers, context=mx.gpu())
t = time.clock()
model.fit(train_data,
          eval_data=test_data,
          arg_params=params,
          optimizer='sgd',
          optimizer_params={'learning_rate': 0.01},
          eval_metric='acc',
          batch_end_callback=mx.callback.Speedometer(batch_s, 200),
          num_epoch=1)
print("Clock time difference: %f" % (time.clock() - t))
acc = mx.metric.Accuracy()
model.score(test_data, acc)
print(acc)