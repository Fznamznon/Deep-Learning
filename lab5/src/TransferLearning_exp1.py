import mxnet as mx
import logging

logging.getLogger().setLevel(logging.DEBUG)

mx.test_utils.download('http://data.mxnet.io/mxnet/models/imagenet/resnext/50-layers/resnext-50-symbol.json')
mx.test_utils.download('http://data.mxnet.io/mxnet/models/imagenet/resnext/50-layers/resnext-50-0000.params')
print("MODEL LOADED")

batch_s = 8
train_data = mx.io.ImageRecordIter(
    path_imgrec="data.rec",
    data_shape=(3, 128, 128),
    batch_size=batch_s)
test_data = mx.io.ImageRecordIter(
    path_imgrec="test_data.rec",
    data_shape=(3, 128, 128),
    batch_size=batch_s)

print("DATA LOADED")
sym, arg_params, aux_params = mx.model.load_checkpoint('resnext-50', 0)
print("PARAMS LOADED")
model = mx.mod.Module(symbol=sym, context=mx.gpu())
print("MODEL CREATED")
model.fit(train_data,
          test_data,
          num_epoch=5,
          allow_missing=True,
          batch_end_callback=mx.callback.Speedometer(batch_s, 500),
          kvstore='device',
          optimizer='sgd',
          optimizer_params={'learning_rate': 0.01},
          initializer=mx.init.Xavier(rnd_type='gaussian', factor_type="in", magnitude=2),
          eval_metric='acc')
metric = mx.metric.Accuracy()
acc = model.score(test_data, metric)
print(acc)
