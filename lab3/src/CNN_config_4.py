import mxnet as mx
import logging
import time

logging.getLogger().setLevel(logging.DEBUG)
batch_s = 8
print("Mariya config")
train_data = mx.io.ImageRecordIter(
    path_imgrec="data.rec",
    data_shape=(3, 128, 128),
    batch_size=batch_s)
test_data = mx.io.ImageRecordIter(
    path_imgrec="test_data.rec",
    data_shape=(3, 128, 128),
    batch_size=batch_s)
print("data - ok")

#First 'layer', if it can be put that way.
data = mx.sym.var('data')
c_layer1 = mx.sym.Convolution(data=data, kernel=(3, 3), num_filter=500, stride=(2, 2))
c_layer1_act = mx.sym.Activation(data=c_layer1, act_type="relu")
c_layer1_pool = mx.sym.Pooling(data=c_layer1_act, pool_type="max", kernel=(2, 2), stride=(2, 2))

fc_input = mx.sym.flatten(data=c_layer1_pool)

fc_layer_1 = mx.sym.FullyConnected(data=fc_input, num_hidden=500)
fc_layer_1_act = mx.sym.Activation(data=fc_layer_1, act_type='relu')
fc_layer_2 = mx.sym.FullyConnected(data=fc_layer_1_act, num_hidden=250)
fc_layer_2_act = mx.sym.Activation(data=fc_layer_2, act_type='sigmoid')
out = mx.sym.FullyConnected(data=fc_layer_2_act, num_hidden=2)
softmax = mx.sym.SoftmaxOutput(data=out, name='softmax')

print("layers - ok")
fcnn_net = mx.mod.Module(symbol=softmax, context=mx.gpu())
start_time = time.clock()
fcnn_net.fit(train_data,
             eval_data=test_data,
             optimizer='sgd',
             optimizer_params={'learning_rate': 0.001},
             eval_metric='acc',
             batch_end_callback=mx.callback.Speedometer(batch_s, 100),
             num_epoch=10)
fit_time = time.clock() - start_time
print("Fit time: %f", fit_time)
acc = mx.metric.Accuracy()
fcnn_net.score(test_data, acc)
print(acc)
