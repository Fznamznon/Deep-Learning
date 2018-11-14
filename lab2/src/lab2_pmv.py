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
input = mx.sym.flatten(data=data)
first = mx.sym.FullyConnected(data=input, num_hidden=2000)
firstact = mx.sym.Activation(data=first, act_type='relu')
second = mx.sym.FullyConnected(data=firstact, num_hidden=1000)
secondact = mx.sym.Activation(data=second, act_type='relu')
third = mx.sym.FullyConnected(data=secondact, num_hidden=500)
thirdact = mx.sym.Activation(data=third, act_type='relu')
fourth = mx.sym.FullyConnected(data=thirdact, num_hidden=250)
fourthact = mx.sym.Activation(data=fourth, act_type='sigmoid')
fc = mx.sym.FullyConnected(data=fourthact, num_hidden=2)
softmax = mx.sym.SoftmaxOutput(data=fc, name='softmax')
#softmax = mx.sym.LogisticRegressionOutput(data=fc, name='softmax')

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
