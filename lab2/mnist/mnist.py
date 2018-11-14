import mxnet as mx
import logging
import time

logging.getLogger().setLevel(logging.DEBUG)
mnist = mx.test_utils.get_mnist()
batch_size = 100
train_iter = mx.io.NDArrayIter(mnist['train_data'], mnist['train_label'], batch_size, shuffle=True)
val_iter = mx.io.NDArrayIter(mnist['test_data'], mnist['test_label'], batch_size)
data = mx.sym.var('data')
data = mx.sym.flatten(data=data)
fc = mx.sym.FullyConnected(data=data, num_hidden=10)
logreg = mx.sym.SoftmaxOutput(data=fc, name='softmax')
logreg_model = mx.mod.Module(symbol=logreg, context=mx.gpu())
logreg_model.fit(train_iter,
                 eval_data=val_iter,
                 optimizer='sgd',
                 optimizer_params={'learning_rate': 0.1},
                 eval_metric='acc',
                 batch_end_callback=mx.callback.Speedometer(batch_size, 100),
                 num_epoch=10)
test_iter = mx.io.NDArrayIter(mnist['test_data'], mnist['test_label'], batch_size)
acc = mx.metric.Accuracy()
logreg_model.score(test_iter, acc)
print(acc)
