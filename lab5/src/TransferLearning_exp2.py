import mxnet as mx
import logging
from collections import namedtuple

Batch = namedtuple('Batch', ['data'])


def get_data(batch_size):
    train_data = mx.io.ImageRecordIter(
        path_imgrec="data.rec",
        data_shape=(3, 128, 128),
        batch_size=batch_size)
    test_data = mx.io.ImageRecordIter(
        path_imgrec="test_data.rec",
        data_shape=(3, 128, 128),
        batch_size=batch_size)
    return train_data, test_data


def get_iter_after_net(resnext, data, batch_size):
    X = []
    y = []

    for batch in data:
        resnext.forward(Batch(batch.data))
        features = resnext.get_outputs()[0].asnumpy()
        tmp = batch.label[0].asnumpy()
        i = 0
        for item in features:
            X.append(item)
            y.append(tmp[i])
            i += 1

    iter = mx.io.NDArrayIter(mx.nd.array(X),
                             mx.nd.array(y),
                             batch_size)
    return iter


def get_resnext():
    sym, arg_params, aux_params = mx.model.load_checkpoint('resnext-50', 0)
    layers = sym.get_internals()
    sym1 = layers['flatten0_output']
    resnext = mx.mod.Module(symbol=sym1, context=mx.gpu(), label_names=None)
    resnext.bind(for_training=False, data_shapes=[('data', (1, 3, 128, 128))])
    resnext.set_params(arg_params, aux_params)
    return resnext


def main():
    batch_size = 8
    resnext = get_resnext()
    print("RESNEXT CREATED")
    train_data, test_data = get_data(50)
    print("DATA LOADED")
    train_iter = get_iter_after_net(resnext, train_data, batch_size)
    test_iter = get_iter_after_net(resnext, test_data, batch_size)
    print("RESNET WORK - DONE")

    data = mx.sym.var('data')
    data = mx.sym.flatten(data=data)
    fc_layer1 = mx.sym.FullyConnected(data=data, num_hidden=500)
    fc_layer1_act = mx.sym.Activation(data=fc_layer1, act_type="tanh")
    fc_layer2 = mx.sym.FullyConnected(data=fc_layer1_act, num_hidden=10)
    fc_out = mx.sym.SoftmaxOutput(data=fc_layer2, name='softmax')

    model = mx.mod.Module(symbol=fc_out, context=mx.gpu())

    logging.getLogger().setLevel(logging.DEBUG)

    model.fit(train_iter,
              test_iter,
              num_epoch=5,
              allow_missing=True,
              batch_end_callback=mx.callback.Speedometer(batch_size, 500),
              kvstore='device',
              optimizer='sgd',
              optimizer_params={'learning_rate': 0.01},
              initializer=mx.init.Xavier(rnd_type='gaussian', factor_type="in", magnitude=2),
              eval_metric='acc')

    metric = mx.metric.Accuracy()
    acc = model.score(test_iter, metric)
    print(acc)


if __name__ == "__main__":
    main()
