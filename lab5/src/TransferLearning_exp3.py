import mxnet as mx
import logging


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


def get_netmodel():
    sym, arg_params, aux_params = mx.model.load_checkpoint('resnext-50', 0)
    layers = sym.get_internals()
    resnext_sym = layers['flatten0_output']
    fc = mx.symbol.FullyConnected(data=resnext_sym, num_hidden=10, name='fc1')
    fc_out = mx.symbol.SoftmaxOutput(data=fc, name='softmax')
    model = mx.mod.Module(symbol=fc_out, context=mx.gpu())
    new_arg_params = dict({k: arg_params[k] for k in arg_params if 'fc1' not in k})
    return model, new_arg_params, aux_params


def main():
    batch_size = 8
    train_data, test_data = get_data(batch_size)

    model, arg_params, aux_params = get_netmodel()
    logging.getLogger().setLevel(logging.DEBUG)
    print("START")
    model.fit(train_data,
              test_data,
              num_epoch=2,
              arg_params=arg_params,
              aux_params=aux_params,
              allow_missing=True,
              batch_end_callback=mx.callback.Speedometer(batch_size, 500),
              kvstore='device',
              optimizer='sgd',
              optimizer_params={'learning_rate': 0.01},
              initializer=mx.init.Xavier(rnd_type='gaussian', factor_type="in", magnitude=2),
              eval_metric='acc')

    metric = mx.metric.Accuracy()
    acc = model.score(test_data, metric)
    print(acc)


if __name__ == "__main__":
    main()