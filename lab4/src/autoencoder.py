import mxnet as mx
import numpy as np
import time

def first_metric(label, pred):
    label = label.reshape(-1, 64*64*3)
    return np.mean((label - pred) ** 2)

eval_metric = mx.metric.create(first_metric)

def pretrain(layer_sizes, act_types, X_train, batch_size=1):
    params = {}
    y_train = X_train
    encoder_iter = mx.io.NDArrayIter(X_train,
                                     y_train,
                                     batch_size,
                                     shuffle=True) 
    data = mx.sym.var('data')
    data = mx.sym.flatten(data=data)
        
    for i, (act_type, num_hidden) in enumerate(zip(act_types, layer_sizes)):
        encoder = mx.symbol.FullyConnected(data=data, num_hidden=num_hidden, name='layer_'+str(i))
        encoder_act = mx.sym.Activation(data=encoder, act_type=act_type, name='layer_'+str(i)+'_activation')

        num_hidden = 64*64*3 if i == 0 else layer_sizes[i-1]
        decoder = mx.symbol.FullyConnected(data=encoder_act, num_hidden=num_hidden, name='decoder')
        decoder_act = mx.sym.LinearRegressionOutput(data=decoder, name='softmax')
        
        autoencoder = mx.mod.Module(symbol=decoder_act, context=mx.gpu())

        autoencoder.fit(encoder_iter,
                        optimizer='sgd',
                        optimizer_params={'learning_rate': 0.01},
                        eval_metric= (eval_metric if i==0 else 'mse'),
                        batch_end_callback=mx.callback.Speedometer(batch_size, 200),
                        num_epoch=1)
                        
        params.update(autoencoder.get_params()[0].items())
        output = autoencoder.symbol.get_internals()['layer_'+str(i)+'_activation_output']
        params.update({'data': X_train}.items())

        
        X_train = output.eval(ctx=mx.cpu(), **params)[0]
        y_train = X_train
        encoder_iter = mx.io.NDArrayIter(X_train,
                                         y_train,
                                         batch_size,
                                         shuffle=True)
    del(params['data'])
    return params


def build_network(layer_sizes, act_types):
    data = mx.sym.var('data')
    data = mx.sym.flatten(data=data)

    layer_activation = data
    for i, (act_type, num_hidden) in enumerate(zip(act_types, layer_sizes)):
        layer = mx.symbol.FullyConnected(layer_activation, num_hidden=num_hidden, name='layer_'+str(i))
        
        if act_type == 'softmax':
            layer_activation = mx.sym.SoftmaxOutput(data=layer, name='softmax')
        else:
            layer_activation = mx.sym.Activation(data=layer, act_type=act_type, name='layer_'+str(i)+'_act')
    return layer_activation