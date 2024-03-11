from torch import nn


class BaseModels(object):
    def __init__(self):
        self.__init__()

    @staticmethod
    def basic_model(
            input_shape,
            act='elu',
            l2_reg=1e-3,
            filter_sz=5,
            num_filters=24,
            num_dense_neurons=512,
    ):
        """
            Returns a 2-tuple of (input, embedding_layer) that can later be used
            to create a model that builds on top of the embedding_layer.
            """
        inp = nn.Input(input_shape)

        x = nn.Conv2D(num_filters, (filter_sz, filter_sz),
                   padding='same', kernel_regularizer=nn.l2(l2_reg),
                   activation=act)(inp)
        x = nn.MaxPooling2D(2, 2)(x)

        x = nn.Conv2D(2 * num_filters, (filter_sz, filter_sz),
                   padding='same', kernel_regularizer=nn.l2(l2_reg),
                   activation=act)(x)
        x = nn.MaxPooling2D(2, 2)(x)

        x = nn.Conv2D(4 * num_filters, (filter_sz, filter_sz),
                   padding='same', kernel_regularizer=nn.l2(l2_reg),
                   activation=act)(x)
        x = nn.MaxPooling2D(2, 2)(x)

        x = nn.Dropout(.5)(x)
        x = nn.Flatten()(x)

        x = nn.Dense(num_dense_neurons, kernel_regularizer=nn.l2(l2_reg), activation=act)(x)
        x = nn.Dropout(.5)(x)
        x = nn.Dense(num_dense_neurons // 2, kernel_regularizer=nn.l2(l2_reg), activation=act)(x)
        emb = nn.Dropout(.5)(x)

        return inp, emb
