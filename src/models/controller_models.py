from torch import nn

import models.controller_model


class BaseModels(object):
    def __init__(self):
        self.__init__()

    @staticmethod
    def basic_model(
            act=nn.ELU,
            filter_sz=5,
            num_filters=24,
            num_dense_neurons=512,
            num_classes=5,
    ):
        """
            Returns a 2-tuple of (input, embedding_layer) that can later be used
            to create a model that builds on top of the embedding_layer.
            """

        return models.controller_model.ControllerModel(
            in_channels=1,
            num_filters=num_filters,
            num_dense_neurons=num_dense_neurons,
            filter_sz=filter_sz,
            act=act,
            num_classes=num_classes,
        )

