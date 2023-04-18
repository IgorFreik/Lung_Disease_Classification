import torch
import torch.nn as nn
import timm


class BaseNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(0, 6)

    def forward(self, x):
        return self.fc(x)


class HuggingFaceModel(nn.Module):
    def __init__(self):
        super(HuggingFaceModel, self).__init__()

    # Defining the forward pass
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of a ChexNet object.
        """
        x = self._model(x)
        return x

    def freeze_all_except_last(self):
        """
        Freezes all model parameters except for the last Linear layer.
        """
        for param in self._model.parameters():
            param.requires_grad = False
        for param in self._model.classifier.parameters():
            param.requires_grad = True

    def unfreeze_all(self):
        """
        Unfreezes all model parameters.
        """
        for param in self._model.parameters():
            param.requires_grad = True

    def get_trainable_params(self, only_last: bool):
        """
        Returns a generator of the trainable parameters of the model.
        There are two scenarios: all parameters or only the parameters of the last Linear layer.
        """
        if only_last:
            return self._model.classifier[0].parameters()
        else:
            return self._model.parameters()

    def get_target_layers(self):
        raise NotImplemented


class CheXNet(HuggingFaceModel):
    def __init__(self, n_classes: int) -> None:
        """
        Initializes a CheXNet model.
        """
        super(CheXNet, self).__init__()
        self._model = timm.create_model('densenet121', pretrained=True)
        self._model.classifier = nn.Sequential(nn.Linear(1024, n_classes), nn.Softmax(dim=1))
