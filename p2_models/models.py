import torch
import torch.nn as nn
import timm
import torchvision.transforms as tt


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

    def get_infer_transforms(self):
        return tt.Compose([
            tt.ToTensor(),
            tt.Resize((224, 224))
        ])


class CheXNet(HuggingFaceModel):
    def __init__(self, n_classes: int) -> None:
        """
        Initializes a CheXNet model.
        :param n_classes: number of output classes.
        """
        super(CheXNet, self).__init__()
        self._model = timm.create_model('densenet121', pretrained=True)
        self._model.classifier = nn.Sequential(nn.Linear(1024, n_classes), nn.Softmax(dim=1))

    def get_target_layers(self):
        return [self._model.features[-1]]

    def get_infer_transforms(self):
        return tt.Compose([
            tt.ToTensor(),
            tt.Resize((224, 224))
        ])


class ResNet50(HuggingFaceModel):
    def __init__(self, n_classes: int) -> None:
        """
        Initializes a ResNet50 model.
        :param n_classes: number of output classes.
        """
        super(ResNet50, self).__init__()
        self._model = timm.create_model('resnet50', pretrained=True)
        self._model.fc = nn.Sequential(nn.Linear(2048, n_classes), nn.Softmax(dim=1))

    def get_target_layers(self):
        return [self._model.layer4[-1]]
