import torch
import torchvision.models as models

def load_pretrained_model(model_name,num_classes):
    if model_name == 'vgg16':
        model = models.vgg16(weights='DEFAULT')#pretrained=True)
        model.classifier[6] = torch.nn.Linear(in_features=4096, out_features=num_classes)
    elif model_name == 'vgg19':
        model = models.vgg19(weights='DEFAULT')#pretrained=True)
        model.classifier[6] = torch.nn.Linear(in_features=4096, out_features=num_classes)
    elif model_name == 'mobilenet':
        model = models.mobilenet_v2(weights='DEFAULT')#pretrained=True)
        model.classifier[1] = torch.nn.Linear(in_features=1280, out_features=num_classes)
    else:
        raise ValueError("Model not supported.")
    return model



