import logging
from fllib.models.LeNet5 import LeNet5
from fllib.models.MobileNet import MobileNetV2, MobileNetV3_large, MobileNetV3_small
from fllib.models.vgg9 import VGG9
from fllib.models.Resnet import ResNet18, ResNet34, ResNet50


logger = logging.getLogger(__name__)
support_models = ['LeNet5', 'vgg9', 'resnet18', 'resnet34', 'resnet50', 'mobilenetv2', 'mobilenetv3_large', 'mobilenetv3_small']

# base_package = 'fllib'
# def load_model(model_name: str, channels=None):
#     dir_path = path.dirname(path.realpath(__file__))
#     model_file = path.join(dir_path, "{}.py".format(model_name))   
#     if not path.exists(model_file):
#         logger.error("Please specify a valid model.")
#     model_path = "{}.models.{}".format(base_package, model_name)
#     model_lib = importlib.import_module(model_path)
#     model = getattr(model_lib, "Model")
   
#     return model(channels=channels)


def load_model(model_name, num_class=10, channels=1):
    if model_name in support_models:
        if model_name == 'LeNet5':
            model = LeNet5()
        elif model_name == 'vgg9':
            model = VGG9(num_classes=num_class, channels=channels)
        
        elif model_name == 'resnet18':
            model = ResNet18(num_classes=num_class, channels=channels)
        elif model_name == 'resnet34':
            model = ResNet34(num_classes=num_class, channels=channels)
        elif model_name == 'resnet50':
            model = ResNet50(num_classes=num_class, channels=channels)
        
        elif model_name == 'mobilenetv2':
            model = MobileNetV2(num_classes=num_class, channels=channels)
        elif model_name == 'mobilenetv3_large':
            model = MobileNetV3_large(num_classes=num_class, channels=channels)
        elif model_name == 'mobilenetv3_small':
            model = MobileNetV3_small(num_classes=num_class, channels=channels)


    else:
        raise ValueError(f'Model name is not correct, the options are listed as follows: {support_models}')
    
    return model
