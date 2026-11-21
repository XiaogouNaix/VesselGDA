import torch
from batchgenerators.utilities.file_and_folder_operations import join
from batchgenerators.utilities.file_and_folder_operations import join
import json
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager
from nnunetv2.utilities.find_class_by_name import recursive_find_python_class
import pydoc
import warnings
from typing import Union
from dynamic_network_architectures.architectures.unet import ResidualEncoderUNet

def load_json(file: str):
    with open(file, 'r') as f:
        a = json.load(f)
    return a

def get_network_from_plans(arch_class_name, arch_kwargs, arch_kwargs_req_import, input_channels, output_channels, allow_init=True, deep_supervision: Union[bool, None]=None, network_class_type=ResidualEncoderUNet, additional_kwargs={}):
    network_class = arch_class_name
    architecture_kwargs = dict(**arch_kwargs)
    for ri in arch_kwargs_req_import:
        if architecture_kwargs[ri] is not None:
            architecture_kwargs[ri] = pydoc.locate(architecture_kwargs[ri])
    nw_class = pydoc.locate(network_class)
    if nw_class is None:
        warnings.warn(f'Network class {network_class} not found. Attempting to locate it within dynamic_network_architectures.architectures...')
        import dynamic_network_architectures
        nw_class = recursive_find_python_class(join(dynamic_network_architectures.__path__[0], 'architectures'), network_class.split('.')[-1], 'dynamic_network_architectures.architectures')
        if nw_class is not None:
            print(f'FOUND IT: {nw_class}')
        else:
            raise ImportError('Network class could not be found, please check/correct your plans file')
    if deep_supervision is not None and 'deep_supervision' not in arch_kwargs.keys():
        arch_kwargs['deep_supervision'] = deep_supervision
    architecture_kwargs.update(additional_kwargs)
    network = network_class_type(input_channels=input_channels, num_classes=output_channels, **architecture_kwargs)
    if hasattr(network, 'initialize') and allow_init:
        network.apply(network.initialize)
    return network

def make_unet(folder, checkpoint_path, network_class_type=ResidualEncoderUNet, additional_kwargs={}, in_channels=1, out_channels=1):
    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    configuration_name = checkpoint['init_args']['configuration']
    dataset_json = load_json(join(folder, 'dataset.json'))
    plans = load_json(join(folder, 'plans.json'))
    plans_manager = PlansManager(plans)
    configuration_manager = plans_manager.get_configuration(configuration_name)
    if 'track' in checkpoint_path:
        configuration_manager.network_arch_init_kwargs['norm_op_kwargs']['track_running_stats'] = True
    network = get_network_from_plans(configuration_manager.network_arch_class_name, configuration_manager.network_arch_init_kwargs, configuration_manager.network_arch_init_kwargs_req_import, in_channels, out_channels, plans_manager.get_label_manager(dataset_json).num_segmentation_heads, network_class_type=network_class_type, additional_kwargs=additional_kwargs)
    return network
