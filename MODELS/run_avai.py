import yaml
import sys

class SafeLoaderWithCustom(yaml.SafeLoader):
    pass

# Xử lý torch.device
def construct_torch_device(loader, node):
    return str(loader.construct_sequence(node)[0])

# Xử lý tuple
def construct_python_tuple(loader, node):
    return tuple(loader.construct_sequence(node))

# Đăng ký constructor
SafeLoaderWithCustom.add_constructor(
    'tag:yaml.org,2002:python/object/apply:torch.device',
    construct_torch_device
)
SafeLoaderWithCustom.add_constructor(
    'tag:yaml.org,2002:python/tuple',
    construct_python_tuple
)

yaml_path = sys.argv[1] if len(sys.argv) > 1 else "config.yaml"

with open(yaml_path, 'r') as f:
    config = yaml.load(f, Loader=SafeLoaderWithCustom)

args = config.get('args', config)

def normalize_value(val):
    if isinstance(val, bool):
        return "true" if val else "false"
    elif isinstance(val, (list, tuple)):
        return '"' + ",".join(map(str, val)) + '"'
    elif val is None:
        return ""
    else:
        return str(val)

for key, value in args.items():
    key_upper = key.upper()
    value_str = normalize_value(value)
    print(f'{key_upper}="{value_str}"')
