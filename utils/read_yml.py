import yaml, re

class GenericLoader(yaml.SafeLoader):
    @classmethod
    def ignore_unknown(cls, loader, suffix, node):
        if isinstance(node, yaml.MappingNode):
            return loader.construct_mapping(node)
        elif isinstance(node, yaml.SequenceNode):
            return loader.construct_sequence(node)
        else:
            return loader.construct_scalar(node)

def create_loader():
    """Create a custom loader class."""
    class Loader(GenericLoader):
        pass

    # Regular expression to match any tag
    tag_pattern = re.compile(r'^tag:yaml.org,\d+:(.*)$')

    # Find all resolvers in SafeLoader
    for key, value in yaml.SafeLoader.yaml_implicit_resolvers.items():
        if isinstance(value, list):
            new_resolvers = []
            for item in value:
                tag, regexp = item
                # Check if the tag matches our pattern
                match = tag_pattern.match(tag)
                if match and match.group(1).startswith('python/'):
                    continue  # Skip this resolver
                new_resolvers.append(item)
            if new_resolvers:
                Loader.yaml_implicit_resolvers[key] = new_resolvers
            else:
                del Loader.yaml_implicit_resolvers[key]

    # Add a constructor for all tags that ignores the tag
    Loader.add_multi_constructor('', Loader.ignore_unknown)

    return Loader

def read_yaml(file_path):
    CustomLoader = create_loader()
    with open(file_path, 'r') as file:
        return yaml.load(file, Loader=CustomLoader)