import yaml


def load_config(config_path: str) -> dict:
    """
    Loads the configuration from a YAML file and returns it as a dictionary.
    """
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config
