"""
Configuration loader with variable substitution
"""
import os
import yaml
from typing import Dict, Any


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load YAML config and substitute variables
    
    Args:
        config_path: Path to YAML config file
        
    Returns:
        Dictionary with config values
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Substitute ${variable} patterns
    def substitute_vars(obj, config_dict):
        if isinstance(obj, dict):
            return {k: substitute_vars(v, config_dict) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [substitute_vars(item, config_dict) for item in obj]
        elif isinstance(obj, str):
            # Replace ${var} with config_dict['var']
            if '${' in obj:
                import re
                pattern = r'\$\{(\w+)\}'
                def replacer(match):
                    key = match.group(1)
                    return str(config_dict.get(key, match.group(0)))
                obj = re.sub(pattern, replacer, obj)
            return obj
        else:
            return obj
    
    # First pass: substitute variables
    config = substitute_vars(config, config)
    
    # Convert nested dict keys to uppercase for compatibility
    def to_upper_keys(d):
        if isinstance(d, dict):
            return {k.upper(): to_upper_keys(v) if isinstance(v, dict) else v for k, v in d.items()}
        return d
    
    # Keep eval config separate
    eval_config = config.pop('eval', {})
    config = to_upper_keys(config)
    
    # Add eval config back
    config['EVAL'] = eval_config
    
    return config

