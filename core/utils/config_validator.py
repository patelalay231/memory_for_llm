from typing import Dict, Any, Tuple
from pydantic import BaseModel
from logger import Logger


class ConfigValidator:
    """Utility class for validating and extracting provider configurations."""
    
    @staticmethod
    def extract_provider_config(
        config: Any,
        config_name: str,
        supported_providers: list[str]
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Extract provider name and config from a nested config structure.
        
        Args:
            config: Configuration object (BaseModel or dict) with provider as key
            config_name: Name of the config being validated (for error messages)
            supported_providers: List of supported provider keys
            
        Returns:
            Tuple of (provider_name, provider_config_dict)
            
        Raises:
            ValueError: If config structure is invalid or provider is not supported
        """
        # Convert to dict if it's a Pydantic model
        if isinstance(config, BaseModel):
            config_dict = config.model_dump(exclude_none=True)
        elif isinstance(config, dict):
            config_dict = config.copy()
        else:
            config_dict = dict(config)
        
        # Find which provider is configured
        provider_name = None
        provider_config = None
        
        for provider_key in supported_providers:
            if provider_key in config_dict:
                if provider_name is not None:
                    # Multiple providers found
                    Logger.debug(
                        f"{config_name} config has multiple provider keys: {provider_name} and {provider_key}",
                        "[ConfigValidator]"
                    )
                    raise ValueError(
                        f"{config_name} config must have exactly one provider key from {supported_providers}"
                    )
                provider_name = provider_key
                provider_config = config_dict[provider_key]
        
        # Validate that exactly one provider was found
        if provider_name is None:
            Logger.debug(
                f"{config_name} config missing provider key. Expected one of: {supported_providers}",
                "[ConfigValidator]"
            )
            raise ValueError(
                f"{config_name} config must have one of these keys: {supported_providers}"
            )
        
        Logger.debug(
            f"Extracted {config_name} provider: {provider_name}",
            "[ConfigValidator]"
        )
        
        return provider_name, provider_config
    
    @staticmethod
    def validate_required_fields(
        config: Dict[str, Any],
        required_fields: list[str],
        config_name: str
    ):
        """
        Validate that all required fields are present in config.
        
        Args:
            config: Configuration dictionary
            required_fields: List of required field names
            config_name: Name of the config being validated (for error messages)
            
        Raises:
            ValueError: If any required field is missing
        """
        missing_fields = [field for field in required_fields if field not in config]
        
        if missing_fields:
            Logger.debug(
                f"{config_name} config missing required fields: {missing_fields}",
                "[ConfigValidator]"
            )
            raise ValueError(
                f"{config_name} config missing required fields: {missing_fields}"
            )
        
        Logger.debug(
            f"{config_name} config has all required fields",
            "[ConfigValidator]"
        )
