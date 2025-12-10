"""Package initialization for config module"""
from config.settings import settings, Settings
from config.prompts import prompts, PromptTemplates

__all__ = ["settings", "Settings", "prompts", "PromptTemplates"]
