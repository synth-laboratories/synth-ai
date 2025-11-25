import platform
from dataclasses import dataclass
from pathlib import Path

from synth_ai.core.json import create_and_write_json, load_json_to_dict


@dataclass
class ClaudeConfig:
    config_path: Path
    server_name: str = "synth-ai"
    cmd: str = "python"
    args: tuple[str, ...] = ("-m", "synth_ai.mcp")
    
    @staticmethod
    def get_default_config_path() -> Path:
        system = platform.system()
        if system == "Darwin":
            return Path.home() / "Library/Application Support/Claude/claude_desktop_config.json"
        return Path.home() / ".config" / "Claude" / "claude_desktop_config.json"
    
    def update_mcp_config(self) -> None:
        print("Adding Synth AI to your Claude MCP config")
        config = load_json_to_dict(self.config_path)
        if not isinstance(config, dict):
            config = {}
        servers = config.setdefault("mcpServers", {})
        if not isinstance(servers, dict):
            servers = {}
            config["mcpServers"] = servers
        servers[self.server_name] = {
            "command": self.cmd,
            "args": list(self.args)
        }
        create_and_write_json(self.config_path, config)
        print("Adding Synth AI to your Claude MCP config at", self.config_path)
