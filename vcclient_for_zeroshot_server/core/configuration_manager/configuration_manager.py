import os

from vcclient_for_zeroshot_server.const import ConfigFile
from vcclient_for_zeroshot_server.core.data_types.data_types import ServerConfiguration


class ConfigurationManager:
    _instance = None

    @classmethod
    def get_instance(cls):
        if cls._instance is None:

            cls._instance = cls()
            return cls._instance

        return cls._instance

    def __init__(self):
        self.reload()

    def reload(self):
        if os.path.exists(ConfigFile):
            self.voice_changer_configuration = ServerConfiguration.model_validate_json(open(ConfigFile, encoding="utf-8").read())
        else:
            self.voice_changer_configuration = ServerConfiguration()
            self.save_server_configuration()

    def get_server_configuration(self) -> ServerConfiguration:
        return self.voice_changer_configuration

    def set_server_configuration(self, conf: ServerConfiguration):
        self.voice_changer_configuration = conf
        self.save_server_configuration()

    def save_server_configuration(self):
        open(ConfigFile, "w", encoding="utf-8").write(self.voice_changer_configuration.model_dump_json(indent=4))
