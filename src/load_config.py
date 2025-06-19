
import yaml

class ConfigSettings:
    def __init__(self, yaml_file_name):
        self.yaml_file_name = yaml_file_name
        self.config_file_path = 'configs/' + yaml_file_name
        
        with open(self.config_file_path) as f:
            config = yaml.safe_load(f)

        self.run_id = config.get('run_id', None)
        self.generation_error_types: list[str] = config.get('generation_error_types', [])
        self.get_load_error_flag: bool = config.get('get_load_error_flag', True)
        self.error_types: list[str] = config.get('generation_error_types', [])
        if self.get_load_error_flag:
            self.error_types = self.error_types + ['load']

        self.zones_error_types: dict[str, list[str]] = config.get('zones_error_types', None)
        self.years: list[int] = config.get('years', None)

        self.forecasting_model: str = config.get('forecasting_model', 'LSTM')
        self.fitting_function: str = config.get('fitting_function', 'asymptotic')
        self.train_test_split: float = config.get('train_test_split', None)
        self.n_lags: int = config.get('n_lags', None)
        self.max_lead_time: int = config.get('max_lead_time', None)

    def __repr__(self):
        # print all attributes in a readable format
        return f"ConfigSettings({self.yaml_file_name}): " + "\n".join(f"{k}={v}" for k, v in self.__dict__.items() if k != 'config_file_path')
