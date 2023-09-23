from dataclasses import dataclass
import secrets


@dataclass
class ConfigParams:
    preview_window_size: int = 50
    lose_function: str = 'mean_squared_error'
    optimizator: str = 'adam'
    neurons_qty_on_temporal_layer: int = 100
    neurons_qty_on_dense_layer: int = 80
    epochs: int = 100
    lote: int = 100
    show_results: bool = False
    test_id: str = secrets.token_hex(6)

@dataclass
class Results:
    graph: str = None 
    mean_squared_error_train: int = None
    mean_squared_error_test: int = None
    mean_absolute_error_train: int = None
    mean_absolute_error_test: int = None
    r2_train: int = None
    r2_test: int = None