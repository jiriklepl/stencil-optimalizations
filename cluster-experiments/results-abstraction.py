import os

EXPERIMENTS_SEPARATOR = 'next-experiment'
MEASUREMENTS_SEPARATOR = 'Time report:'

class Key:
    algorithm_name='algorithm_name'
    grid_dimensions='grid_dimensions'
    grid_dim_x='<special_case_grid_dim_x>'
    grid_dim_y='<special_case_grid_dim_y>'
    iterations='iterations'
    base_grid_encoding='base_grid_encoding'
    max_runtime_seconds='max_runtime_seconds'
    warmup_rounds='warmup_rounds'
    measurement_rounds='measurement_rounds'
    data_loader_name='data_loader_name'
    pattern_expression='pattern_expression'
    measure_speedup='measure_speedup'
    speedup_bench_algorithm_name='speedup_bench_algorithm_name'
    validate='validate'
    print_validation_diff='print_validation_diff'
    validation_algorithm_name='validation_algorithm_name'
    animate_output='animate_output'
    colorful='colorful'
    random_seed='random_seed'
    state_bits_count='state_bits_count'
    thread_block_size='thread_block_size'
    warp_dims_x='warp_dims_x'
    warp_dims_y='warp_dims_y'
    warp_tile_dims_x='warp_tile_dims_x'
    warp_tile_dims_y='warp_tile_dims_y'
    streaming_direction='streaming_direction'

class MeasurementKey:
    set_and_format_input_data='set_and_format_input_data'
    initialize_data_structures='initialize_data_structures'
    run='run'
    performed_iters='performed iters'
    runtime_per_iter='runtime per iter'
    finalize_data_structures='finalize_data_structures'

class Measurement:
    def __init__(self, content: str):
        self.content: str = content.strip()

    def get_value(self, key: str) -> float:
        raw = self._load_raw(key)
        return raw
    
    def _load_raw(self, key: str):
        splitted_by_key = self.content.split(f'{key}:')

        if (len(splitted_by_key) < 2):
            return None

        if (key == MeasurementKey.performed_iters):
            return splitted_by_key[1].split('\n')[0].strip()
        else:
            return splitted_by_key[1].split('ms')[0].strip()


class Experiment:
    def __init__(self, content: str):
        self.content: str = content.strip()

    def get_param(self, key: str):
        if key in [Key.grid_dim_x, Key.grid_dim_y]:
            raw = self._load_raw(Key.grid_dimensions)
            return self._parse_dim(key, raw)
        
        raw = self._load_raw(key)
        
        try:
            return int(raw)
        except:
            return raw
        
    def _parse_dim(self, dim_idx: str, raw_dims: str):
        both_dims = raw_dims.split('x')

        if dim_idx == Key.grid_dim_x:
            return int(both_dims[0])
        elif dim_idx == Key.grid_dim_y:
            return int(both_dims[1])
        
    def _load_raw(self, key: str):
        return self.content.split(f'{key}:')[1].split('\n')[0].strip()
    
    def get_measurements(self) -> list[Measurement]:
        measurements = []
        for measurement_content in self.content.split(MEASUREMENTS_SEPARATOR)[1:]:
            measurements.append(Measurement(measurement_content))
        return measurements
        
class Results:
    def __init__(self, results_content: str):
        self.experiments: list[Experiment] = []

        for exp_content in results_content.split(EXPERIMENTS_SEPARATOR)[1:]:
            self.experiments.append(Experiment(exp_content))

    @staticmethod
    def from_file(file_name) -> 'Results':
        file_contents = None
        with open(file_name, 'r') as f:
            file_contents = f.read()

        return Results(file_contents)

    @staticmethod
    def from_directory(dir_name) -> 'Results':
        results = None

        for file_name in os.listdir(dir_name):
            file_path = os.path.join(dir_name, file_name)
            print ('path: ', file_path)
            if os.path.isfile(file_path):
                f_results = Results.from_file(file_path)

                if results is None:
                    results = f_results
                else:
                    results.extend_with(f_results)

        return results

    def extend_with(self, results: 'Results'):
        self.experiments.extend(results.experiments)


x = Results.from_directory('./experiments-outputs')

# name = x.experiments[0].get_param(Key.algorithm_name)
# name = x.experiments[0].get_param(Key.iterations)
name = x.experiments[0].get_param(Key.grid_dim_x)
print(name)

measurement = x.experiments[0].get_measurements()[0]

print(measurement.get_value(MeasurementKey.performed_iters))
print(measurement.get_value(MeasurementKey.runtime_per_iter))