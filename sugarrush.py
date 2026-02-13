from typing import *
import numpy as np
import torch # todo should this be try import?
import json

# To those who would question the efficiency of storing each result as an object
# rather than pushing it directly into an ndarray, I would point out the separation
# of scales between what the GPU is doing (often 10s of GBs) and what we're doing here
# (each report must only be a few 100 bytes) if I ever run models which generate more
# logs, this may be an issue. Till then, KISS

model_config_t = dict[str, Any]
result_t = dict[str, Union[int, float, torch.Tensor, np.ndarray]] # todo report np arrays
result_narrowed_t = dict[str, np.ndarray]
report_result_t = Callable[[str, result_t], None]
train_individual_config_t = Callable[[model_config_t, report_result_t], Any]

# TODO: strings might be useful?
# TODO: global report ordering?
def report_result(
	result_type:str,
	result:result_t,
	config_results:dict[str, list[result_narrowed_t]],
	log_intervals:dict[str, int],
	config_i,
	configs_len
):
	# Append a new list if needed:
	if not result_type in config_results:
		config_results[result_type] = []
	
	# Convert everything to a nice easy numpy array:
	result_narrowed: result_narrowed_t = {}
	for key in result:
		value = result[key]
		if isinstance(value, int) or isinstance(value, float):
			result_narrowed[key] = np.array(value)
		elif isinstance(value, torch.Tensor):
			result_narrowed[key] = value.detach().cpu().numpy()
		elif isinstance(value, np.ndarray):
			result_narrowed[key] = value
		else:
			raise Exception('Result\'s key `{}` of value `{}` has unsupported type `{}`'.format(key, value, type(value)))
	config_results[result_type].append(result_narrowed)

	# Calculate index for reporting:
	report_i = len(config_results)
	if 'batch' in result_narrowed:
		report_i = int(result_narrowed['batch'])
	elif 'epoch' in result_narrowed:
		report_i = int(result_narrowed['epoch'])

	# Log if needed:
	if (not result_type in log_intervals) or (report_i % log_intervals[result_type] == 0):
		# print(result_type, result_i, result_narrowed)
		print('config {}/{} {}'.format(config_i+1, configs_len, result_type) + ', '.join(['{}:{}'.format(key, result_narrowed[key]) for key in result_narrowed]))

def run_training(
	train_individual_config:train_individual_config_t,
	configs:list[model_config_t],
	log_intervals: dict[str, int]
) -> Tuple[list[dict[str, list[result_narrowed_t]]], list[Any]]:
	config_results:list[dict[str, list[result_narrowed_t]]] = []
	config_returns:list[Any] = []

	for config_i in range(len(configs)):
		config = configs[config_i]
		print('using config ', json.dumps(config))
		config_results.append({})
		config_returns.append(train_individual_config(
			config,
			lambda report_type, result: report_result(report_type, result, config_results[-1], log_intervals,
			config_i,
			len(configs)
		)))

	return config_results, config_returns

def extract_column_single_config(results:dict[str, list[result_narrowed_t]], report_type:str, column_name:str) -> np.ndarray:
	return np.array([result[column_name] for result in results[report_type]])

def extract_column_all_configs(results:list[dict[str, list[result_narrowed_t]]], report_type:str, column_name:str) -> list[np.ndarray]:
	return [extract_column_single_config(result, report_type, column_name) for result in results]