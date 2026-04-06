from collections.abc import Callable, Iterable, Mapping
from typing import *
from typing import Any
import numpy as np
import torch
import torch.multiprocessing as mp
import json
import datetime

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
		print('config {}/{} {} '.format(config_i+1, configs_len, result_type) + ', '.join(['{}:{}'.format(key, result_narrowed[key]) for key in result_narrowed]))

# TODO: typing
def run_single_config(args):
	start_time = datetime.datetime.now()
	config_i, configs, train_individual_config, log_intervals = args
	config = configs[config_i]
	print('using config ', json.dumps(config))
	current_config_results = {}
	current_config_return = train_individual_config(
		config,
		lambda report_type, result: report_result(report_type, result, current_config_results, log_intervals,
		config_i,
		len(configs)
	))

	duration_s = (datetime.datetime.now() - start_time).total_seconds()
	report_result('timing', {'duration_s':duration_s}, current_config_results, log_intervals, config_i, len(configs))

	return current_config_results, current_config_return

def run_training(
	train_individual_config:train_individual_config_t,
	configs:list[model_config_t],
	log_intervals: dict[str, int],
	num_workers: int = 1
) -> Tuple[list[dict[str, list[result_narrowed_t]]], list[Any]]:
	config_results:list[dict[str, list[result_narrowed_t]]] = []
	config_returns:list[Any] = []
	
	if num_workers > 1:
		# Note: this approach is somewhat limited - python intentionally blocks subprocesses from launching
		# subsubprocesses because the python devs don't trust you with that power, despite it already being
		# possible to do plenty of other ill-advised things with their programming language.

		# There are workarounds but they're all pretty ugly. Version two of sr should probably just invoke
		# instances of the model using subprocess or something.

		# **The practical outcome here is that any data loaders need num_workers=0 :-(**

		mp.set_start_method('spawn')
		pool = mp.Pool(num_workers)
		map_outputs = pool.map(run_single_config,
			zip(
				range(len(configs)),
				[configs for _ in range(len(configs))],
				[train_individual_config for _ in range(len(configs))],
				[log_intervals for _ in range(len(configs))],
			),
		)
		return zip(*map_outputs)
	else:
		for config_i in range(len(configs)):
			current_config_results, current_config_return = run_single_config((config_i, configs, train_individual_config, log_intervals))
			config_results.append(current_config_results)
			config_returns.append(current_config_return)

		return config_results, config_returns

def extract_column_single_config(results:dict[str, list[result_narrowed_t]], report_type:str, column_name:str) -> np.ndarray:
	return np.array([result[column_name] for result in results[report_type]])

def extract_column_all_configs(results:list[dict[str, list[result_narrowed_t]]], report_type:str, column_name:str) -> list[np.ndarray]:
	return [extract_column_single_config(result, report_type, column_name) for result in results]

class NumpyEncoder(json.JSONEncoder):
	def default(self, o: Any) -> Any:
		if isinstance(o, np.ndarray):
			return o.tolist()
		return super().default(o)
