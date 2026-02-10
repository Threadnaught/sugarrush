from typing import *
import json

# A SUPER simple library for doing hyperparameter searches

class model_config:
	def __init__(self, epoch_count:int):
		self.epoch_count = epoch_count
		pass

class step_result:
	def __init__(self, type:str, epoch:int, step:int | None):
		self.type = type
		self.epoch = epoch
		self.step = step
		self.config: model_config | None = None
	
	def __str__(self):
		step_str = ''
		if self.step != None:
			step_str = ' step {}'.format(self.step)
		return '{} epoch {}/{}{} '.format(self.type, self.epoch, self.config.epoch_count, step_str)

def register_train_step_result(result:step_result, config: model_config, result_list:dict[str,list[step_result]], log_interval:dict[str, int]):
	# Add result to relevant list
	result.config = config
	if not result.type in result_list:
		result_list[result.type] = []
	result_list[result.type].append(result)
	
	# Log if if there is no registered step for this result, we don't have a log interval for this type, or the log interval lines up
	if result.step is None or \
		not result.type in log_interval or \
		result.step % log_interval[result.type] == 0:
		print(result)

# TODO: type gore
def run_training(
	train_individual_config:Callable[[model_config, Callable[[step_result], None]], None],
	configs: list[model_config],
	log_interval: dict[str, int]
) -> list[dict[str,list[step_result]]]:
	results: list[dict[str,list[step_result]]] = []
	for config in configs:
		# Once again the map/object distinction in python bites me
		print('Running config ', json.dumps(config.__dict__))
		results.append({})
		train_individual_config(config, lambda result:register_train_step_result(result, config, results[-1], log_interval))
	
	return results