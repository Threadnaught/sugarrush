import sugarrush as sr
import random
import matplotlib.pyplot as plt
from typing import *

# Define your training function:
def train_individual_config(config, report) -> None:
    for epoch in range(10):
        for batch in range(100):
            report(
                'train', # what category of info this is
                {'batch':batch, 'epoch':epoch, 'loss':random.uniform(0,config['max'])} 
            )
        report('test', {'epoch':epoch, 'loss':random.uniform(0,config['max'])})

# Run the training
results, _ = sr.run_training(
    train_individual_config, # Your training function
    [{'max':2},{'max':5}], # Your configs to train
    {'train':10} # Your log interval (missing implies reporting logging every time)
)

# Convert the config reports into numpy arrays
zeroth_losses, first_losses = sr.extract_column_all_configs(results, 'train', 'loss')

# Plot them
plt.plot(first_losses)
plt.plot(zeroth_losses)

plt.show()