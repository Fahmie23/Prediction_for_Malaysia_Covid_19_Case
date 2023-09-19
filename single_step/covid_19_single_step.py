#%%
# 1. Import packages and setup

import os
import datetime

import IPython
import IPython.display
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from tensorflow import keras as keras
from tensorflow.keras import callbacks
from tensorflow.keras.regularizers import l2
from tensorflow.keras.utils import plot_model

mpl.rcParams['figure.figsize'] = (8, 6)
mpl.rcParams['axes.grid'] = False

# %%
#2.  Load the data from csv files
file_path = "cases_malaysia_covid.csv"
df = pd.read_csv(file_path)
df.head()

# %%
# 2.1 Selecting the columns that is needed for the model
selected_columns = ["date", "cases_new", "cases_import", "cases_recovered", "cases_active"]

df = df[selected_columns]

# %%
# 2.2 Data Inspection
# Checking wheter the dataset has columns that are needed for the model.
df.head()

# %%
# Checking the data types for each of the columns
df.info()

#%%
# From the checking that have been made, 'date' and 'cases_new' column need be change to respective data types which is datetime and integer respectively

# Convert 'date' column to datetime data type
date_time = pd.to_datetime(df.pop('date'), format='%d/%m/%Y')

#%%
# Replace '?' with NaN for missing data
df['cases_new'] = df['cases_new'].replace('?', np.nan)
# Replace ' ' (space) with NaN for missing data
df['cases_new'] = df['cases_new'].replace(' ', np.nan)
# Convert 'cases_new' from object to float datatype.
df['cases_new'] = df['cases_new'].astype(float) 

#%%
#Checking the dataset after the datatype conversion
df.info()

#%%
# Checking NaN (missing values) inside the dataset
missing_counts = df.isna().sum()
missing_counts

#%%
# Interpolate missing values
df.interpolate(inplace=True, method='polynomial', order=3)  # This replaces NaN values in-place

#%%
#Checking the dataset after substitute the NaN (missing values) with interpolated values
df.info()

#%%
# Plotting the graphs to inspect any trend
plot_cols = ['cases_new', 'cases_import', 'cases_recovered', 'cases_active']
plot_features = df[plot_cols]
plot_features.index = date_time
_ = plot_features.plot(subplots=True)

plot_features = df[plot_cols][:200]
plot_features.index = date_time[:200]
_ = plot_features.plot(subplots=True)

#%%
# Create subplots for each column
plt.figure(figsize=(12, 6))
plt.subplots_adjust(wspace=0.5)  # Adjust horizontal space between subplots

# Iterate through each column and create a box plot
for i, column in enumerate(df.columns):
    plt.subplot(1, len(df.columns), i + 1)
    sns.boxplot(x=df[column])
    plt.title(column)

#%%
# Inspect some basic statistics from the dataset
df.describe().transpose()

# %%
# 4. Data Cleaning

# %%
# 5. Feature Engineering

# %%
# 6. Split the data
#Note: We don't want to shuffle the data when splitting, to ensure the data is still in order based on the time steps
column_indices = {name: i for i, name in enumerate(df.columns)}

n = len(df)
train_df = df[0:int(n*0.7)]
val_df = df[int(n*0.7):int(n*0.9)]
test_df = df[int(n*0.9):]

num_features = df.shape[1]

# %%
# 7. Data Normalization
train_mean = train_df.mean()
train_std = train_df.std()

train_df = (train_df - train_mean) / train_std
val_df = (val_df - train_mean) / train_std
test_df = (test_df - train_mean) / train_std

# %%
#Inspect distribution of the features after normalization
df_std = (df - train_mean) / train_std
df_std = df_std.melt(var_name='Column', value_name='Normalized')
plt.figure(figsize=(12, 6))
ax = sns.violinplot(x='Column', y='Normalized', data=df_std)
_ = ax.set_xticklabels(df.keys(), rotation=90)

# %%
# 8. Data windowing

class WindowGenerator():
  def __init__(self, input_width, label_width, shift,
               train_df=train_df, val_df=val_df, test_df=test_df,
               label_columns=None):
    # Store the raw data.
    self.train_df = train_df
    self.val_df = val_df
    self.test_df = test_df

    # Work out the label column indices.
    self.label_columns = label_columns
    if label_columns is not None:
      self.label_columns_indices = {name: i for i, name in
                                    enumerate(label_columns)}
    self.column_indices = {name: i for i, name in
                           enumerate(train_df.columns)}

    # Work out the window parameters.
    self.input_width = input_width
    self.label_width = label_width
    self.shift = shift                    # shift = offset

    self.total_window_size = input_width + shift

    self.input_slice = slice(0, input_width)
    self.input_indices = np.arange(self.total_window_size)[self.input_slice]

    self.label_start = self.total_window_size - self.label_width
    self.labels_slice = slice(self.label_start, None)
    self.label_indices = np.arange(self.total_window_size)[self.labels_slice]

  def __repr__(self):
    return '\n'.join([
        f'Total window size: {self.total_window_size}',
        f'Input indices: {self.input_indices}',
        f'Label indices: {self.label_indices}',
        f'Label column name(s): {self.label_columns}'])
  
  # 8.1 Split window
  def split_window(self, features):
    inputs = features[:, self.input_slice, :]
    labels = features[:, self.labels_slice, :]
    if self.label_columns is not None:
      labels = tf.stack(
        [labels[:, :, self.column_indices[name]] for name in self.label_columns],
        axis=-1)

    # Slicing doesn't preserve static shape information, so set the shapes
    # manually. This way the `tf.data.Datasets` are easier to inspect.
    inputs.set_shape([None, self.input_width, None])
    labels.set_shape([None, self.label_width, None])

    return inputs, labels
  
  # 8.2 Data Visualization
  def plot(self, model=None, plot_col='cases_new', max_subplots=3):
    inputs, labels = self.example
    plt.figure(figsize=(12, 8))
    plot_col_index = self.column_indices[plot_col]
    max_n = min(max_subplots, len(inputs))
    for n in range(max_n):
      plt.subplot(max_n, 1, n+1)
      plt.ylabel(f'{plot_col} [normed]')
      plt.plot(self.input_indices, inputs[n, :, plot_col_index],
              label='Inputs', marker='.', zorder=-10)

      if self.label_columns:
        label_col_index = self.label_columns_indices.get(plot_col, None)
      else:
        label_col_index = plot_col_index

      if label_col_index is None:
        continue

      plt.scatter(self.label_indices, labels[n, :, label_col_index],
                  edgecolors='k', label='Labels', c='#2ca02c', s=64)
      if model is not None:
        predictions = model(inputs)
        plt.scatter(self.label_indices, predictions[n, :, label_col_index],
                    marker='X', edgecolors='k', label='Predictions',
                    c='#ff7f0e', s=64)

      if n == 0:
        plt.legend()

    plt.xlabel('Time [h]')

  # Create tf.data.Datasets
  # make modifications here if you want to change 
  def make_dataset(self, data):
    data = np.array(data, dtype=np.float32)
    ds = tf.keras.utils.timeseries_dataset_from_array(
        data=data,
        targets=None,
        sequence_length=self.total_window_size,
        sequence_stride=1,
        shuffle=True,
        batch_size=32,)                           # important

    ds = ds.map(self.split_window)

    return ds

  # Adding property
  @property
  def train(self):
    return self.make_dataset(self.train_df)

  @property
  def val(self):
    return self.make_dataset(self.val_df)

  @property
  def test(self):
    return self.make_dataset(self.test_df)

  @property
  def example(self):
    """Get and cache an example batch of `inputs, labels` for plotting."""
    result = getattr(self, '_example', None)
    if result is None:
      # No example batch was found, so get one from the `.train` dataset
      result = next(iter(self.train))
      # And cache it for next time
      self._example = result
    return result

# %%
w1 = WindowGenerator(input_width=30, label_width=1, shift=30,
                     label_columns=['cases_new'])
w1
# %%
w2 = WindowGenerator(input_width=30, label_width=30, shift=30,
                     label_columns=['cases_new'])
w2


# %%
# trying out the split window function

# Stack three slices, the length of the total window.
example_window = tf.stack([np.array(train_df[:w2.total_window_size]),
                           np.array(train_df[100:100+w2.total_window_size]),
                           np.array(train_df[200:200+w2.total_window_size])])

example_inputs, example_labels = w2.split_window(example_window)
print('All shapes are: (batch, time, features)')
print(f'Window shape: {example_window.shape}')
print(f'Inputs shape: {example_inputs.shape}')
print(f'Labels shape: {example_labels.shape}')

# %%
# Testing the 8.2 function
# w2.example = example_inputs, example_labels
w2.plot()

# %%
# Each element is an (inputs, label) pair.
w2.train.element_spec

# %%
# 9. Model development
# Create the data widnow
wide_window = WindowGenerator(
    input_width=30, label_width=30, shift=1,
    label_columns=['cases_new'])

print(wide_window)
wide_window.plot()

# %%
# Single-step LSTM model
lstm_model = keras.Sequential()
lstm_model.add(keras.layers.LSTM(64, return_sequences=True, kernel_regularizer=l2(0.01)))
lstm_model.add(keras.layers.Dropout(0.2))
lstm_model.add(keras.layers.LSTM(32, return_sequences=True, kernel_regularizer=l2(0.01)))
lstm_model.add(keras.layers.Dropout(0.2))
lstm_model.add(keras.layers.Dense(1, kernel_regularizer=l2(0.01)))


MAX_EPOCHS = 50

def compile_and_fit(model, window, epochs=MAX_EPOCHS, patience=2):
  early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                    patience=patience,
                                                    mode='min')

  model.compile(loss=tf.keras.losses.MeanSquaredError(),
                optimizer=tf.keras.optimizers.Adam(),
                metrics=[tf.keras.metrics.MeanAbsoluteError(), tf.keras.metrics.MeanAbsolutePercentageError()])

  # Create TensorBoard callback
  base_log_path = r"tensorboard_logs\covid_single_step_model"
  log_path = os.path.join(base_log_path, datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
  tb = callbacks.TensorBoard(log_path)

  history = model.fit(window.train, epochs=epochs,
                      validation_data=window.val,
                      callbacks=[early_stopping,tb])
  return history

history = compile_and_fit(lstm_model, wide_window, patience = 3)

# %%
wide_window.plot(lstm_model)

#%%
mae = history.history['mean_absolute_error']  # Extract MAE values
mape = history.history['mean_absolute_percentage_error']  # Extract MAPE values
# Get the MAE and MAPE values for the last epoch (or any specific epoch you're interested in)
final_mae = mae[-1]  # MAE at the last epoch
final_mape = mape[-1]  # MAPE at the last epoch

print("Final MAE:", final_mae)
print("Final MAPE:", final_mape)

#%%
plot_model(lstm_model,show_shapes=True, show_layer_names=True)
# %%
# Multi-output, single step model
wide_window = WindowGenerator(
    input_width=30, label_width=30, shift=1)

for example_inputs, example_labels in wide_window.train.take(1):
  print(f'Inputs shape (batch, time, features): {example_inputs.shape}')
  print(f'Labels shape (batch, time, features): {example_labels.shape}')

# %%
lstm_model_2 = keras.Sequential()
lstm_model_2.add(keras.layers.LSTM(32, return_sequences=True, kernel_regularizer=l2(0.01)))
lstm_model_2.add(keras.layers.Dense(example_labels.shape[-1], kernel_regularizer=l2(0.01)))

history = compile_and_fit(lstm_model_2, wide_window, patience = 3)
# %%
wide_window.plot(lstm_model_2, plot_col= 'cases_new')

#%%
mae = history.history['mean_absolute_error']  # Extract MAE values
mape = history.history['mean_absolute_percentage_error']  # Extract MAPE values
# Get the MAE and MAPE values for the last epoch (or any specific epoch you're interested in)
final_mae = mae[-1]  # MAE at the last epoch
final_mape = mape[-1]  # MAPE at the last epoch

print("Final MAE:", final_mae)
print("Final MAPE:", final_mape)
# %%
plot_model(lstm_model_2,show_shapes=True, show_layer_names=True)
# %%
