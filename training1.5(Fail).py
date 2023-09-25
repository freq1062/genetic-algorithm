train_notes = np.stack([all_notes[key] for key in ['pitch', 'chord']], axis=1)
train_notes = np.asarray(train_notes).astype('float32')
train_times = np.stack([all_times[key] for key in ['duration', 'start']], axis=1)
train_times = np.asarray(train_times).astype('float32')
#print(train_notes)
notes_ds = tf.data.Dataset.from_tensor_slices(train_notes)
notes_ds.element_spec
times_ds = tf.data.Dataset.from_tensor_slices(train_notes)
times_ds.element_spec

def create_sequences(
    key_order,
    dataset: tf.data.Dataset, 
    seq_length: int,
    vocab_size = 128,
) -> tf.data.Dataset:
  """Returns TF Dataset of sequence and label examples."""
  seq_length = seq_length+1

  # Take 1 extra for the labels
  windows = dataset.window(seq_length, shift=1, stride=1,
                              drop_remainder=True)

  # `flat_map` flattens the" dataset of datasets" into a dataset of tensors
  flatten = lambda x: x.batch(seq_length, drop_remainder=True)
  sequences = windows.flat_map(flatten)

  # Normalize note pitch
  def scale_pitch(x):
    if key_order == ['pitch', 'chord']:
        x = x/[vocab_size,vocab_size]
    else:
        x = x/[1.0, 1.0]
    return x

  # Split the labels
  def split_labels(sequences):
    inputs = sequences[:-1]
    print(inputs)
    labels_dense = sequences[-1]
    labels = {key:labels_dense[i] for i,key in enumerate(key_order)}
    return scale_pitch(inputs), labels

  return sequences.map(split_labels, num_parallel_calls=tf.data.AUTOTUNE)

seq_length = 25 #hp
vocab_size = 128
note_seq_ds = create_sequences(['pitch', 'chord'], notes_ds, seq_length, vocab_size)
note_seq_ds.element_spec
time_seq_ds = create_sequences(['duration', 'start'], times_ds, seq_length, vocab_size)
time_seq_ds.element_spec

print(note_seq_ds,time_seq_ds)

for seq, target in note_seq_ds.take(1):
  print('sequence shape:', seq.shape)
  print('sequence elements (first 10):', seq[0: 10])
  print()
  print('target:', target)

for seq, target in time_seq_ds.take(1):
  print('sequence shape:', seq.shape)
  print('sequence elements (first 10):', seq[0: 10])
  print()
  print('target:', target)

#configure model

batch_size = 64
buffer_size = len(all_notes) - seq_length  # the number of items in the dataset
note_train_ds = (note_seq_ds
            .shuffle(buffer_size)
            .batch(batch_size, drop_remainder=True)
            .cache()
            .prefetch(tf.data.experimental.AUTOTUNE))
note_train_ds.element_spec
note_train_ds = np.asarray(note_train_ds)

time_train_ds = (time_seq_ds
            .shuffle(buffer_size)
            .batch(batch_size, drop_remainder=True)
            .cache()
            .prefetch(tf.data.experimental.AUTOTUNE))
time_train_ds.element_spec
time_train_ds = np.asarray(time_train_ds)

def mse_with_positive_pressure(y_true: tf.Tensor, y_pred: tf.Tensor):
  mse = (y_true - y_pred) ** 2
  positive_pressure = 10 * tf.maximum(-y_pred, 0.0)
  return tf.reduce_mean(mse + positive_pressure)

#input_shape = (seq_length, 4)
learning_rate = 0.005 #hp

input_shape = (seq_length,2)

timeInputs = tf.keras.Input(input_shape)
pitchInputs = tf.keras.Input(input_shape)

#print(np.shape(pitchInputs))
pmodel1 = tf.keras.layers.LSTM(128)(pitchInputs)
pmodel2 = tf.keras.layers.Dense(128)(pmodel1)
pmodel3 = tf.keras.layers.Dense(25)(pmodel2)

tmodel1 = tf.keras.layers.Dense(1, activation=keras.activations.relu, use_bias=True)(timeInputs)

merged = tf.keras.layers.Concatenate(axis=1)([tmodel1, pmodel3])

outputs = {
    'pitch':tf.keras.layers.Dense(1, activation=keras.activations.relu, use_bias=True)(merged),
    'start':tf.keras.layers.Dense(1, activation=keras.activations.relu, use_bias=True)(tmodel1),
    'duration':tf.keras.layers.Dense(1, activation=keras.activations.relu, use_bias=True)(merged)
}
#x = tf.keras.layers.LSTM(128)(inputs)
#x = tf.keras.layers.Dense(128)(x)

loss = {
    'pitch':"mean_squared_error",
    'start':mse_with_positive_pressure,
    'duration':mse_with_positive_pressure
}

model = tf.keras.models.Model([timeInputs,pitchInputs],outputs)

#input1 = tf.keras.layers.Input(shape=(1, ))
#input2 = tf.keras.layers.Input(shape=(1,))
#merged = tf.keras.layers.Concatenate(axis=1)([input1, input2])
#dense1 = tf.keras.layers.Dense(2, input_dim=2, activation=keras.activations.sigmoid, use_bias=True)(merged)
#output = tf.keras.layers.Dense(1, activation=keras.activations.relu, use_bias=True)(dense1)
#model10 = tf.keras.models.Model(inputs=[input1, input2], output=output)

#outputs = {
  #'pitch': tf.keras.layers.Dense(128, name='pitch')(x),
  ##'beat': tf.keras.layers.Dense(1, name='beat')(x),
  #'duration': tf.keras.layers.Dense(1, name='duration')(x),
  ##'bar': tf.keras.layers.Dense(1, name='bar')(x),
  #'start': tf.keras.layers.Dense(1, name='start')(x),
  #'chord': tf.keras.layers.Dense(128, name='chord')(x)
#}

#model = tf.keras.Model(inputs, outputs)#[outPitch, outStep, outDur])

#loss = {
#     'pitch': "mean_squared_error",#tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),#
      #'duration': mse_with_positive_pressure,#Used to be mse with positive pressure
      #'start': mse_with_positive_pressure,#same
      #'chord': "mean_squared_error"
#}

optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

#model.compile(loss=loss, optimizer=optimizer)

model.summary()

model.compile(
    loss=loss,
    loss_weights={
        'pitch': 10,#0.05
        'duration':0.25,#1.0
        'start': 0.25,#1.0
        #'chord': 10,
    },
    optimizer=optimizer,
)

model.evaluate([note_train_ds, time_train_ds], return_dict=True)
callbacks = [
    tf.keras.callbacks.ModelCheckpoint(
        filepath='./training_checkpoints/ckpt_{epoch}',
        save_weights_only=True),
    tf.keras.callbacks.EarlyStopping(
        monitor='loss',
        patience=5,
        verbose=1,
        restore_best_weights=True),
]

epochs = 20

history = model.fit(
    [note_train_ds, time_train_ds],
    epochs=epochs,
    callbacks=callbacks,
)

# serialize model to JSON
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
#serialize weights to HDF5
save_model(model, "model.h5")
print("Saved model to disk")

plt.plot(history.epoch, history.history['loss'], label='total loss')
plt.show()