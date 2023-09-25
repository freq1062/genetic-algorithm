import collections
import datetime
import glob
import numpy as np
import pathlib
import pandas as pd
import pretty_midi
import seaborn as sns
import tensorflow as tf
#import extract_single as es
from keras.models import save_model
import time
import sys
import keras
from tqdm.auto import tqdm

from IPython import display
from matplotlib import pyplot as plt
from typing import Optional
from music21 import converter, corpus, instrument, midi, note, chord, pitch
import music21

all_notes = []
all_times = []

data_dir = pathlib.Path('data/maestro-v2.0.0')
if not data_dir.exists():
  tf.keras.utils.get_file(
      'maestro-v2.0.0-midi.zip',
      origin='https://storage.googleapis.com/magentadata/datasets/maestro/v2.0.0/maestro-v2.0.0-midi.zip',
      extract=True,
      cache_dir='.', cache_subdir='data',
  )

filenames = glob.glob(str(data_dir/'**/*.mid*'))
print('Number of files:', len(filenames))

#filename = "Debussy_Reverie.mid"

def open_midi(filename):
    # There is an one-line method to read MIDIs
    # but to remove the drums we need to manipulate some
    # low level MIDI events.
    mf = midi.MidiFile()
    mf.open(filename)
    mf.read()
    mf.close()

    return midi.translate.midiFileToStream(mf)

#file = open_midi(filename)

def list_instruments(midi):
    partStream = midi.parts.stream()
    print("List of instruments found on MIDI file:")
    for p in partStream:
        print (p.partName)

def getChord(notes, keySig):
    for j in range(0, len(notes)):
        if any([x in notes[j] for x in list("1234567890")]):
            notes[j] = "".join([notes[j][i] for i in range(len(notes[j])-1)])
    #Get the notes in a scale of the key
    keyNum = music21.key.Key(keySig).sharps
    accidentals = ["F", "C", "G", "D", "A", "E", "B"]
    if keyNum <0:
        accidentals = accidentals[::-1]
    octave = ["A", "B", "C", "D", "E", "F", "G"]
    for i in range(0, abs(keyNum)):
        index = octave.index(accidentals[i])
        if keyNum<0:
            octave[index] = octave[index]+"-"
        elif keyNum>0:
            octave[index] = octave[index]+"#"
    for j in range(0, octave.index(keySig.upper())):
        octave.append(octave[0])
        octave.pop(0)
    #Build all the possible chords and check similarity to entered notes
    occurences = collections.Counter(notes)
    notes = list(dict.fromkeys(notes))
    max = [str(keySig), 0]
    score = 0
    for i in range(0, len(notes)):
        if notes[i] in octave:
            if octave[(octave.index(notes[i])+2) % len(octave)] in notes: #3rd
                score+=1 * occurences[octave[(octave.index(notes[i])+2) % len(octave)]]
            if octave[(octave.index(notes[i])+4) % len(octave)] in notes: #5th
                score+=1 * occurences[octave[(octave.index(notes[i])+4) % len(octave)]]
            if octave[(octave.index(notes[i])+6) % len(octave)] in notes: #7th
                score+=0.5 * occurences[octave[(octave.index(notes[i])+6) % len(octave)]]
            if score > max[1]:
                max = [notes[i], score]
        score = 0
    return pitch.Pitch(str(max[0])+"0").midi

def extract_notes(midi_part, key):
    notes = collections.defaultdict(list)
    prevBeat = 0
    barCounter = 0
    chords = []
    barNotes = []
    #key = str(music21.converter.parse(filename).analyze("key"))
    if "major" in key: key = key[0] 
    else: key = key[0].lower()
    for nt in tqdm(midi_part.flatten().notes[:200]):     
        if isinstance(nt, note.Note):
            notes['pitch'].append(nt.pitch.midi)
            #notes['beat'].append(nt.beat)
            notes["duration"].append(nt.duration.quarterLength)
            if prevBeat > nt.beat:
                chords.append(getChord(barNotes, key))
                barNotes = []
                barNotes.append(str(nt.pitch))
                barCounter+=1
            else: barNotes.append(str(nt.pitch))
            notes["bar"].append(barCounter)
            notes['start'].append(barCounter*4+nt.beat)
            prevBeat = nt.beat
        elif isinstance(nt, chord.Chord):
            for pitch in nt.pitches:
                notes['pitch'].append(pitch.midi)
                #notes['beat'].append(nt.beat)
                notes["duration"].append(nt.duration.quarterLength)
                if prevBeat > nt.beat:
                    chords.append(getChord(barNotes, key))
                    barNotes = []
                    barNotes.append(str(pitch))
                    barCounter+=1
                else: barNotes.append(str(pitch))
                notes["bar"].append(barCounter)
                notes['start'].append(barCounter*4+nt.beat)
                prevBeat = nt.beat
    chords.append(getChord(barNotes, key))
    for dat in notes["bar"]:
        notes["chord"].append(chords[dat])
    del notes["bar"]
    #sns.pairplot(pd.DataFrame({name: np.array(value) for name, value in notes.items()}).astype(float))
    #plt.show()
    return (pd.DataFrame({name: np.array(value) for name, value in notes.items()}).astype(float))
#['pitch','duration','bar','start','chord']

def mse_with_positive_pressure(y_true: tf.Tensor, y_pred: tf.Tensor):
  mse = (y_true - y_pred) ** 2
  positive_pressure = 10 * tf.maximum(-y_pred, 0.0)
  return tf.reduce_mean(mse + positive_pressure)

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
    x = x/[vocab_size,1.0,1.0]
    return x

  # Split the labels
  def split_labels(sequences):
    inputs = sequences[:-1]
    print(inputs)
    labels_dense = sequences[-1]
    labels = {key:labels_dense[i] for i,key in enumerate(key_order)}
    return scale_pitch(inputs), labels

  return sequences.map(split_labels, num_parallel_calls=tf.data.AUTOTUNE)

for f in filenames[:1]: #How many songs to train from
  data = extract_notes(open_midi(f), str(music21.converter.parse(f).analyze("key")))
  all_notes.append(data)
  print("Finished file",filenames.index(f))

all_notes = pd.concat(all_notes)
print('Number of notes parsed:', len(all_notes))
#print(all_notes[0:3], all_times[0:3])
#exit()

normalizer = tf.keras.layers.Normalization(axis=-1)
normalizer.adapt(np.array(all_notes))
#print(normalizer.mean.numpy())

seq_length = 25
train_notes = np.stack([all_notes[key] for key in ['pitch', 'start', 'duration']], axis=1)
notes_ds = tf.data.Dataset.from_tensor_slices(train_notes)
notes_ds = create_sequences(['pitch', 'chord'], notes_ds, seq_length, 128)
train_ds = (notes_ds
            .shuffle(len(all_notes) - seq_length)
            .batch(128, drop_remainder=True)
            .cache()
            .prefetch(tf.data.experimental.AUTOTUNE))
#model = tf.keras.Sequential([
#    normalizer,
#    tf.keras.layers.Dense(units=1)
#])

input_shape = (seq_length,3)
inputs = tf.keras.Input(input_shape)
x = tf.keras.layers.LSTM(128)(inputs)
x = tf.keras.layers.Dense(128)(x)

outputs = {
    'pitch':tf.keras.layers.Dense(1, name='pitch')(x)#(tf.keras.layers.Concatenate()([x])),
    #'start':tf.keras.layers.Dense(1, name='start')(x),
    #'duration':tf.keras.layers.Dense(1, name='duration')(x),
    #'chord':tf.keras.layers.Dense(1, name='chord')(x)
}

def custom_loss(y_true: tf.Tensor, y_pred: tf.Tensor):
    squared_difference = tf.square(y_true-y_pred)
    return tf.reduce_mean(squared_difference, axis=-1)

losses = {
    'pitch':"mean_squared_error",
    'start':mse_with_positive_pressure,
    'duration':mse_with_positive_pressure,
    'chord':"mean_absolute_error"
}

optimizer = tf.keras.optimizers.Adam(learning_rate=0.005)

model = tf.keras.models.Model(inputs,outputs)

model.summary()

model.compile(
    loss=losses,
    loss_weights={
        'pitch': 1,#0.05
        'start': 0.25,#1.0
        'duration':0.25,#1.0
        'chord':0.25,
    },
    optimizer=optimizer,
)

model.evaluate(train_ds, return_dict=True)

callbacks = [
    tf.keras.callbacks.ModelCheckpoint(
        filepath='./training_checkpoints/ckpt_{epoch}',
        save_weights_only=True),
    tf.keras.callbacks.EarlyStopping(
        monitor='loss',
        patience=10,
        verbose=1,
        restore_best_weights=True),
]

history = model.fit(
    train_ds,
    epochs=20,
    callbacks=callbacks,
)

model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
#serialize weights to HDF5
save_model(model, "model.h5")
print("Saved model to disk")

plt.plot(history.epoch, history.history['loss'], label='total loss')
#plt.plot(history.epoch, history.history['accuracy'], label='accuracy')
plt.show()

#model.predict(all_notes[:128])