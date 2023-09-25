import numpy as np
from keras.models import Sequential, load_model
from keras.layers import LSTM, Dense, Dropout
from music21 import converter, instrument, note, chord, stream
import tensorflow as tf

# Load music data
midi = converter.parse('Debussy_Reverie.mid')

# Extract notes and chords
notes = []
for element in midi.flat:
    if isinstance(element, note.Note):
        notes.append(str(element.pitch))
    elif isinstance(element, chord.Chord):
        notes.append('.'.join(str(n) for n in element.normalOrder))
# Define vocabulary
pitchnames = sorted(set(item for item in notes))
note_to_int = dict((note, number) for number, note in enumerate(pitchnames))
# Convert notes to integers
sequence_length = 100
network_input = []
network_output = []
for i in range(0, len(notes) - sequence_length, 1):
    sequence_in = notes[i:i + sequence_length]
    sequence_out = notes[i + sequence_length]
    network_input.append([note_to_int[char] for char in sequence_in])
    network_output.append(note_to_int[sequence_out])
n_patterns = len(network_input)
n_vocab = len(set(notes))
# Reshape input data
X = np.reshape(network_input, (n_patterns, sequence_length, 1))
X = X / float(n_vocab)
# One-hot encode output data
y = tf.keras.utils.to_categorical(network_output)

model = load_model('model.h5')
print("Loaded model from disk")

model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

model.summary()

# Define model
#model = Sequential()
#model.add(LSTM(512, input_shape=(X.shape[1], X.shape[2]), return_sequences=True))
#model.add(Dropout(0.3))
#model.add(LSTM(512))
#model.add(Dense(256))
#model.add(Dropout(0.3))
#model.add(Dense(n_vocab, activation='softmax'))

# Compile model
#model.compile(loss='categorical_crossentropy', optimizer='adam')

# Train model
#model.fit(X, y, epochs=100, batch_size=64)

# Generate new music
start = np.random.randint(0, len(network_input)-1)
int_to_note = dict((number, note) for number, note in enumerate(pitchnames))
pattern = network_input[start]
prediction_output = []

# Generate notes
for note_index in range(500):
    prediction_input = np.reshape(pattern, (1, len(pattern), 1))
    prediction_input = prediction_input / float(n_vocab)
    prediction = model.predict(prediction_input, verbose=0)
    index = np.argmax(prediction)
    result = int_to_note[index]
    prediction_output.append(result)
    pattern.append(index)
    pattern = pattern[1:len(pattern)]

# Create MIDI file
offset = 0
output_notes = []
for pattern in prediction_output:
    if ('.' in pattern) or pattern.isdigit():
        notes_in_chord = pattern.split('.')
        notes = []
        for current_note in notes_in_chord:
            new_note = note.Note(int(current_note))
            new_note.storedInstrument = instrument.Piano()
            notes.append(new_note)
        new_chord = chord.Chord(notes)
        new_chord.offset = offset
        output_notes.append(new_chord)
    else:
        new_note = note.Note(int(pattern))
        new_note.offset = offset
        new_note.storedInstrument = instrument.Piano()
        output_notes.append(new_note)
    offset += 0.5

midi_stream = stream.Stream(output_notes)
midi_stream.write('midi', fp='output.mid')