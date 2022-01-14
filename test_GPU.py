import tensorflow.keras as keras
from keras_dna import Generator, ModelWrapper, MultiGenerator

from keras_dna.sequence import SeqIntervalDl, StringSeqIntervalDl
import os

os.chdir('tutorial_dr')

#%%
model = keras.models.Sequential()

model.add(keras.layers.Conv1D(16, 3, activation='relu', input_shape=(2001, 4)))
model.add(keras.layers.MaxPooling1D(2))

model.add(keras.layers.Conv1D(32, 10, activation='relu'))
model.add(keras.layers.MaxPooling1D(2))

model.add(keras.layers.Conv1D(64, 20, activation='relu'))
model.add(keras.layers.MaxPooling1D(2))

model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(1, activation='linear'))
          
model.compile(loss='mse', optimizer='adam', metrics=['mae'])

#%%
early = keras.callbacks.EarlyStopping(monitor='val_loss',
                                      min_delta=0,
                                      patience=10,
                                      verbose=0,
                                      mode='auto')

#%%
generator_train = Generator(batch_size=64,
                            fasta_file='sacCer3.fa',
                            annotation_files=['scerevisiae.bw'],
                            window=2001,
                            incl_chromosomes=['chrI', 'chrII', 'chrIII', 'chrIV', 'chrV', 'chrVI'],
                            output_shape=(64, 1))

#%%
wrap = ModelWrapper(model=model,
                    generator_train=generator_train,
                    validation_chr=['chrVII', 'chrVIII'])

#%%
wrap.train(epochs=10,
           steps_per_epoch=50,
           validation_steps=20,
           callbacks=[early])

#%%
wrap.evaluate(incl_chromosomes=['chrM'], verbose=1)

#%%
wrap.get_correlation(incl_chromosomes=['chrM'], verbose=1)