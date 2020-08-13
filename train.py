
from keras.models import Sequential   #Used to generate CNN sequentially
from keras.layers import Dense, Flatten,Convolution2D, MaxPooling2D   #Different layers used

classifier = Sequential()

classifier.add(Convolution2D(32, (3, 3), input_shape= (64,64,3),activation='relu'))

classifier.add(MaxPooling2D(pool_size=(2,2)))

# Adding a second convolutional layer
#classifier.add(Convolution2D(64, 3, 3, activation = 'relu'))
#classifier.add(MaxPooling2D(pool_size = (2, 2)))

classifier.add(Flatten())

classifier.add(Dense(output_dim = 128, activation='relu'))

classifier.add(Dense(output_dim = 4, activation='softmax'))

#classifier.add(Dense(output_dim = 3, activation='sigmoid'))


classifier.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics=['accuracy'])

from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory('data/train',
                                                 target_size=(64,64),color_mode="rgb",
                                                 batch_size=64,
                                                 class_mode='binary')

test_set = test_datagen.flow_from_directory('data/test',
                                            target_size=(64,64),color_mode="rgb",
                                            batch_size=64,
                                            class_mode='binary')

classifier.fit_generator(training_set,
                         steps_per_epoch= 341,
                         nb_epoch = 1,
                         validation_data=test_set,
                         nb_val_samples = 134,
                         )

classifier.save('model.h5')
print("Model saved")
input("press enter key to continue")
