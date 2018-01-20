import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import tensorflow as tf
from tensorflow.python.keras.layers import Conv2D, Dense, MaxPool2D, Flatten, Dropout


def plot_image(images, points_list, rows, cols, annotate):
    fig, ax = plt.subplots(nrows=rows, ncols=cols)
    for im_j in range(rows):
        for im_i in range(cols):
            image = images[im_i * 2 + im_j]
            image = image.reshape(96, 96)
            points = points_list[im_i * 2 + im_j]
            obj = ax if (rows == 1 and cols == 1) else ax[im_i, im_j]
            obj.imshow(image, cmap='gray')
            obj.scatter(x=points[np.arange(0, 29, 2)], y=points[np.arange(1, 30, 2)], c='r', s=1)

            bb_x = points[18] - (points[10] - points[18]) / 2
            bb_y = points[18] - (points[10] - points[18])
            bb_w = points[14] - points[18] + ((points[10] - points[18]))
            bb_h = points[29] - bb_y + (points[10] - points[18])

            r_x = bb_x if bb_x > 0 else 0
            r_y = bb_y if bb_y > 0 else 0
            r_w = bb_w if (r_x + bb_w) <= 96 else (96 - r_x)
            r_h = bb_h if (bb_h + r_y) <= 96 else (96 - r_y)
            rect = patches.Rectangle((r_x, r_y), r_w, r_h, linewidth=1, edgecolor='r', facecolor='none')
            print(r_x, r_y, r_x + r_w, r_y + r_h)
            obj.add_patch(rect)
    plt.show()


def move_image(img, move):
    shape = img.shape
    w = shape[1]
    h = shape[0]
    base_size = h, w
    base = np.zeros(base_size, dtype=np.uint8)
    base[:] = 255
    if move[0] < 0 and move[1] < 0:
        base[0:h + move[0], 0:w + move[1]] = img[move[0] * -1:h, move[1] * -1:w]
    elif move[0] < 0 and move[1] >= 0:
        base[0:h + move[0], move[1]:w + move[1]] = img[move[0] * -1:h, 0:w - move[1]]
    elif move[0] >= 0 and move[1] < 0:
        base[move[0]:h + move[0], 0:w + move[1]] = img[0:h - move[0], move[1] * -1:w]
    elif move[0] >= 0 and move[1] >= 0:
        base[move[0]:h + move[0], move[1]:w + move[1]] = img[0:h - move[0], 0:w - move[1]]
    return base


class prediction_history(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        for k in range(10):
            print('')
            print(', '.join(str('{0:.2f}'.format(x)) for x in labels[1501 + k:1502 + k][0][0:15]))
            print(', '.join(str('{0:.2f}'.format(x)) for x in self.model.predict(input_X[1501 + k:1502 + k])[0][0:15]))
            print('************************************************************')


dataset = pd.read_csv('../data-sets/face-points-detect/training.csv')

newset = dataset.dropna()
newset.isnull().sum()

newset = dataset.dropna()
newset.isnull().sum()

input_X = np.array([img.split(' ') for img in newset['Image']], dtype=float).reshape(-1, 96, 96, 1) / 255.0
labels = newset[dataset.columns[0:30]].as_matrix()

for i in range(input_X.shape[0]):
    mx = np.random.randint(-50, 50)
    my = np.random.randint(-50, 50)
    input_X[i] = move_image(input_X[i].reshape(96, 96) * 255, (mx, my)).reshape(96, 96, 1) / 255.0
    labels[i][np.arange(0, 30, 2)] = labels[i][np.arange(0, 30, 2)] + my + 50
    labels[i][np.arange(1, 30, 2)] = labels[i][np.arange(1, 30, 2)] + mx + 50
    labels[i] = labels[i] / (96 + 50)


from_i = 0
to_i = from_i + 1500
model = tf.keras.models.Sequential()
model.add(Conv2D(input_shape=(96, 96, 1), filters=128, kernel_size=(5, 5), activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))
model.add(Dropout(0.25))
model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))
model.add(Dropout(0.25))
model.add(Conv2D(filters=48, kernel_size=(3, 3), activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(500, activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(90, activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(30))

model.compile(optimizer=tf.keras.optimizers.RMSprop(), loss=tf.keras.losses.mean_squared_error, metrics=['accuracy'])
model.fit(x=(input_X[from_i:to_i]), y=(labels[from_i:to_i]), epochs=50, batch_size=32, validation_split=0.1, verbose=1, callbacks=[prediction_history()])

plot_image(input_X[1501:1501+16], model.predict(input_X[1501:1501+16])[0:16] * 96, 4, 4, False)

model.save('./keras-model/keras-model-5.h5')

