{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 2,
   "id": "cb5ebfa8-9d80-4ed4-bcea-757014b838f5",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "C:\\Users\\Rakesh G N\\anaconda3\\Lib\\site-packages\\keras\\src\\layers\\convolutional\\base_conv.py:113: UserWarning: Do not pass an `input_shape`/`input_dim` argument to a layer. When using Sequential models, prefer using an `Input(shape)` object as the first layer in the model instead.\n",
      "  super().__init__(activity_regularizer=activity_regularizer, **kwargs)\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "🔁 Training emotion model...\n",
      "Epoch 1/15\n",
      "\u001b[1m359/359\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m25s\u001b[0m 59ms/step - accuracy: 0.3161 - loss: 1.7086\n",
      "Epoch 2/15\n",
      "\u001b[1m359/359\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m21s\u001b[0m 59ms/step - accuracy: 0.4135 - loss: 1.5276\n",
      "Epoch 3/15\n",
      "\u001b[1m359/359\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m22s\u001b[0m 62ms/step - accuracy: 0.4496 - loss: 1.4387\n",
      "Epoch 4/15\n",
      "\u001b[1m359/359\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m22s\u001b[0m 62ms/step - accuracy: 0.4770 - loss: 1.3681\n",
      "Epoch 5/15\n",
      "\u001b[1m359/359\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m23s\u001b[0m 63ms/step - accuracy: 0.4982 - loss: 1.3187\n",
      "Epoch 6/15\n",
      "\u001b[1m359/359\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m24s\u001b[0m 66ms/step - accuracy: 0.5213 - loss: 1.2636\n",
      "Epoch 7/15\n",
      "\u001b[1m359/359\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m24s\u001b[0m 66ms/step - accuracy: 0.5407 - loss: 1.2125\n",
      "Epoch 8/15\n",
      "\u001b[1m359/359\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m23s\u001b[0m 63ms/step - accuracy: 0.5589 - loss: 1.1630\n",
      "Epoch 9/15\n",
      "\u001b[1m359/359\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m23s\u001b[0m 64ms/step - accuracy: 0.5787 - loss: 1.1114\n",
      "Epoch 10/15\n",
      "\u001b[1m359/359\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m22s\u001b[0m 62ms/step - accuracy: 0.6043 - loss: 1.0533\n",
      "Epoch 11/15\n",
      "\u001b[1m359/359\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m22s\u001b[0m 62ms/step - accuracy: 0.6263 - loss: 1.0000\n",
      "Epoch 12/15\n",
      "\u001b[1m359/359\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m23s\u001b[0m 63ms/step - accuracy: 0.6406 - loss: 0.9490\n",
      "Epoch 13/15\n",
      "\u001b[1m359/359\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m22s\u001b[0m 62ms/step - accuracy: 0.6586 - loss: 0.9013\n",
      "Epoch 14/15\n",
      "\u001b[1m359/359\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m24s\u001b[0m 67ms/step - accuracy: 0.6808 - loss: 0.8511\n",
      "Epoch 15/15\n",
      "\u001b[1m359/359\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m25s\u001b[0m 68ms/step - accuracy: 0.7035 - loss: 0.7887\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "WARNING:absl:You are saving your model as an HDF5 file via `model.save()` or `keras.saving.save_model(model)`. This file format is considered legacy. We recommend using instead the native Keras format, e.g. `model.save('my_model.keras')` or `keras.saving.save_model(model, 'my_model.keras')`. \n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "✅ emotion_model.h5 SAVED SUCCESSFULLY\n"
     ]
    }
   ],
   "source": [
    "import os\n",
    "import cv2\n",
    "import numpy as np\n",
    "from tensorflow.keras.models import Sequential\n",
    "from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout\n",
    "from tensorflow.keras.utils import to_categorical\n",
    "from sklearn.model_selection import train_test_split\n",
    "\n",
    "IMG_SIZE = 48\n",
    "EMOTIONS = [\"angry\", \"disgust\", \"fear\", \"happy\", \"sad\", \"surprise\", \"neutral\"]\n",
    "\n",
    "data = []\n",
    "labels = []\n",
    "\n",
    "for idx, emotion in enumerate(EMOTIONS):\n",
    "    folder = os.path.join(\"dataset/emotion\", emotion)\n",
    "    for img in os.listdir(folder):\n",
    "        img_path = os.path.join(folder, img)\n",
    "        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)\n",
    "        if image is None:\n",
    "            continue\n",
    "        image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))\n",
    "        data.append(image)\n",
    "        labels.append(idx)\n",
    "\n",
    "data = np.array(data).reshape(-1, IMG_SIZE, IMG_SIZE, 1) / 255.0\n",
    "labels = to_categorical(labels)\n",
    "\n",
    "X_train, X_test, y_train, y_test = train_test_split(\n",
    "    data, labels, test_size=0.2, random_state=42\n",
    ")\n",
    "\n",
    "model = Sequential([\n",
    "    Conv2D(32, (3, 3), activation=\"relu\", input_shape=(48, 48, 1)),\n",
    "    MaxPooling2D(2, 2),\n",
    "    Conv2D(64, (3, 3), activation=\"relu\"),\n",
    "    MaxPooling2D(2, 2),\n",
    "    Flatten(),\n",
    "    Dense(256, activation=\"relu\"),\n",
    "    Dropout(0.5),\n",
    "    Dense(len(EMOTIONS), activation=\"softmax\")\n",
    "])\n",
    "\n",
    "model.compile(\n",
    "    optimizer=\"adam\",\n",
    "    loss=\"categorical_crossentropy\",\n",
    "    metrics=[\"accuracy\"]\n",
    ")\n",
    "\n",
    "print(\"🔁 Training emotion model...\")\n",
    "model.fit(X_train, y_train, epochs=15, batch_size=64)\n",
    "\n",
    "model.save(\"model/emotion_model.h5\")\n",
    "print(\"✅ emotion_model.h5 SAVED SUCCESSFULLY\")\n"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.13.5"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
