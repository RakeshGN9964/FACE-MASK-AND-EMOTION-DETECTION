{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "09586209-bf9b-4597-b550-53b1b92a80c9",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "C:\\Users\\Rakesh G N\\anaconda3\\Lib\\site-packages\\keras\\src\\export\\tf2onnx_lib.py:8: FutureWarning: In the future `np.object` will be defined as the corresponding NumPy scalar.\n",
      "  if not hasattr(np, \"object\"):\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "🔁 Training started...\n",
      "Epoch 1/10\n",
      "\u001b[1m189/189\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m192s\u001b[0m 931ms/step - accuracy: 0.8941 - loss: 0.2845\n",
      "Epoch 2/10\n",
      "\u001b[1m189/189\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m196s\u001b[0m 1s/step - accuracy: 0.9793 - loss: 0.0852\n",
      "Epoch 3/10\n",
      "\u001b[1m189/189\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m165s\u001b[0m 874ms/step - accuracy: 0.9818 - loss: 0.0633\n",
      "Epoch 4/10\n",
      "\u001b[1m189/189\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m165s\u001b[0m 873ms/step - accuracy: 0.9848 - loss: 0.0475\n",
      "Epoch 5/10\n",
      "\u001b[1m189/189\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m168s\u001b[0m 887ms/step - accuracy: 0.9882 - loss: 0.0416\n",
      "Epoch 6/10\n",
      "\u001b[1m189/189\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m189s\u001b[0m 996ms/step - accuracy: 0.9894 - loss: 0.0348\n",
      "Epoch 7/10\n",
      "\u001b[1m189/189\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m166s\u001b[0m 879ms/step - accuracy: 0.9899 - loss: 0.0331\n",
      "Epoch 8/10\n",
      "\u001b[1m189/189\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m160s\u001b[0m 847ms/step - accuracy: 0.9912 - loss: 0.0287\n",
      "Epoch 9/10\n",
      "\u001b[1m189/189\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m165s\u001b[0m 870ms/step - accuracy: 0.9922 - loss: 0.0282\n",
      "Epoch 10/10\n",
      "\u001b[1m189/189\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m164s\u001b[0m 867ms/step - accuracy: 0.9944 - loss: 0.0221\n"
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
      "✅ mask_detector.h5 SAVED SUCCESSFULLY\n"
     ]
    }
   ],
   "source": [
    "import os\n",
    "import cv2\n",
    "import numpy as np\n",
    "from tensorflow.keras.applications import MobileNetV2\n",
    "from tensorflow.keras.layers import AveragePooling2D, Dense, Flatten, Dropout\n",
    "from tensorflow.keras.models import Model\n",
    "from tensorflow.keras.optimizers import Adam\n",
    "from tensorflow.keras.utils import to_categorical\n",
    "from sklearn.model_selection import train_test_split\n",
    "from sklearn.preprocessing import LabelBinarizer\n",
    "\n",
    "IMG_SIZE = 224\n",
    "EPOCHS = 10\n",
    "BS = 32\n",
    "\n",
    "data = []\n",
    "labels = []\n",
    "\n",
    "base_dir = \"dataset/mask\"\n",
    "\n",
    "for label in [\"with_mask\", \"without_mask\"]:\n",
    "    folder = os.path.join(base_dir, label)\n",
    "    for img in os.listdir(folder):\n",
    "        img_path = os.path.join(folder, img)\n",
    "        image = cv2.imread(img_path)\n",
    "        if image is None:\n",
    "            continue\n",
    "        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)\n",
    "        image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))\n",
    "        data.append(image)\n",
    "        labels.append(label)\n",
    "\n",
    "data = np.array(data, dtype=\"float32\") / 255.0\n",
    "labels = np.array(labels)\n",
    "\n",
    "lb = LabelBinarizer()\n",
    "labels = lb.fit_transform(labels)\n",
    "labels = to_categorical(labels)\n",
    "\n",
    "X_train, X_test, y_train, y_test = train_test_split(\n",
    "    data, labels, test_size=0.2, random_state=42\n",
    ")\n",
    "\n",
    "baseModel = MobileNetV2(\n",
    "    weights=\"imagenet\",\n",
    "    include_top=False,\n",
    "    input_shape=(IMG_SIZE, IMG_SIZE, 3)\n",
    ")\n",
    "\n",
    "for layer in baseModel.layers:\n",
    "    layer.trainable = False\n",
    "\n",
    "head = baseModel.output\n",
    "head = AveragePooling2D(pool_size=(7, 7))(head)\n",
    "head = Flatten()(head)\n",
    "head = Dense(128, activation=\"relu\")(head)\n",
    "head = Dropout(0.5)(head)\n",
    "head = Dense(2, activation=\"softmax\")(head)\n",
    "\n",
    "model = Model(inputs=baseModel.input, outputs=head)\n",
    "\n",
    "model.compile(\n",
    "    optimizer=Adam(learning_rate=1e-4),\n",
    "    loss=\"binary_crossentropy\",\n",
    "    metrics=[\"accuracy\"]\n",
    ")\n",
    "\n",
    "print(\"🔁 Training started...\")\n",
    "model.fit(X_train, y_train, epochs=EPOCHS, batch_size=BS)\n",
    "\n",
    "model.save(\"model/mask_detector.h5\")\n",
    "print(\"✅ mask_detector.h5 SAVED SUCCESSFULLY\")\n"
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
