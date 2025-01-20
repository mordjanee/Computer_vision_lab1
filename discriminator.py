
from tensorflow.keras import layers, Model

class Discriminator(Model):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.flatten = layers.Flatten()
        self.dense1 = layers.Dense(256, activation="relu")
        self.dense2 = layers.Dense(1, activation="sigmoid")  # Output probability (real/fake)

    def call(self, x):
        x = self.flatten(x)
        x = self.dense1(x)
        return self.dense2(x)