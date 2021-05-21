class ParaSineLayer(tf.keras.layers.Layer):
    def __init__(self, in_features, units, bias=True, is_first=False, omega_0=np.pi ** 3):
        super(ParaSineLayer, self).__init__()
        self.in_features = in_features
        self.units = units
        self.is_first = is_first
        self.omega_0 = omega_0

        self.dense = tf.keras.layers.Dense(self.units,
                                           use_bias=bias,
                                           kernel_initializer=self.init_weights(),
                                           input_shape=(self.in_features,))
        
    
    def init_weights(self):
        if self.is_first:
            return tf.keras.initializers.RandomUniform(minval=-1 / np.pi ** 5,
                                                       maxval= 1 / np.pi ** 5)
        else:
            return tf.keras.initializers.RandomUniform(minval=-np.sqrt(6. / self.in_features) / self.omega_0,
                                                       maxval= np.sqrt(6. / self.in_features) / self.omega_0)
    

    def build(self, input_shape):

        self.a_1 = self.add_weight(
            name='a_1',
            shape=(self.units,),
            initializer='zeros',
            trainable=True)

        self.a0 = self.add_weight(
            name='a0',
            shape=(self.units,),
            initializer='ones',
            trainable=True)
        self.w0 = self.add_weight(
            name='w0',
            shape=(self.units,),
            initializer='ones',
            trainable=True)
        self.shift0 = self.add_weight(
            name='shift0',
            shape=(self.units,),
            initializer='zeros',
            trainable=True)

        self.a1 = self.add_weight(
            name='a1',
            shape=(self.units,),
            initializer='ones',
            trainable=True)
        self.w1 = self.add_weight(
            name='w1',
            shape=(self.units,),
            initializer='ones',
            trainable=True)
        self.shift1 = self.add_weight(
            name='shift1',
            shape=(self.units,),
            initializer='zeros',
            trainable=True)


        super(ParaSineLayer, self).build(input_shape)


    def call(self, input_tensor):
        befor_activation = self.dense(input_tensor)
        after_activation = self.a_1 * self.omega_0 * befor_activation + \
                           self.a0 * tf.sin(self.w0 * self.omega_0 * befor_activation + self.shift0) + \
                           self.a1 * tf.cos(self.w1 * self.omega_0 * befor_activation + self.shift1)
        return after_activation
    
    
    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.units)





def ParaSINNetwork():
    tf.keras.backend.clear_session()

    inputs = layers.Input(shape=(784,))
    
    ## Parametric
    X = ParaSineLayer(784, 256, is_first=True)(inputs)
    
    ## Gelu
    #X = layers.Dense(1024, activation=None, use_bias=False, kernel_initializer=tf.keras.initializers.RandomUniform(minval=-1 / np.pi ** 5, maxval=-1 / np.pi ** 5))(inputs)
    #X = layers.Lambda(lambda x: tf.nn.gelu(x) * np.pi ** 3)(X)

    features = layers.Dropout(0.5)(X)
    features = X
    
    # Classify outputs.
    logits = layers.Dense(10)(features)
    # Create the Keras model.
    model = keras.Model(inputs=inputs, outputs=logits)
    
    return model

parasin_classifier = ParaSINNetwork()
parasin_classifier.summary()
