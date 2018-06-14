#Discriminator Modification

#Need a Resedual Block




#Also can use an embeddiing layer to represent action 
#So the each action is represented with some sort of 

#I have for actions 
#1.Moveforward #2.turn right #3.turn left #4.Movebackword   

#Each of the  so we need to have 4 different embedding vectors 
#It is crusial to train these embeddings continuously with 

#Ebedding matrix should be get the batch output from the Policy generator output 
encoder_inputs = tf.placeholder(shape=(None, None), dtype=tf.int32, name='encoder_inputs') #This can be the output from the 
embeddings = tf.Variable(tf.random_uniform([action_size, embedding_size], -1.0, 1.0), dtype=tf.float32)



encoder_inputs_embedded = tf.nn.embedding_lookup(embeddings, GEN_PI)


#The we can use a residual layer to merge with embeddings with target and the policy

def residual_unit(layer):
    '''
    Input(s): layer - conv layer before this res unit
    
    Output(s): ResUnit layer - implemented as described in the paper
    '''
    step1 = tf.layers.batch_normalization(layer)
    step2 = tf.nn.relu(step1)
    step3 = conv2d_custom(step2, 3, 32, 32, activation=None, max_pool=False) #32 number of feautres is hyperparam
    step4 = tf.layers.batch_normalization(step3)
    step5 = tf.nn.relu(step4)
    step6 = conv2d_custom(step5, 3, 32, 32, activation=None, max_pool=False)
    return layer + step6  #Should have the same spacial dimentions

