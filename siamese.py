import tensorflow as tf
class S_Layer:
	def __init__(self):
		print("Creating a New Siamese Class")

	def siamese_f(self,state,target):
		print("I am inside the siamese_f")
		self.siamese_s=self.construct_Siamese(state)
		self.siamese_t=self.construct_Siamese(target)
		self.obs=tf.concat(values=[self.siamese_s, self.siamese_t], axis=1,name='obs')
		return self.obs
		print("shaa")




	'''
	def __init__(self,state,target):
		
		self.siamese_s=self.construct_Siamese(state)
		self.siamese_t=self.construct_Siamese(target)
		self.obs = tf.concat(values=[self.siamese_s, self.siamese_t], axis=1,name='obs')
	'''



                
	def construct_Siamese(self, input):
		print("Building the Siamese with shared variables")
		with tf.variable_scope("Siamese", reuse=tf.AUTO_REUSE):
			layer_1 = tf.layers.dense(inputs=input, units=512, activation=tf.nn.leaky_relu, name='layer1')
			layer_2 = tf.layers.dense(inputs=layer_1, units=512, activation=tf.nn.leaky_relu, name='layer2')
			#v = tf.get_variable("v", [1])
			#print(v)
		return layer_2


        