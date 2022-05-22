import datetime
import numpy as np
import tensorflow as tf  
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Input, Layer, LSTM


#TODO: Implement multiple write heads
#TODO: Implement multiple memory types
#TODO: Should the multiread/write head weights really be summed? 

# set default precision
tf.keras.backend.set_floatx('float32')


class NTM(Model):
    def __init__(self, read_num, write_num, length=10, output_dim=1, memshape=[10,10], logits=100, training=True,**kwargs):
        # initialise the superclass
        super(NTM, self).__init__(name="Exuberant_Witness",**kwargs)

        # store desired memshape
        self.memrows = memshape[0]
        self.memcols = memshape[1]
        # decay parameter
        self.gamma = 0.95
        # store desired output dimension for read heads
        self.output_dim = output_dim
        # store number of logits for internal network
        self.logits = logits
        # store input length
        self.input_length = length
        # store training flag
        self.training = training
        # number of read heads
        self.read_num = read_num
        #number of write heads
        self.write_num = write_num
        assert self.write_num==1, "Only a single write head is currently supported"
        
        # instantiate the read heads
        self.read_array = []
        for i in range(self.read_num):
            self.read_array.append(RevReadHead(memory_shape=memshape))
        # instantiate the write heads
        self.write_array = []
        for i in range(self.write_num):
            self.write_array.append(RevWriteHead(memory_shape=memshape))
        
        # initialise the memory controller (talks to the read/write heads)
        self.memory_controller = LSTM(units=self.logits, return_state=True, activation='tanh', recurrent_activation="sigmoid", use_bias=True)
        
        # build the initial state of the network
        A = tf.ones(shape=[self.memrows, self.memcols], dtype=tf.float32)*1e-4
        read_weights_sum = tf.zeros(shape=(1, self.memcols), dtype=tf.float32)
        write_weights = tf.zeros(shape=[1, self.memrows], dtype=tf.float32)
        wu = tf.zeros(shape=[1, self.memrows], dtype=tf.float32)
        self.lstm_state = [tf.zeros(shape=[1, self.logits], dtype=tf.float32)]*2


        # assign initial values
        self.prev_state = {"memory": A,
                 "read_weight": read_weights_sum,
                 "write_weight": write_weights,
                 "usage_weight": wu
                 }



    def abc(self, x, training=False, **kwargs):
        """
        Calls the complete neural turing machine on a sequence of inputs.
        """

        # repeatedly apply instance map to the
        output_sequence = tf.scan(fn=lambda init, elem: self.instance_map(elem, training=training), elems=tf.transpose(x, perm=[1,0,2]))

        # remove tranposition of matrices
        return tf.transpose(output_sequence, perm=[1,0,2])


    def __call__(self, x, training=False):
        """ 
        Calls the neural turing machine on a single time slice and outputs the instance
        """

        # retrieve previous state of arry
        A, wr_prev, ww_prev, wu_prev = self.prev_state["memory"], self.prev_state["read_weight"], self.prev_state["write_weight"], self.prev_state["usage_weight"]

        # concatenate all inputs to LSTM head
        controller_input = tf.concat([x, wr_prev], axis=-1)

        # pass input and previous controller state 
        controller_output, controller_h, controller_c = self.memory_controller(controller_input, initial_state=self.lstm_state)

        # package output state
        self.lstm_state = [controller_h, controller_c]

        # get previous least used
        least_indices, wlu_prev = self.least_used(wu_prev)

        # pass the controller output to the read heads and get memory read weights
        read_weights = []
        kts = []
        for i in range(self.read_num):
            # produces the read head weights and keys
            kt, wr = self.read_array[i].read_head_address(controller_output, A)
            kts.append(kt)
            read_weights.append(wr)

        # compute sum over keys and weights #TODO: convert to tf functions
        kts = sum(kts)
        read_weights_sum = sum(read_weights)

        # pass the controller output to the write heads and get memory write weights
        write_weights =  self.write_array[i].write_head_address(controller_output, wr_prev, wlu_prev)

        # given write weight vector, read weight vector and previous usage weight, get new update weights
        wu = self.gamma*wu_prev+read_weights_sum + write_weights

        # zero least used memory
        A = A * tf.expand_dims(1 - tf.one_hot(least_indices))

        # write to memory using least access information and sum over batches
        A = tf.tanh(A + tf.einsum('bi,bj->ij', write_weights, kts))

        # get read vector using using write weights and updated memory
        read_vectors = []
        for i in range(self.read_num):
            read_vectors.append(  tf.einsum('mi,ij->mj', read_weights[i], A))


        # # if in training mode, make sure we are writing to memory
        # if training:
        #     # apply new data to memory block
        #     new_mem = self.write_head(dense_out, training=training)
        #     # update memory block
        #     self.prev_mem = new_mem
        #     # update the memory state
        #     self.reverie.force_write(self.prev_mem)
            
        # # pass to read head
        # read = self.read_head(dense_out, training=training)
        # # pass retrieved memory to network output
        # network_out = self.network_out(read, training=training)
        # # and final output
        # ntm_output = self.model_out(network_out, training=training)

        # return output of network
        read_outputs = tf.concat(read_vectors, axis=-1)
        ntm_output = tf.concat([controller_output, read_outputs], axis=-1)

        # package iteration state
        self.prev_state = {"memory": A,
                         "read_weight": read_weights_sum,
                         "write_weight": write_weights,
                         "usage_weight": wu
                         }


        return ntm_output


    def least_used(self, wu):
        _, indices = tf.nn.top_k(wu, k=self.memrows)
        wlu = tf.cast(tf.slice(indices, [0, self.memory_size - self.nb_reads], [w_u.get_shape()[0], self.nb_reads]), dtype=tf.int32)
        wlu = tf.reduce_sum(tf.one_hot(wlu, self.memory_size), axis=1)
        return indices, wlu



    def train_step(self, data):
        """
        Overloads train step function in order to make use of fit method - useful for other keras functionality
        accepts a tuple of data consisting of the input state and memory  
        """

        # extract input/output data from input tuple
        x,y = data 

        with tf.GradientTape() as tape:
            # execute prediction in training mode
            y_pred = self(x, training=True)
            # compute loss based on compilation
            loss = self.compiled_loss(y, y_pred)

        # use gradient tape to get gradients of variables with respect to loss
        trainable_variables = self.trainable_variables
        grads = tape.gradient(loss, trainable_variables)

        # run step of gradient descent using chosen optimizer
        self.optimizer.apply_gradients(zip(grads, trainable_variables))

        # update metrics if any
        self.compiled_metrics.update_state(y, y_pred)

        # return metric values
        return {m.name: m.result() for m in self.metrics}



    def save_model(self, path, weight_name=r"\Neural_weights.h5", memory_name=r"\memory_block.npy"):
        """
        Saves the trained weights of the model to the specified path (must include name)
        """

        # save weights
        self.save_weights(path + weight_name)
        # save memory blocks
        with open(path + memory_name, 'wb') as f:
            np.save(f, self.reverie.memory.numpy())


    def load_model(self, path, batch_size=20):
        """
        Retrieves trained weights from saved model - assumdes identical network
        """

        # checks if model layers have been defined
        try:
            self.load_weights(path+r"\Neural_weights.h5")
        except ValueError:
            # force initialise model 
            self(np.random.rand(batch_size,self.input_length), training=True)
            # now try
            self.load_weights(path+r"\Neural_weights.h5")

        with open(path + r"\memory_block.npy", 'rb') as f:
            mem = np.load(f)

        self.prev_mem = self.reverie.memory = tf.identity(mem)

    def compute_output_shape(self, input_shape):
        assert isinstance(input_shape, list)
        shape_a, shape_b = input_shape
        return [(shape_a[0], self.output_dim), shape_b[:-1]]

    def get_config(self):
        base_config = super(NTM, self).get_config()
        base_config['output_dim'] = self.output_dim
        return base_config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    @property
    def output_size(self):
        return [self.output_dim, self.output_dim]


class RevInterface(Layer):
    """
    Base class for read/write head. Essentially a container for generating 
    the weight vector 
    """

    def __init__(self, memory_shape, **kwargs):
        # setup superclass inheritance 
        super(RevInterface, self).__init__(**kwargs)

        # compute and store memory size
        self.memcols = memory_shape[1]
        self.memrows = memory_shape[0]


# we use seperate read and write head classes since they may not always match in number
class RevReadHead(RevInterface):
    """
    Performs read operations on memory instances
    """ 
    def __init__(self, network_topology='ff', **kwargs):
        # initiailise controller interface
        super(RevReadHead, self).__init__(**kwargs)
 
        # compute and store required network depth
        self.layer_dim = self.memcols

        # network topology for controller head
        if network_topology is 'ff':
            self.head_out = Dense(self.layer_dim, name="Read_controller")
        else:
            raise NotImplementedError("Custom heads not yet implemented")

    # overwrite tensorflow call method with our own
    def read_head_address(self, controller_input, memory, training=False):
        """
        Applies read head to input data vector and produces memory read output
        """

        # compute key vector and force to be between [-1, 1]
        kt = tf.tanh(self.head_out(controller_input))

        # compute inner product bewteen key vector and memory rows
        inner_product = tf.einsum('bj,ij->bi', kt, memory)

        # compute norm of components
        k_norm = tf.math.sqrt(tf.reduce_sum(tf.math.square(k), 2))
        memory_norm = tf.math.sqrt(tf.reduce_sum(tf.math.square(A), 2))
        norm = k_norm * memory_norm

        # compute cosine similarity function
        K = inner_product / (norm + 1e-16)

        # compute softmax normalisation
        return kt, tf.nn.softmax(K, axis=-1)


    def compute_output_shape(self, input_shape):
        assert isinstance(input_shape, list)
        shape_a, shape_b = input_shape
        return [(shape_a[0], self.output_dim), shape_b[:-1]]

    def get_config(self):
        base_config = super(RevReadHead, self).get_config()
        base_config['output_dim'] = self.mem_output_dim
        return base_config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    @property
    def output_size(self):
        return [self.output_dim, self.output_dim]



class RevWriteHead(RevInterface):
    """
    Performs write operations on memory instances
    """

    def __init__(self, network_topology='ff', **kwargs):
        # initiailise controller interface
        super(RevWriteHead, self).__init__(**kwargs)
    

        # require a single parameter network (need alpha )
        self.layer_dim = 1 

        # network topology for controller head
        if network_topology is 'ff':
            self.head_out = Dense(self.layer_dim, name="Write_controller")
        else:
            raise NotImplementedError("Custom heads not yet implemented")

    # overwrite tensorflow call method with our own
    def write_head_address(self, controller_input, wr_prev, wlu_prev, training=False):
        """
        Applies read head to input data vector and produces memory write output
        """

        # apply head network to input data
        head_output_data = tf.tanh(self.head_out(input_data), axis=-1)

        # extract alpha and gamma
        alpha = tf.sigmoid(head_output_data)

        # return convex combination of previous read weights and write weights
        write_weights = alpha * wr_prev + (1-alpha)*wlu_prev

        return write_weights


    def compute_output_shape(self, input_shape):
        assert isinstance(input_shape, list)
        shape_a, shape_b = input_shape
        return [(shape_a[0], self.output_dim), shape_b[:-1]]

    def get_config(self):
        base_config = super(RevWriteHead, self).get_config()
        base_config['output_dim'] = self.mem_input_dim
        return base_config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    @property
    def output_size(self):
        return [self.output_dim, self.output_dim]


def model_retrieve(model_path, input_shape, memshape, output_dim, logits, batch_size):
    """
    Loads a previously trained model 

    TODO: Greatly expand this functionality to archival system
    """

    # contruct skeleton of model 
    model = NTM(input_shape=input_shape, memshape=memshape, output_dim=output_dim, logits=logits)

    # compile model with same charateristics
    opt = tf.keras.optimizers.Adam(lr=0.0)

    # define loss function
    loss = tf.keras.losses.MeanSquaredError()
    # compile
    model.compile(optimizer=opt, loss=loss)
    # and finally load
    model.load_model(model_path, batch_size)

    return model



def memget(filename):
    """
    Retrieves a memory block from hard storage  
    """

    print("calling function that is not implemented yet: utility/memget")
    
    return None


def cosine_distance(u, mat):
    """
    Computes the cosine distance between a batched set of vectors u and a matrix mat
    """


    # compute pairwise innerproduct
    inner_prod = tf.einsum("ij,kj->ik", u, mat)

    # compute norms
    u_norm = tf.norm(u, ord=2, axis=1)
    mat_norm = tf.norm(mat, ord=2, axis=1)

    # compute product of norms
    norm_prod = tf.einsum("i,j->ij",u_norm, mat_norm)

    # compute norm division 
    cos_dist = tf.math.divide(inner_prod, norm_prod)

    return cos_dist