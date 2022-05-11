import datetime
import numpy as np
import tensorflow as tf 
import matplotlib.pyplot as plt 
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Input, Layer


# TODO: Different similarity measure
# TODO: Implement variable shift vectors

# set default precision
tf.keras.backend.set_floatx('float64')


class NTM(Model):
    
    def __init__(self, length=10, memshape=None, output_dim=1, logits=100, training=True,**kwargs):
        # initialise the superclass
        super(NTM, self).__init__(name="Exuberant_Witness",**kwargs)

        # store desired memshape
        self.memshape = memshape
        # store desired output dimension for read heads
        self.output_dim = output_dim
        # store number of logits for internal network
        self.logits = logits
        # store input length
        self.input_length = length
        # store training flag
        self.training = training
        # instantiate memory
        self.reverie = Reverie(memshape=self.memshape, ident=None)
        # save initial state of previous memory
        self.prev_mem = self.reverie.memory
        # construct base input network
        self.network_in = Dense(self.logits, activation="relu", name="Controller_dense_in", input_shape=[length])
        # construct base output network
        self.network_out = Dense(self.logits, activation="relu", name="Controller_dense_out", input_shape=[memshape[-1]]) 
        # instantiate the read head
        self.read_head = RevReadHead(self.reverie, input_dim=self.logits, network_topology=None)
        # instantiate write head
        self.write_head = RevWriteHead(self.reverie, input_dim=self.logits)
        # final layer to map to appropriate output dimensions
        self.model_out = Dense(self.output_dim, activation="relu" ,use_bias=True, name="NTM_out")


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


    def call(self, x, training=False):
        """ 
        Calls the neural turing machine on an input and produces the networks prediction. 
        """
        
        # apply network input 
        dense_out = self.network_in(x, training=training)
        

        # if in training mode, make sure we are writing to 
        if training:
            # apply new data to memory block
            new_mem = self.write_head(dense_out, training=training)
            # update memory block
            self.prev_mem = new_mem
            # update the memory state
            self.reverie.force_write(self.prev_mem)
            
        # pass to read head
        read = self.read_head(dense_out, training=training)
        # pass retrieved memory to network output
        network_out = self.network_out(read, training=training)
        # and final output
        ntm_output = self.model_out(network_out, training=training)

        
        


        return ntm_output

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

    def __init__(self, reverie):
        # setup superclass inheritance 
        super(RevInterface, self).__init__()
            
        # store pointer to the memory class
        self.reverie = reverie

        # compute and store memory size
        self.mem_input_dim = self.reverie.memshape[1]
        self.mem_output_dim = self.reverie.memshape[0]

        # initialise weight history
        self.prev_weight = tf.convert_to_tensor(1e-2*np.ones((1, self.mem_output_dim), dtype=np.float64))


    def head_control(self, key, beta, gate_interp, shift, gamma, training=False):
        """
        Maps the head control signals to correct ranges for read/write access to memory.
        These signals comes from a neural network of some kind and can take on any real value
        """


        # normalise beta
        beta = tf.math.softplus(beta)

        # remap gate interpolation
        gate_interp = tf.nn.sigmoid(gate_interp)

        # normalise shift vector
        shift = tf.nn.softmax(shift)

        # force gamma power to be strictly increasing
        gamma = tf.nn.softplus(gamma) + 1

        # compute weight vector
        head_weight = self.reverie.weight_compute(key, beta, gate_interp, shift, gamma, self.prev_weight)


        return head_weight



# we use seperate read and write head classes since they may not always match in number
class RevReadHead(RevInterface):
    """
    Performs read operations on memory instances
    """ 
    def __init__(self, reverie, input_dim, network_topology=None):
        # initiailise controller interface
        super(RevReadHead, self).__init__(reverie)
 
        # compute and store required network depth
        self.layer_width = self.mem_input_dim + 6

        # network topology for controller head
        if network_topology is None:
            self.head_out = Dense(self.layer_width, input_shape=(input_dim,), name="Read_controller")
        else:
            raise NotImplementedError("Custom heads not yet implemented")

    # overwrite tensorflow call method with our own
    def call(self, input_data, training=False):
        """
        Applies read head to input data vector and produces memory read output
        """


        # apply head network to input data
        head_output_data = self.head_out(input_data)

        # split head output data and control signals
        output_shape = [self.mem_input_dim, 1,1,1,3]
        key, beta, gate_interp, gamma, shift = tf.split(head_output_data, output_shape, axis=-1)

        # use to retrieve memory 
        w = self.head_control(key, beta, gate_interp, shift, gamma, training=training)
        
        # store internal rep of memory (only update if training has begun, lest we overwrite)
        if training:
            self.prev_weight = w


        # use computed weight to perform read operation
        mem_read = self.reverie.read_memory(w)

        return mem_read

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

    def __init__(self, reverie, input_dim, network_topology=None):
        # initiailise controller interface
        super(RevWriteHead, self).__init__(reverie)
 
        # compute and store required network depth (need add and erase channels here as well)
        self.layer_width = 3*self.mem_input_dim + 6

        # network topology for controller head
        if network_topology is None:
            self.head_out = Dense(self.layer_width, input_shape=(input_dim,))
        else:
            raise NotImplementedError("Custom heads not yet implemented")

    # overwrite tensorflow call method with our own
    def call(self, input_data, training=False):
        """
        Applies read head to input data vector and produces memory read output
        """

        # apply head network to input data
        head_output_data = self.head_out(input_data)
    
        # split head output data and control signals
        output_shape = [self.mem_input_dim, 1,1,1,3, self.mem_input_dim, self.mem_input_dim]
        key, beta, gate_interp, gamma, shift, erase, add = tf.split(head_output_data, output_shape, axis=-1)

        # remap erase vector
        erase = tf.math.sigmoid(erase)

        # use to retrieve memory 
        w = self.head_control(key, beta, gate_interp, shift, gamma)
        # update saved weight
        self.prev_weight = w
        
        # write to memory location
        self.reverie.write_memory(w, erase, add)

        return self.reverie.memory

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



class Reverie(object):
    """
    Memory object for neural turing machine. Initialises/retrieves a 
    model memory given input parameters. 
    """

    def __init__(self, memshape=None, filename=None, ident=None):
        # parse input parameters and setup our system memory
        super(Reverie, self).__init__()

        # check if we are  reading memory from file
        if filename is None:
            if memshape is None:
                # set default memory shape
                print("No memory file specified and memshape is {}: using default memory size of (40,20)".format(memshape))
                self.memshape = [40,20]
            else:
                self.memshape = memshape

            # initialise memory using small but not machine precision values
            self.memory = tf.convert_to_tensor(np.random.rand(*self.memshape)*(1e-1), dtype=tf.float64)

        else:
            # save filename into internal memory 
            self.filename = filename
            # use retreival service to get saved memory 
            self.memory, self.memshape = memget(self.filename)

        # use current milliseconds as system id
        if ident is None:
            self.ident = datetime.datetime.now().microsecond


    def read_memory(self, weight):
        """
        Applies read operation on memory block according
        """

        # compute weighted sum of row vectors that form memory
        rt = tf.linalg.matvec(tf.transpose(self.memory), weight)

        return rt


    def write_memory(self, weight, erase, add):
        """
        Applies write operation on memory block according to 
        """
        #TODO: Unclear how to handle batch dimension in the memory write

        # compute erasure matrix
        erase_mat = 1 - tf.einsum("ij,ik->ijk", weight, erase)
        # compute memory elementwise multiplication
        erase_mem = tf.einsum('kl,ikl->ikl', self.memory, erase_mat)
        # compute add operation
        add_mem = tf.einsum('ij,ik->ijk', weight, add)

        # apply update to memory state and collapse batch write
        self.memory = tf.reduce_sum(erase_mem + add_mem, axis=0)

        return self.memory


    def weight_compute(self, key, beta, gate_interp, shift, gamma, prev_weight):
        """
        Computes content/location based memory weight vector. 
        """ 

        # first compute cosine similarity of key with memory
        content_weight = self.weight_content_compute(key, beta)


        # now compute the interpolation gate value
        gated_weight = gate_interp*content_weight + (1-gate_interp)*prev_weight

        # compute gate convolution
        shifted_weight = self.weight_shift(gated_weight, shift)

        # apply sharpening
        numerator = tf.pow(shifted_weight, gamma)
        w = numerator/tf.math.reduce_sum(numerator)

        return w
        

    def weight_content_compute(self, key, beta, metric="cosine"):
        """
        Computes overlap of key with each memory location
        """


        if metric is "cosine":
            #Kuv = 1 - tf.keras.losses.cosine_similarity(key, self.memory, axis=1)
            # compute cosine distance over memory matrix and batches
            Kuv = cosine_distance(key, self.memory)
            print(Kuv)
            # apply softmax to output in order to normalise and determine content focus
            wct = tf.nn.softmax(beta*Kuv, axis=1, name="content_weight_normalisation_{}".format(self.ident))
            print(wct)
            print()
        else: 
            raise ValueError("Unrecognised similarity metric requested: {}".format(metric))

        return wct


    def weight_shift(self, weight, shift):
        """
        Computes continuous shift operation on input weight vector
        """

        # pad shift matrix to length of weight matrix
        batch_size = np.shape(weight)[0]

        shift = tf.pad(shift, paddings=tf.constant([[0,0],[0,np.shape(weight)[-1]-np.shape(shift)[-1]]]))

        # construct circulant matrix
        shift_list = tf.TensorArray(tf.float64, size=0, dynamic_size=True)
        # write to first index of array
        for i in range(0, np.shape(weight)[-1]):
            shift_list = shift_list.write(shift_list.size(),shift)

        # stack vectors
        shift_mat = tf.transpose(shift_list.stack(), perm=[1,0,2])

        # apply shift to weight and return 
        weight_shift = tf.einsum("ijk,ik->ij", shift_mat, weight)

        return weight_shift

    def force_write(self, memory):
        """
        Assigns memory block
        """

        # check memory size is preserved
        #assert np.all(np.shape(memory) == np.shape(self.memory)), "Assigned memory does not match new one"

        self.memory = tf.clip_by_norm(tf.identity(memory), np.shape(memory)[0], axes=1)


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