import tensorflow as tf
import ops
import utils
from discriminator import Discriminator
from generator import Generator

REAL_LABEL = 0.9



class CycleGAN:
  def __init__(self,
               batch_size=1,
               image_size=256,
               use_lsgan=True,
               norm='instance',
               lambda1=10,
               lambda2=10,
               learning_rate=2e-4,
               beta1=0.5,
               ngf=32,
               use_gpu=0,
               discrim_inp_ch=28,
              ):
    """
    Args:
      X_train_file: string, X tfrecords file for training
      Y_train_file: string Y tfrecords file for training
      batch_size: integer, batch size
      image_size: integer, image size
      lambda1: integer, weight for forward cycle loss (X->Y->X)
      lambda2: integer, weight for backward cycle loss (Y->X->Y)
      use_lsgan: boolean
      norm: 'instance' or 'batch'
      learning_rate: float, initial learning rate for Adam
      beta1: float, momentum term of Adam
      ngf: number of gen filters in first conv layer
    """
    if type(use_gpu) is int:
        use_gpu = [use_gpu]
    self.use_gpu = use_gpu
    
    self.lambda1 = lambda1
    self.lambda2 = lambda2
    self.use_lsgan = use_lsgan
    use_sigmoid = not use_lsgan
    self.batch_size = batch_size
    self.image_size = image_size
    self.learning_rate = learning_rate
    self.beta1 = beta1
    
    self.opt = self.make_optimizer()
    
    self.is_training = tf.placeholder_with_default(True, shape=[], name='is_training')

    self.G = Generator('G', self.is_training, output_channel=3,
                       ngf=ngf, norm=norm, image_size=image_size)
    self.D_Y = Discriminator('D_Y',
        self.is_training, norm=norm, use_sigmoid=use_sigmoid)
    self.F = Generator('F', self.is_training, output_channel=3,
                       norm=norm, image_size=image_size)
    
    self.D_X = Discriminator('D_X',
        self.is_training, norm=norm, use_sigmoid=use_sigmoid)

    self.fake_x = tf.placeholder(tf.float32,
        shape=[batch_size, image_size, image_size, discrim_inp_ch])
    self.fake_y = tf.placeholder(tf.float32,
        shape=[batch_size, image_size, image_size, discrim_inp_ch])

  def model(self, inp_x, inp_y, time_x, time_y):
    # want to generate [inp_x, time_y] -> inp_y
    G_opt_grads = []
    D_Y_opt_grads = []
    F_opt_grads = []
    D_X_opt_grads = []
    
    fake_x_list = tf.split(self.fake_x, len(self.use_gpu))
    fake_y_list = tf.split(self.fake_y, len(self.use_gpu))
    inp_x_list = tf.split(inp_x, len(self.use_gpu))
    inp_y_list = tf.split(inp_y, len(self.use_gpu))
    time_x_list = tf.split(time_x, len(self.use_gpu))
    time_y_list = tf.split(time_y, len(self.use_gpu))
    
    tot_G_loss = 0.
    tot_F_loss = 0.
    tot_D_Y_loss = 0.
    tot_D_X_loss = 0.
    
    fake_xs = []
    fake_ys = []
    
    with tf.variable_scope(tf.get_variable_scope()):
      for i, gpu_id in enumerate(self.use_gpu):
        print('Initializing graph on gpu %i' % gpu_id)
        with tf.device('/gpu:%d' % gpu_id):
          pooled_fake_x = fake_x_list[i] # already has the date appended
          pooled_fake_y = fake_y_list[i] # already has the date appended
          inp_x = inp_x_list[i]
          inp_y = inp_y_list[i]
          time_x = time_x_list[i]
          time_y = time_y_list[i]
          
          inp_shapes = tf.shape(inp_x)
        
          tiled_time_x = tf.expand_dims(tf.expand_dims(time_x, axis=1), axis=1) * tf.ones((inp_shapes[0],
                                                                                           inp_shapes[1],
                                                                                           inp_shapes[2], 25))
          tiled_time_y = tf.expand_dims(tf.expand_dims(time_y, axis=1), axis=1) * tf.ones((inp_shapes[0],
                                                                                           inp_shapes[1],
                                                                                           inp_shapes[2], 25))
          # Real Examples
          inp_x_time_x = tf.concat([inp_x, tiled_time_x], axis=-1)
          inp_y_time_y = tf.concat([inp_y, tiled_time_y], axis=-1)

          # X -> Y
          fake_y = self.G(inp_x, tiled_time_y)
          fake_y_time_y = tf.concat([fake_y, tiled_time_y], axis=-1)
          fake_ys.append(fake_y_time_y)
        
          G_gan_loss = self.generator_loss(self.D_Y, fake_y_time_y, use_lsgan=self.use_lsgan)
          D_Y_loss = self.discriminator_loss(self.D_Y, inp_y_time_y,
                                             pooled_fake_y, use_lsgan=self.use_lsgan)

          # Y -> X
          fake_x = self.F(inp_y, tiled_time_x)
          fake_x_time_x = tf.concat([fake_x, tiled_time_x], axis=-1)
          fake_xs.append(fake_x_time_x)
          
          F_gan_loss = self.generator_loss(self.D_X, fake_x_time_x, use_lsgan=self.use_lsgan)
          D_X_loss = self.discriminator_loss(self.D_X, inp_x_time_x,
                                             pooled_fake_x, use_lsgan=self.use_lsgan)
            
          cycle_loss = self.cycle_consistency_loss(self.G, self.F,
                                                   inp_x, inp_y,
                                                   fake_x, fake_y,
                                                   tiled_time_x, tiled_time_y)
            
          F_loss = F_gan_loss + cycle_loss
          G_loss =  G_gan_loss + cycle_loss
        
          tot_F_loss += F_loss
          tot_D_X_loss += D_X_loss
          tot_G_loss += G_loss
          tot_D_Y_loss += D_Y_loss
          
          tf.get_variable_scope().reuse_variables()
          
          G_opt_grads.append(self.opt.compute_gradients(G_loss, var_list=self.G.variables))
          F_opt_grads.append(self.opt.compute_gradients(F_loss, var_list=self.F.variables))
          D_Y_opt_grads.append(self.opt.compute_gradients(D_Y_loss, var_list=self.D_Y.variables))
          D_X_opt_grads.append(self.opt.compute_gradients(D_X_loss, var_list=self.D_X.variables))
    G_grads = ops.average_gradients(G_opt_grads)
    F_grads = ops.average_gradients(F_opt_grads)
    D_Y_grads = ops.average_gradients(D_Y_opt_grads)
    D_X_grads = ops.average_gradients(D_X_opt_grads)
    
    G_ts = self.opt.apply_gradients(G_grads, global_step=self.global_step)
    F_ts = self.opt.apply_gradients(F_grads, global_step=self.global_step)
    D_Y_ts = self.opt.apply_gradients(D_Y_grads, global_step=self.global_step)
    D_X_ts = self.opt.apply_gradients(D_X_grads, global_step=self.global_step)
    
    with tf.control_dependencies([G_ts, D_Y_ts, F_ts, D_X_ts]):
      ts = tf.no_op(name='optimizers')
    
    with tf.control_dependencies([G_ts, F_ts]):
      gen_ts = tf.no_op(name='gen_optimizers')
    
    fake_x = tf.concat(fake_xs, axis=0)
    fake_y = tf.concat(fake_ys, axis=0)
    
    num_gpu = max(1., len(self.use_gpu) + 0.)

    return (tot_G_loss/num_gpu, tot_D_Y_loss/num_gpu,
            tot_F_loss/num_gpu, tot_D_X_loss/num_gpu,
            fake_y, fake_x, ts, gen_ts)

  def make_optimizer(self, name='Adam'):
    """ Adam optimizer with learning rate 0.0002 for the first 100k steps (~100 epochs)
        and a linearly decaying rate that goes to zero over the next 100k steps
    """
    self.global_step = tf.Variable(0, trainable=False)
    starter_learning_rate = self.learning_rate
    end_learning_rate = 0.0
    start_decay_step = 100000
    decay_steps = 100000
    beta1 = self.beta1
    learning_rate = (
        tf.where(
                tf.greater_equal(self.global_step, start_decay_step),
                tf.train.polynomial_decay(starter_learning_rate, self.global_step-start_decay_step,
                                          decay_steps, end_learning_rate,
                                          power=1.0),
                starter_learning_rate
        )

    )

    return tf.train.AdamOptimizer(learning_rate, beta1=beta1, name=name)

  def optimize(self, G_loss, D_Y_loss, F_loss, D_X_loss):

    G_optimizer = make_optimizer(G_loss, self.G.variables, name='Adam_G')
    D_Y_optimizer = make_optimizer(D_Y_loss, self.D_Y.variables, name='Adam_D_Y')
    F_optimizer =  make_optimizer(F_loss, self.F.variables, name='Adam_F')
    D_X_optimizer = make_optimizer(D_X_loss, self.D_X.variables, name='Adam_D_X')

    with tf.control_dependencies([G_optimizer, D_Y_optimizer, F_optimizer, D_X_optimizer]):
      return tf.no_op(name='optimizers')

  def discriminator_loss(self, D, y, fake_y, use_lsgan=True):
    """ Note: default: D(y).shape == (batch_size,5,5,1),
                       fake_buffer_size=50, batch_size=1
    Args:
      G: generator object
      D: discriminator object
      y: 4D tensor (batch_size, image_size, image_size, 3)
    Returns:
      loss: scalar
    """
    if use_lsgan:
      # use mean squared error
      error_real = tf.reduce_mean(tf.squared_difference(D(y), REAL_LABEL))
      error_fake = tf.reduce_mean(tf.square(D(fake_y)))
    else:
      # use cross entropy
      error_real = -tf.reduce_mean(ops.safe_log(D(y)))
      error_fake = -tf.reduce_mean(ops.safe_log(1-D(fake_y)))
    loss = (error_real + error_fake) / 2
    return loss

  def generator_loss(self, D, fake_y, use_lsgan=True):
    """  fool discriminator into believing that G(x) is real
    """
    if use_lsgan:
      # use mean squared error
      loss = tf.reduce_mean(tf.squared_difference(D(fake_y), REAL_LABEL))
    else:
      # heuristic, non-saturating loss
      loss = -tf.reduce_mean(ops.safe_log(D(fake_y))) / 2
    return loss

  def cycle_consistency_loss(self, G, F, x, y,
                             fake_x, fake_y,
                             time_x, time_y):
    """ cycle consistency loss (L1 norm)
    """
    
    fake_fake_x = F(fake_y, time_x)
    forward_loss = tf.reduce_mean(tf.abs(fake_fake_x-x))
    fake_fake_y = G(fake_x, time_y)
    backward_loss = tf.reduce_mean(tf.abs(fake_fake_y-y))
    
    loss = self.lambda1*forward_loss + self.lambda2*backward_loss
    return loss