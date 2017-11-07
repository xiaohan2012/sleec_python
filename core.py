import tensorflow as tf


class RegressionModel():
    def __init__(self, d, l, lambda1, lambda2):
        self.X = tf.placeholder(dtype=tf.float64, shape=(None, d), name='X')
        self.V = tf.Variable(tf.random_uniform([l, d], -1.0, 1.0, dtype=tf.float64),
                             name='V')
        self.W = tf.placeholder(dtype=tf.float64, shape=(None, l), name='W')
        
        self.VX = tf.matmul(self.V, tf.transpose(self.X), name='VX')
        
        self.l2_loss = tf.nn.l2_loss(self.V, name='l2_loss')
        # self.l2_loss = tf.reduce_mean(tf.pow(self.V, 2), name='l2_loss')
        self.l1_loss = tf.reduce_mean(tf.abs(self.VX), name='l1_loss')
        
        self.error = tf.reduce_mean(tf.pow(self.W - tf.transpose(self.VX), 2))
        self.loss = self.error + lambda2 * self.l2_loss + lambda1 * self.l1_loss


def learn_V(X_val, Z_val, lambda1=1, lambda2=1, learning_rate=0.1,
            iter_max=10, print_log=True):
    n, d = X_val.shape
    _, l = Z_val.shape
    model = RegressionModel(d, l, lambda1, lambda2)

    feed_dict = {
        model.X: X_val,
        model.W: Z_val
    }
    train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(model.loss)
    
    with tf.Session() as sess:
        # global_step = tf.Variable(0, name="global_step", trainable=False)
        sess.run(tf.global_variables_initializer())
        for i in range(iter_max):
            # current_step = tf.train.global_step(sess, global_step)
            # loss_val = sess.run([loss], feed_dict=feed_dict)
            _, loss_val, error_val, l1_val, l2_val = sess.run(
                [train_op, model.loss, model.error, model.l1_loss, model.l2_loss],
                feed_dict=feed_dict)

            if print_log:
                print("at step {}, loss={}, error={}, l1={}, l2={}".format(
                    i, loss_val, error_val, l1_val, l2_val))
                
        return sess.run([model.V], feed_dict=feed_dict)[0]
