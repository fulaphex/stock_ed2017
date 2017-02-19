import tensorflow as tf
import numpy as np

SEED = 314
BATCH_SIZE = 200
INPUT_DIM = 8
NUM_HIDDEN = 0
NUM_MIX = 3
HIDDEN_DIMS = [INPUT_DIM] + [1000 for i in xrange(NUM_HIDDEN)]
LEARNING_RATE = 1e-7
NUM_EPOCHS = 1e5

# def take(X,A,k):
#     resX = []
#     n,d = X.shape
#     for i in range(k,len(X)):
#         resX.append(array([X[j] for j in range(i-k,i)]).reshape(d*k))
#     return array(resX), A[k:]

class Data_feed:
    def __init__(self, *args):
        self.variables = args
        self.loc = 0
    def get_batch(self, batch_size):
        res = [var[self.loc:self.loc+batch_size] for var in self.variables]
        self.loc += batch_size
        if self.loc >= self.variables[0].shape[0]:
            self.loc = 0
        return res
    def reset(self):
        self.location = 0
        return

class Net:
    def __init__(self):
        self.X = tf.placeholder(tf.float32, shape=(None, INPUT_DIM))
        self.A = tf.placeholder(tf.float32, shape=(None, NUM_MIX))

        out = self.X
        for i in xrange(NUM_HIDDEN):
            with tf.name_scope("hidden"+str(i)):
                in_dim, out_dim = HIDDEN_DIMS[i], HIDDEN_DIMS[i+1]
                weight = tf.Variable(tf.truncated_normal([in_dim, out_dim], stddev=1.0 / np.sqrt(float(in_dim))), name = 'weight')
                bias   = tf.Variable(tf.zeros([out_dim]), name = 'bias')
                out = tf.nn.sigmoid(tf.matmul(out, weight) + bias)

        with tf.name_scope("softmax"):
            in_dim, out_dim = HIDDEN_DIMS[-1], NUM_MIX
            weight = tf.Variable(tf.truncated_normal([in_dim, out_dim], stddev=1.0 / np.sqrt(float(in_dim))), name = 'weight_softmax')
            bias   = tf.Variable(tf.zeros([out_dim]), name = 'bias_softmax')
            out = tf.matmul(out, weight) + bias
            self.smax = tf.nn.softmax(out)

        self.loss = tf.nn.softmax_cross_entropy_with_logits(logits = out, labels = self.A)
        # self.loss = tf.log(tf.reduce_sum(tf.multiply(A, smax), axis = 1))

        optimizer = tf.train.GradientDescentOptimizer(LEARNING_RATE)
        self.train_op = optimizer.minimize(self.loss)
        self.init_op = tf.global_variables_initializer()

if __name__ == "__main__":
    np.random.seed(SEED)
    A = np.load('A1.npy')
    X = np.load('X1.npy')
    ind = np.logical_not(np.argmax(A, axis = 1) == 1)
    Xin = X[ind]
    Ain = A[ind]

    INPUT_SIZE = Xin.shape[0]
    TRAIN_SIZE = int(0.7 * INPUT_SIZE)
    VALIDATION_SIZE = int(0.15 * INPUT_SIZE)
    TEST_SIZE = INPUT_SIZE - TRAIN_SIZE - VALIDATION_SIZE

    perm = np.random.permutation(INPUT_SIZE)
    X_train = Xin[perm[:TRAIN_SIZE]]
    A_train = Ain[perm[:TRAIN_SIZE]]
    X_valid = Xin[perm[TRAIN_SIZE:TRAIN_SIZE+VALIDATION_SIZE]]
    A_valid = Ain[perm[TRAIN_SIZE:TRAIN_SIZE+VALIDATION_SIZE]]
    X_test  = Xin[perm[TRAIN_SIZE+VALIDATION_SIZE:]]
    A_test  = Ain[perm[TRAIN_SIZE+VALIDATION_SIZE:]]

    n = Net()
    train_d = Data_feed(X_train, A_train)
    valid_d = Data_feed(X_valid, A_valid)
    test_d  = Data_feed(X_test , A_test)

    sess = tf.Session()
    sess.run(n.init_op)
    losses = []
    pvalid_loss = np.inf
    pacc = 0
    print 'hidden layers:', NUM_HIDDEN, HIDDEN_DIMS[1:]
    print 'epochs:', NUM_EPOCHS
    print 'learning rate:', LEARNING_RATE
    print
    for e in np.arange(NUM_EPOCHS)+1:
    # e = 0
    # while True:
    #     e += 1
        x, a = train_d.get_batch(TRAIN_SIZE)
        _, l = sess.run([n.train_op, n.loss], feed_dict = {n.X:x, n.A:a})
        if not e % 25 and e > 1e3:
        # if not e % 25:
            valid_d.reset()
            x, a = valid_d.get_batch(VALIDATION_SIZE)
            y, valid_loss = sess.run([n.smax, n.loss], feed_dict = {n.X:x, n.A:a})
            corr = np.argmax(a, axis = 1)
            pred = np.argmax(y, axis = 1)

            acc = np.mean(pred == corr)
            if acc < pacc - 1e-2:
                print 'epoch', e
                print 'acc', np.mean(pred == corr)
                print 'loss', np.mean(valid_loss)
                print 'ended after', e, 'epochs'
                break
            pvalid_loss = valid_loss
            pacc = acc

        if not e % 1e4:
            print 'epoch', e
            print 'acc', np.mean(pred == corr)
            print


    x, a = X_test, A_test
    y = sess.run(n.smax, feed_dict = {n.X:x})
    corr = np.argmax(a, axis = 1)
    pred = np.argmax(y, axis = 1)
    acc = np.mean(np.argmax(y, axis = 1) == np.argmax(a, axis = 1))
    print 'test acc', acc
    if np.all(pred == 0) or np.all(pred == 2):
        print 'constant predictions on test set'
