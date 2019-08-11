import tensorflow as tf
a=tf.constant(2,name='a')
b=tf.constant(3)
x=tf.add(a,b,name='add')

from datetime import datetime

now = datetime.utcnow().strftime("%Y%m%d%H%M%S")
root_logdir = "tf_logs"
logdir = "{}/run-{}/".format(root_logdir, now)

with tf.Session() as sess:
    writer = tf.summary.FileWriter(logdir, sess.graph )
    print(sess.run(x))
writer.close()
