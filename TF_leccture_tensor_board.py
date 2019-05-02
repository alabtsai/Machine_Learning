import tensorflow as tf
a=tf.constant(2)
b=tf.constant(3)
x=tf.add(a,b)


with tf.Session() as sess:
    writer = tf.summary.FileWriter('./graphsTSAI', sess.graph )
    print(sess.run(x))
writer.close()
