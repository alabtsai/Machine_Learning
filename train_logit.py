import numpy as np
import tensorflow as tf
import utils

batch_size = 128

mnist_folder = 'data/mnist'
utils.download_mnist(mnist_folder)
train, val, test = utils.read_mnist(mnist_folder, flatten=True)


n_test,_=test[1].shape


train_data = tf.data.Dataset.from_tensor_slices(train)
test_data = tf.data.Dataset.from_tensor_slices(test)

train_data = train_data.batch(batch_size)
test_data = test_data.batch(batch_size)

iterator = tf.data.Iterator.from_structure(train_data.output_types,
                                           train_data.output_shapes)


train_init = iterator.make_initializer(train_data)
test_init = iterator.make_initializer(test_data)

img, label = iterator.get_next()
# 
w = tf.get_variable(name='weights', shape=(784, 10), initializer=tf.random_normal_initializer(0, 0.01))
b = tf.get_variable(name='bias', shape=(1, 10), initializer=tf.random_normal_initializer(0, 0.01) )
logits = tf.matmul(img, w)+b
# define loss function
# use cross entropy of softmax of logits as the loss function
entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=label, name='entropy')
loss = tf.reduce_mean(entropy, name='loss') # computes the mean over all the examples in the batch
# define training op
# using gradient descent with learning rate of 0.01 to minimize loss
learning_rate = 0.01
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)


# calculate accuracy with test set
preds = tf.nn.softmax(logits)
correct_preds = tf.equal(tf.argmax(preds, 1), tf.argmax(label, 1))
accuracy = tf.reduce_sum(tf.cast(correct_preds, tf.float32))



#saver=tf.train.Saver({"weights":w, "bias":b })
saver=tf.train.Saver()
n_epochs = 30


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    
    # train the model n_epochs times
    for i in range(n_epochs):
        sess.run(train_init)
        total_loss=0
        n_batches=0
        try:
            while True:
                _,l,imgg=sess.run([optimizer, loss, img])
                #num_sample,_=imgg.shape
                total_loss  += l
                n_batches += 1
                #print(sess.run(tf.is_nan(l2)))
                #print('loss=',l)
                #tf.cond(tf.is_nan(l2),lambda:print(sess.run(where)), lambda:print('ok'))
                #list1.append(sess.run(preds))
                #print('n_batch=',n_batches)
                #print('img shape',imgg.shape)
        except tf.errors.OutOfRangeError:
            pass
        print('Average loss epoch {0}: {1}'.format(i,total_loss/n_batches))
    #get the  weight and bias
    weight_,bias_=sess.run([w,b])
    save_path=saver.save(sess,"/tmp/my_model_final.ckpt")
    
    # test the model
    sess.run(test_init)# drawing samples from test_data
    total_correct_preds =  0
    try:
        while True:
            accuracy_batch = sess.run(accuracy)
            total_correct_preds += accuracy_batch
    except tf.errors.OutOfRangeError:
        pass
        
    print('Accuracy {0}'.format(total_correct_preds/n_test))



with tf.Session() as sess:
    saver.restore(sess,"/tmp/my_model_final.ckpt")
    w_out, b_out = sess.run([w, b])
    print('w = ', w_out[:2])
    print('b = ', b_out[:2])

print(weight_[:2])
print(bias_)

