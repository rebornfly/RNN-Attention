import tensorflow as tf

logits =  tf.constant([[ 0.19874704 , 0.20004079,  0.20086856,  0.20125246 , 0.19909115 ]])
labels = tf.constant([1])
real_distance = tf.to_int32(tf.abs(tf.subtract(tf.argmax(logits, 1), tf.argmax(labels,0))))
distance_index = tf.constant([2], dtype=tf.int32)
distance = tf.pow(distance_index, real_distance)
#distance_per_label = tf.transpose( tf.matmul(labels, tf.transpose(distance))  )
print(labels, logits)
cross = tf.nn.sparse_softmax_cross_entropy_with_logits(logits= logits, labels = labels)
print(cross)
xent = tf.multiply(tf.to_float(distance) , tf.nn.sparse_softmax_cross_entropy_with_logits(logits= logits, labels = labels))

loss_1 = tf.reduce_mean(xent)
loss_2 = tf.reduce_mean(
    tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=labels, logits=logits))
sess = tf.Session()
print(sess.run(distance))
print(sess.run(loss_1))
print(sess.run(loss_2))

