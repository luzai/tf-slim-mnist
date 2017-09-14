import tensorflow as tf
import tensorflow.contrib.lookup

sess = tf.InteractiveSession()
sess.as_default()
keys = tf.convert_to_tensor([1, 2])
values = [3, 4]
input_tensor = tf.convert_to_tensor(1)
table = tf.contrib.lookup.HashTable(
    tf.contrib.lookup.KeyValueTensorInitializer(keys, values), -1)
out = table.lookup(input_tensor)
table.init.run()
print(out.eval())