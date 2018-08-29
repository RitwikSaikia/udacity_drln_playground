import tensorflow as tf

g1 = tf.Graph()
with g1.as_default() as g:
    with g.name_scope( "g1" ) as g1_scope:
        matrix1 = tf.constant([[3., 3.]])
        matrix2 = tf.constant([[2.],[2.]])
        product = tf.matmul( matrix1, matrix2, name = "product")

tf.reset_default_graph()

g2 = tf.Graph()
with g2.as_default() as g:
    with g.name_scope( "g2" ) as g2_scope:
        matrix1 = tf.constant([[4., 4.]])
        matrix2 = tf.constant([[5.],[5.]])
        product = tf.matmul( matrix1, matrix2, name = "product" )

tf.reset_default_graph()

use_g1 = False

if ( use_g1 ):
    g = g1
    scope = g1_scope
else:
    g = g2
    scope = g2_scope

with tf.Session( graph = g ) as sess:
    tf.initialize_all_variables()
    result = sess.run( sess.graph.get_tensor_by_name( scope + "product:0" ) )
    print( result )