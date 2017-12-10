"""Réseau de neurones convolutifs pour le dataset MNIST, construit avec tf.layers."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

tf.logging.set_verbosity(tf.logging.INFO)

def cnn_model_fn(features, labels, mode):
  """Construire l'architecture du réseau (ou model)"""
  # Couche d'input
  # features contient un paramètre appelé "x" qui représente nos features (i.e.: les images)
  # Transformer x en un tensor 4-D: [batch_size, width, height, channels]
  # Les images du dataset MNIST sont 28x28 pixels, noir et blanc, donc une seule valeur dans chaque pixel, 
  # d'où [-1, 28, 28, 1]
  # La première dimension (ici "-1") est pour indiquer le nombre de "features" dans le tensor (batch size en anglais), 
  # nous passons "-1" pour signifier que le nombre est dynamique
  input_layer = tf.reshape(features["x"], [-1, 28, 28, 1])

  # Couche de convolution #1
  # Effectue 32 convolutions (donc 32 filtres, ou "kernels"), de dimension 5x5 pour chaque filtre
  # La couche de convolution est, généralement suivie par une "couche" qui applique une fonction d'activation
  # sur l'output de lacouche de convolution, ici nous utiliserons l'actifaion ReLU (Rectified Linear Unit)
  # Nous utiliserons le "same padding". Cela, pour conserver les dimension d'hauteur et largeur de la convolution précédente
  # Le tensor d'input est de dimension : [batch_size, 28, 28, 1]
  # Le tensor d'output (conv1) est de dimension : [batch_size, 28, 28, 32]
  conv1 = tf.layers.conv2d(
      inputs=input_layer,
      filters=32,
      kernel_size=[5, 5],
      padding="same",
      activation=tf.nn.relu)

  # Couche de convolution #1
  # Nous utiliserons une couche de max-pooling avec un filtre de dimension 2x2 et un pas ("stride" en anglais) de 2
  # Le tensor d'input est de dimension : [batch_size, 28, 28, 32]
  # Le tensor d'output (pool1) est de dimension : [batch_size, 14, 14, 32]
  pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

  # Convolutional Layer #2
  # Effectue 64 convolutions de dimension 5x5 pour chaque filtre.
  # Nous utiliserons le "same padding".
  # Le tensor d'input est de dimension : [batch_size, 14, 14, 32]
  # Le tensor d'output (conv2) est de dimension : [batch_size, 14, 14, 64]
  conv2 = tf.layers.conv2d(
      inputs=pool1,
      filters=64,
      kernel_size=[5, 5],
      padding="same",
      activation=tf.nn.relu)

  # Couche de pooling #2
  # Nous utiliserons une couche de max-pooling avec un filtre de dimension 2x2 et un pas ("stride" en anglais) de 2
  # Le tensor d'input est de dimension : [batch_size, 14, 14, 64]
  # Le tensor d'output (pool2) est de dimension : [batch_size, 7, 7, 64]
  pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

  # Couche de convolution #3
  # Effectue 128 convolutions de dimension 3x3 pour chaque filtre.
  # Padding is added to preserve width and height.
  # Le tensor d'input est de dimension : [batch_size, 7, 7, 64]
  # Le tensor d'output (conv3) est de dimension : [batch_size, 7, 7, 128]
  conv3 = tf.layers.conv2d(
      inputs=pool2,
      filters=128,
      kernel_size=[3, 3],
      padding="same",
      activation=tf.nn.relu)
  
  # Couche de pooling #3
  # Nous utiliserons une couche de max-pooling avec un filtre de dimension 2x2 et un pas ("stride" en anglais) de 2
  # Le tensor d'input est de dimension : [batch_size, 7, 7, 128]
  # Le tensor d'output (pool2) est de dimension : [batch_size, 3, 3, 128]
  pool3 = tf.layers.max_pooling2d(inputs=conv3, pool_size=[3, 3], strides=2)

  # Nous transformons la dimension du tensor précdent (bacth_size, 3, 3, 128) 
  # en un tensor 2D de dimension (batch_size, 3 * 3 * 128)
  # Le tensor d'input est de dimension : [batch_size, 3, 3, 128]
  # Le tensor d'output (pool3_flat) est de dimension : [batch_size, 3 * 3 * 128]
  pool3_flat = tf.reshape(pool3, [-1, 3 * 3 * 128])

  # Couche dense (Fully-connected)
  # Couche dense de 1024 neurones (batch_size, 1024)
  # Le tensor d'input est de dimension : [batch_size, 3 * 3 * 128]
  # Le tensor d'output (dense) est de dimension : [batch_size, 1024]
  dense = tf.layers.dense(inputs=pool3_flat, units=1024, activation=tf.nn.relu)

  # Couche dropout; 1 - 0.4 = 0.6 est la probabilité que le "noeud" soit gardé
  dropout = tf.layers.dropout(
      inputs=dense, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)

  # Couche logits
  # Le tensor d'input est de dimension : [batch_size, 1024]
  # Le tensor d'output (logits) est de dimension : [batch_size, 10] (pour signifier les 10 classes, de 0 à 9)
  logits = tf.layers.dense(inputs=dropout, units=10)

  predictions = {
      # Generer les predictions (pour les modes PREDICT et EVAL)
      "classes": tf.argmax(input=logits, axis=1),
      "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
  }
  if mode == tf.estimator.ModeKeys.PREDICT:
    return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

  # Calculer le Loss (perte) (pour les modes TRAIN et EVAL)
  onehot_labels = tf.one_hot(indices=tf.cast(labels, tf.int32), depth=10)
  loss = tf.losses.softmax_cross_entropy(
      onehot_labels=onehot_labels, logits=logits)

  # Configurer l'opération d'entrainement (pour le mode TRAIN)
  if mode == tf.estimator.ModeKeys.TRAIN:
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
    train_op = optimizer.minimize(
        loss=loss,
        global_step=tf.train.get_global_step())
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

  # Calculer les metrics d'évaluation (pour le mode EVAL)
  eval_metric_ops = {
      "accuracy": tf.metrics.accuracy(
          labels=labels, predictions=predictions["classes"])}
  return tf.estimator.EstimatorSpec(
      mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)
      
def main(unused_argv):
  # Charger les données d'entrainement et d'évaluation
  mnist = tf.contrib.learn.datasets.load_dataset("mnist")
  train_data = mnist.train.images  # de type np.array
  train_labels = np.asarray(mnist.train.labels, dtype=np.int32)
  eval_data = mnist.test.images  # de type np.array
  eval_labels = np.asarray(mnist.test.labels, dtype=np.int32)

  # Créer l'estimateur (Estimator)
  mnist_classifier = tf.estimator.Estimator(
      model_fn=cnn_model_fn, model_dir=None)

  # Optionnel
  # Préparer la journalisation pour les prédictions
  # Ajouter le tensor "Softmax" dans la journalisation (avec un key "probabilities")
  # tensors_to_log = {"probabilities": "softmax_tensor"}
  # logging_hook = tf.train.LoggingTensorHook(
  #   tensors = tensors_to_log, every_n_iter=50)

  # Commencer l'entrainement
  train_input_fn = tf.estimator.inputs.numpy_input_fn(
      x={"x": train_data},
      y=train_labels,
      batch_size=100,
      num_epochs=None,
      shuffle=True)
  mnist_classifier.train(
      input_fn=train_input_fn,
      steps=2500
      #,
      #hooks=[logging_hook]
      )

  # Evaluer le modèle et afficher les résultats
  eval_input_fn = tf.estimator.inputs.numpy_input_fn(
      x={"x": eval_data},
      y=eval_labels,
      num_epochs=1,
      shuffle=False)
  eval_results = mnist_classifier.evaluate(input_fn=eval_input_fn)
  print(eval_results)
  
if __name__ == "__main__":
 tf.app.run()
