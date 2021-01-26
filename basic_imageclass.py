import tensorflow as tf 
import numpy as np
import matplotlib.pyplot as plt

#visualize predict result as images
def plot_image(i,predict_array,true_label,img):
    true_label,img = true_label[i],img[i]
    plt.xticks([])
    plt.yticks([])
    plt.imshow(img,cmap=plt.cm.binary)

    predicted_label = np.argmax(predict_array)
    if predicted_label ==true_label:
        color = 'blue'
    else:
        color = 'red'
    plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                    100*np.max(predict_array),
                                    class_names[true_label],
                                    color=color))
#visualize predict label as graphs
def plot_value_array(i, predictions_array, true_label):
  true_label = true_label[i]
  plt.grid(False)
  plt.xticks(range(10))
  plt.yticks([])
  thisplot = plt.bar(range(10), predictions_array, color="#777777")
  plt.ylim([0, 1])
  predicted_label = np.argmax(predictions_array)

  thisplot[predicted_label].set_color('red')
  thisplot[true_label].set_color('blue')



#load data
fashion_mnist = tf.keras.datasets.fashion_mnist

(train_imgs,train_labels), (test_imgs,test_labels) = fashion_mnist.load_data()

class_names = [
                'T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'
            ]


#preprocess data
train_imgs = train_imgs/ 255.0
test_imgs = test_imgs /255.0


#check data 

# plt.figure(figsize=(10,10))
# for i in range(25):
#     plt.subplot(5,5,i+1)
#     plt.xticks([])
#     plt.yticks([])
#     plt.imshow(train_imgs[i],cmap=plt.cm.binary)
#     plt.xlabel(class_names[train_labels[i]])
# plt.show()


#model layers
'''
first layers transform form 2-d array(28 by 28) to 1-d array (28*28=784)
second layers: hidden node with 128 node by relu function
third node: fully connected layers (give score for 10 result) 
'''

model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28,28)), 
    tf.keras.layers.Dense(128,activation='relu'),
    tf.keras.layers.Dense(10)
])

#select hyperparameter here
model.compile(
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
)

#train your model
model.fit(train_imgs,train_labels ,epochs=10)


#accuracy results
test_loss,test_acc =model.evaluate(test_imgs,test_labels,verbose=2)
print('\nTest acc:',test_acc)


#test your model with test set
prob_model = tf.keras.Sequential([model,tf.keras.layers.Softmax()])

prediction = prob_model.predict(test_imgs)

print('predict:',np.argmax(prediction[0]),'\nanswer:',test_labels[0])


#visualize ansewer as graph
i = 0
plt.figure(figsize=(6,3))
plt.subplot(1,2,1)
plot_image(i,prediction[i],test_labels,test_imgs)
plt.subplot(1,2,2)
plot_value_array(i,prediction[i],test_labels)
plt.show()



