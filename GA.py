import argparse
import numpy
import PIL
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
class GA:
    def __init__(self, model, image, threshold):
        pass
    
    def get_true_label(self):
        """TODO
        """
        pass

    def initialize_population(self):
        """TODO
        """
        pass

    def compute_fitness(self):
        """TODO
        """
        pass
    
    def crossover(self, image1, image2):
        """TODO
        """
        pass
    
    def mutate(self, image):
        """TODO
        """
        pass

    def next_generation(self):
        """TODO
        """
        pass
    
    def get_perturbations(self):
        """TODO
        """
        pass
    
class Model:
    def __init__(self, model, perturbations):
        """TODO
        1) only perturbations are used for retraining
        2) train data + perturbations are used for retraining
        """
        pass
    
    def retrain_model(self):
        """TODO
        """
        pass
    
    def evaluate_model(self):
        """TODO
        Evaluated on whole test data
        """
        pass

if __name__ == "__main__":
    # parser = argparse.ArgumentParser(description='GA-based adversarial perturbation generator')
    # parser.add_argument('model_name', type=str,
    #                 help='Required model name argument')
    # parser.add_argument('image_name', type=str,
    #                 help='Required image name argument')
    # parser.add_argument('threshold', type=float, default=0.1,
    #                 help='Required threshold for percentage of pixels\
    #                     that can be changed')
    # parser.add_argument('-p', '--population_size', type=int, default=25,
    #                 help="Optional population size argument")

    # args = parser.parse_args()

    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

    # Open image as numpy array
    image = numpy.asarray(PIL.Image.open('automobile10.png'))
    image = image/255.0
    image = image.reshape(-1, 32, 32, 3)

    model = tf.keras.models.load_model('cifar10')
    model.compile(optimizer='adam',
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=['accuracy'])
    
    # Get prediction
    pred = model.predict(image)
    print(pred)

    # Get predicted class name
    index = tf.math.argmax(pred, axis=1).numpy()[0]
    print("Predicted class is {}".format(class_names[index]))