import argparse
import numpy
import PIL
import tensorflow as tf
from tensorflow.keras import datasets, layers, models


class GA:
    def __init__(self, model, image, threshold):
        pass
    
    def get_true_label(self):
        """TODO: zhanto and aza
        identify and store the original image classification
        """
        pass

    def initialize_population(self):
        """TODO: zhans and mukha
        generate POPULATION_SIZE number of parents
        """
        pass

    def compute_fitness(self):
        """TODO: zhanto and aza
        fitness function
        """
        pass
    
    def crossover(self, image1, image2):
        """TODO: zhans and mukha
        generate 1 or 2 offsprings from 2 parents
        uniformal selection of pixels
        image: numpy array
        """
        pass
    
    def mutate(self, image):
        """TODO: zhans and mukha
        randomly change certain pixels
        image: numpy array
        """
        pass

    def next_generation(self):
        """TODO: zhans and mukha
        design selection and crossover, then implement
        """
        pass
    
    def get_perturbations(self):
        """TODO: zhanto and aza
        combine all functions together
        """
        pass
    
class Model:
    def __init__(self, model, perturbations):
        """TODO: zhanto and aza
        1) only perturbations are used for retraining
        2) train data + perturbations are used for retraining
        """
        pass
    
    def retrain_model(self):
        """TODO: zhanto and aza
        retrain the model with new images
        """
        pass
    
    def evaluate_model(self):
        """TODO: zhanto and aza
        Evaluated on whole test data
        """
        pass

if __name__ == "__main__":
    """TODO: zhanto and aza
    uncomment and implement the flow
    """
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