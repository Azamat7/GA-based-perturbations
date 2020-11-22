import argparse
import numpy
import PIL
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import random

class GA:
    def __init__(self, model, image, threshold):
        # image: numpy array
        self.original_image = image
        self.image_shape = image.shape
        self.width = self.image_shape[0]
        self.height = self.image_shape[1]
        self.max_percent_changed = 0.1 # 10%
        self.max_pixels_changed = image_size * max_percent_changed
        self.change_pixel_by = 1.1
        self.population = []
        self.population_size = 100
        pass
    
    def get_true_label(self):
        """TODO: zhanto and aza
        identify and store the original image classification
        """
        pass

    def generate_random_modified_image(self):
        """ TODO: mukha
        generates new copy of an image with random pixels modified
        """
        new_image = numpy.copy(image)
        # Main logic lies here. Modification may be required
        for change_count in range(max_pixels_changed):
            x_coordinate = random.randint(0, self.width - 1)
            y_coordinate = random.randint(0, self.height - 1)
            red = min(new_image[x_coordinate][y_coordinate][0] * change_pixel_by, 1.0)
            green = min(new_image[x_coordinate][y_coordinate][1] * change_pixel_by, 1.0)
            blue = min(new_image[x_coordinate][y_coordinate][2] * change_pixel_by, 1.0)
            new_pixel = [red, green, blue]
            new_image[x_coordinate][y_coordinate][:] = new_pixel # not sure if it is correct way of assigning
        # End of main logic
        
        return new_image

    def initialize_population(self):
        """TODO: mukha
        generates POPULATION_SIZE number of parents.
        For each new parent, we only change at most
        fixed number of pixels, maxPixelsChanged.
        Each modified pixel would differ from the
        original image's pixel by some constant
        multiple.
        """
        for member_count in range(self.population_size):
            new_member = self.generate_random_modified_image
            self.population.append(new_member)
        pass

    def compute_fitness(self):
        """TODO: zhanto and aza
        fitness function
        """
        pass

    def get_changed_pixel_coordinates(self, image):
        """ TODO: mukha
        retrieves coordinates of the modified pixels
        """
        changed_pixels_coordinates = []
        for x_coordinate in range(self.width):
            for y_coordinate in range(self.height):
                if image[x_coordinate][y_coordinate] != self.original_image[x_coordinate][y_coordinate]:
                    changed_pixels_coordinates.append((x_coordinate, y_coordinate))

        return changed_pixels_coordinates
    
    def crossover(self, image1, image2):
        """TODO: mukha
        generates 1 offspring from 2 parents
        uniformal selection of pixels
        image: numpy array

        We first create copy of unmodified
        original image.
        We select at most maxPixelsChanged
        number of modified pixels from either
        parent and add to the offspring.
        Afterwards, apply mutation and add
        the offspring to the current population.
        """
        offspring = numpy.copy(image)

        # Main logic lies here. Modification may be required
        first_image_modified_pixels = self.get_changed_pixel_coordinates(image1)
        second_image_modified_pixels = self.get_changed_pixel_coordinates(image2)
        for change_count in range(self.max_pixels_changed):
            first_or_second = random.randint(0, 1)
            selected_image = None
            pixel_coordinate = None
            if first_or_second == 0: # take pixel from first image
                pixel_coordinate = numpy.random.choice(first_image_modified_pixels)
                selected_image = image1
            else: # take pixel from second image
                pixel_coordinate = numpy.random.choice(second_image_modified_pixels)
                selected_image = image2
            x_coordinate = pixel_coordinate[0]
            y_coordinate = pixel_coordinate[1]
            offspring[x_coordinate][y_coordinate] = selected_image[x_coordinate][y_coordinate]
        # End of main logic

        self.mutate(offspring)
        return offspring
    
    def mutate(self, image):
        """TODO: zhans
        randomly change certain pixels
        image: numpy array
        """
        pass

    def next_generation(self):
        """TODO: zhans
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