import argparse
import numpy
import PIL
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import random

class GA:
    def __init__(self, model, image, threshold, max_iterations):
        # image: numpy array
        self.original_image = image
        self.image_shape = image.shape
        self.width = self.image_shape[1]
        self.height = self.image_shape[2]

        self.max_percent_changed = 0.1 # 10%
        self.total_pixel_number = self.width * self.height
        self.max_pixels_changed = int(self.total_pixel_number * self.max_percent_changed)

        self.change_pixel_by_list = [0.8, 0.9, 1.1, 1.2]

        self.population = []
        self.population_size = threshold
        self.max_iterations = max_iterations

        self.model = model
        self.set_true_label()

    def get_change_pixel_by(self):
        index = numpy.random.choice(len(self.change_pixel_by_list))
        return self.change_pixel_by_list[index]
    
    def set_true_label(self):
        """
        Identifies and stores the original image classification
        """
        index, prob = self.predict(self.original_image)
        self.true_label = index

    def generate_random_modified_image(self, max_pixels_changed):
        """ TODO: mukha
        generates new copy of an image with random pixels modified
        """
        new_image = numpy.copy(image)
        # Main logic lies here. Modification may be required
        for change_count in range(max_pixels_changed):
            x_coordinate = random.randint(0, self.width - 1)
            y_coordinate = random.randint(0, self.height - 1)
            change_pixel_by = self.get_change_pixel_by()
            red = min(new_image[0][x_coordinate][y_coordinate][0] * change_pixel_by, 1.0)
            green = min(new_image[0][x_coordinate][y_coordinate][1] * change_pixel_by, 1.0)
            blue = min(new_image[0][x_coordinate][y_coordinate][2] * change_pixel_by, 1.0)
            new_pixel = [red, green, blue]
            new_image[0][x_coordinate][y_coordinate] = new_pixel # not sure if it is correct way of assigning
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
        initial_max_pixels_changed = int(self.total_pixel_number * 0.03) # 3% of total pixels changed for initial population
        for member_count in range(self.population_size):
            new_member = self.generate_random_modified_image(initial_max_pixels_changed)
            self.population.append(new_member)

    def normalize(self, pred):
        """
        Normalize model prediction result
        Returns normalized array
        """
        min_pred = numpy.amin(pred)
        pred = pred - min_pred if min_pred < 0 else pred
        return tf.keras.utils.normalize(pred, axis=-1, order=1)

    def predict(self, image):
        """
        Returns index with highest probabiltiy and probability
        ex: (1, 0.33)
        """
        pred = self.model.predict(image)
        normalized = self.normalize(pred)
        index = tf.math.argmax(normalized, axis=1).numpy()[0]
        return index, normalized[0][index]

    def compute_fitness(self, image):
        """
        Returns if the highest probability corresponds to the original index
        and the probability
        ex: (False, 0.15)
        """
        index, probability = self.predict(image)
        if index != self.true_label:
            return False, probability
        return True, probability
    
    def population_fitness(self):
        self.fitnesses = [self.compute_fitness(image) for image in self.population]

    def check_population(self):
        for i in range(self.population_size):
            if not self.fitnesses[i][0]:
                self.adversarial = self.population[i]
                return True
        return False

    def get_changed_pixel_coordinates(self, image):
        """ TODO: mukha
        retrieves coordinates of the modified pixels
        """
        changed_pixels_coordinates = []
        for x_coordinate in range(self.width):
            for y_coordinate in range(self.height):
                comparison = image[0][x_coordinate][y_coordinate] != self.original_image[0][x_coordinate][y_coordinate]
                if comparison.any():
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
        first_image_modified_pixels = numpy.array(self.get_changed_pixel_coordinates(image1))
        l1 = len(first_image_modified_pixels)
        second_image_modified_pixels = numpy.array(self.get_changed_pixel_coordinates(image2))
        l2 = len(second_image_modified_pixels)
        for change_count in range(max(l1, l2)):
            first_or_second = random.randint(0, 1)
            selected_image = None
            pixel_coordinate = None
            selected_pixels = None
            pixel_index = None
            if first_or_second == 0: # take pixel from first image
                selected_pixels = first_image_modified_pixels
                pixel_index = numpy.random.choice(len(first_image_modified_pixels))
                selected_image = image1
            else: # take pixel from second image
                selected_pixels = second_image_modified_pixels
                pixel_index = numpy.random.choice(len(second_image_modified_pixels))
                # pixel_coordinate = numpy.random.choice(second_image_modified_pixels)
                selected_image = image2
            pixel_coordinate = selected_pixels[pixel_index]
            x_coordinate = pixel_coordinate[0]
            y_coordinate = pixel_coordinate[1]
            offspring[0][x_coordinate][y_coordinate] = selected_image[0][x_coordinate][y_coordinate]
        # End of main logic

        self.mutate(offspring)
        return offspring
    
    def mutate(self, image):
        """TODO: zhans
        randomly change certain pixels
        image: numpy array
        """
        prob=0.01
        for x in range (self.width):
            for y in range (self.height):
                if random.uniform(0, 1)<prob:
                    red = min(image[0][x][y][0] * self.get_change_pixel_by(), 1.0)
                    green = min(image[0][x][y][1] * self.get_change_pixel_by(), 1.0)
                    blue = min(image[0][x][y][2] * self.get_change_pixel_by(), 1.0)
                    image[0][x][y][:]=[red, green, blue]
        return image

    def sort(self):
        # used only once before creating offsprings
        fitness=[]
        pop=[]
        for i in range(self.population_size):
            fitness.append((self.population[i], self.fitnesses[i]))
	    
        result = sorted(fitness, key=lambda x: x[1][1], reverse=False)
        for el in result:
            pop.append(el[0])
        self.population=pop
        return result

    def adding_offspring(self,index,offspring,probability,result):
        # after each iteration adds the offspring to the population
        pop=[]
        for i in range (len(result)):
            if result[i][1][1]>probability:
                index=i
                break
        result=result[:i]+[(offspring,(index,probability))]+result[i:]
        for el in result[:self.population_size]:
            pop.append(el[0])
        self.population=pop
        return result[:self.population_size]

    def selection(self, result):    
        parents=random.sample(result,4)
        output = sorted(parents, key=lambda x: x[1][1], reverse=False)
        
        return output[0][0]

    def next_generation(self):
        """TODO: zhans
        design selection and crossover, then implement
        """
        result=self.sort()
        index=True
        while index!=False:
            parent1=self.selection(result)
            parent2=self.selection(result)
            offspring1=self.crossover(parent1,parent2)
            offspring1=self.mutate(offspring1)
            index, probability=self.compute_fitness(offspring1)
            result=self.adding_offspring(index,offspring1,probability,result)

        return offspring1
    
    def get_perturbations(self):
        """TODO: zhanto and aza
        combine all functions together
        """
        for i in range(self.max_iterations):
            print("Iteration {}".format(i+1))
            self.initialize_population()
            self.population_fitness()
            if self.check_population():
                break
            self.next_generation()
        
        if hasattr(self, 'adversarial'):
            im = PIL.Image.fromarray((self.adversarial*255.0).astype(numpy.uint8)[0])
            im.save("adversarial.png")
        else:
            print("Could not find any adversarial :(")
    
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

    ga = GA(model, image, 100, 10)
    ga.get_perturbations()

    # random_image = ga.generate_random_modified_image()
    # im = PIL.Image.fromarray((random_image*255.0).astype(numpy.uint8)[0])
    # im.save("random_image.png")
    # # print(ga.get_changed_pixel_coordinates(random_image))

    # ga.initialize_population()
    # print(len(ga.get_changed_pixel_coordinates(ga.population[0])))
    # print(len(ga.get_changed_pixel_coordinates(ga.population[1])))
    # offspring = ga.crossover(ga.population[0], ga.population[1])
    # print(len(ga.get_changed_pixel_coordinates(offspring)))
    # offspringImage = PIL.Image.fromarray((offspring*255.0).astype(numpy.uint8)[0])
    # offspringImage.save("random_image_crossover.png")
    # print(ga.get_changed_pixel_coordinates(offspring))