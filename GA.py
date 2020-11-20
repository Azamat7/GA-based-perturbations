import argparse

class GA:
    def __init__(self, model, image, threshold):
    
    def get_true_label(self):
        """TODO
        """

    def initialize_population(self):
        """TODO
        """

    def compute_fitness(self):
        """TODO
        """
    
    def crossover(self, image1, image2):
        """TODO
        """
    
    def mutate(self, image):
        """TODO
        """

    def next_generation(self):
        """TODO
        """
    
    def get_perturbations(self):
        """TODO
        """
    
class Model:
    def __init__(self, model, perturbations):
        """TODO
        1) only perturbations are used for retraining
        2) train data + perturbations are used for retraining
        """
    
    def retrain_model(self):
        """TODO
        """
    
    def evaluate_model(self):
        """TODO
        Evaluated on whole test data
        """

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='GA-based adversarial perturbation generator')
    parser.add_argument('model_name', type=str,
                    help='Required model name argument')
    parser.add_argument('image_name', type=str,
                    help='Required image name argument')
    parser.add_argument('threshold', type=float, default=0.1,
                    help='Required threshold for percentage of pixels\
                        that can be changed')
    parser.add_argument('-p', '--population_size', type=int, default=25,
                    help="Optional population size argument")

    args = parser.parse_args()
    # model = tf.open(args.model_name)
    # image = open(args.image_name)
