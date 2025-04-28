import numpy
from NeuralNetwork import *
from snake import *
from concurrent.futures import ProcessPoolExecutor

def eval_wrapper(args):
    sol, gameParams = args
    return eval(sol, gameParams)

def eval(sol, gameParams):
    score = 0
    for i in range(gameParams["nbGames"]):
        game = Game(gameParams["height"], gameParams["width"])
        while game.enCours:
            game.direction = sol.nn.predict(game.getFeatures())
            game.refresh()
        score += 1000*game.score+game.steps 
    sol.score = score / (gameParams["nbGames"]* gameParams["height"] * gameParams["width"] * 1000) 
    return sol

'''
Représente une solution avec
_un réseau de neurones
_un score (à maximiser)

vous pouvez ajouter des attributs ou méthodes si besoin
'''
class Individu:
    def __init__(self, nn):
        self.nn = nn
        self.score = 0


'''
La méthode d'initialisation de la population est donnée :
_on génère N individus contenant chacun un réseau de neurones (de même format)
_on évalue et on trie des individus
'''
def initialization(taillePopulation, arch, gameParams):
    population = []
    for i in range(taillePopulation):
        nn = NeuralNetwork((arch[0],))
        for j in range(1, len(arch)):
            nn.addLayer(arch[j], "elu")
        population.append(Individu(nn))

    for sol in population: eval(sol, gameParams)
    population.sort(reverse=True, key=lambda sol:sol.score)
    
    return population

def optimize(taillePopulation, tailleSelection, pc, arch, gameParams, nbIterations, nbThreads, scoreMax, mr):
    population = initialization(taillePopulation, arch, gameParams)

    for i in range(nbIterations):
        selected_population = population[:tailleSelection]
        enfants = []
        
        while len(enfants) < taillePopulation - tailleSelection:

            parent1 = numpy.random.choice(selected_population)
            parent2 = numpy.random.choice(selected_population)

            # parent1 = selected_population[0]
            # parent2 = selected_population[1]

            child1, child2 = croisement(parent1, parent2, pc)
            
            mutation(child1, mr)
            mutation(child2, mr)
            
            enfants.append(child1)
            enfants.append(child2)

        with ProcessPoolExecutor(max_workers=nbThreads) as executor:
            enfants = list(executor.map(eval_wrapper, [(sol, gameParams) for sol in enfants]))
        
        population = selected_population + enfants
        
        population.sort(reverse=True, key=lambda sol: sol.score)
        
        print(f"Iteration {i}, Best score: {population[0].score}")
        
        if population[0].score >= scoreMax:
            print(f"Target score {scoreMax} reached, stopping optimization early")
            break
    
    return population[0].nn

def croisement(parent1, parent2, pc):
    if np.random.rand() > pc:
        child1 = Individu(parent1.nn.clone())
        child2 = Individu(parent2.nn.clone())
        return child1, child2

    child_nn1 = NeuralNetwork(parent1.nn.inputShape)
    child_nn2 = NeuralNetwork(parent2.nn.inputShape)

    for layer in parent1.nn.layers:
        child_nn1.addLayer(layer.outputShape[0], "elu")
        child_nn2.addLayer(layer.outputShape[0], "elu")

    for layer_idx in range(len(parent1.nn.layers)):
        parent1_layer = parent1.nn.layers[layer_idx]
        parent2_layer = parent2.nn.layers[layer_idx]
        child1_layer = child_nn1.layers[layer_idx]
        child2_layer = child_nn2.layers[layer_idx]

        alpha_weights = np.random.rand(*parent1_layer.weights.shape)
        alpha_bias = np.random.rand(*parent1_layer.bias.shape)

        child1_layer.weights = alpha_weights * parent1_layer.weights + (1 - alpha_weights) * parent2_layer.weights
        child2_layer.weights = (1 - alpha_weights) * parent1_layer.weights + alpha_weights * parent2_layer.weights

        child1_layer.bias = alpha_bias * parent1_layer.bias + (1 - alpha_bias) * parent2_layer.bias
        child2_layer.bias = (1 - alpha_bias) * parent1_layer.bias + alpha_bias * parent2_layer.bias

    return Individu(child_nn1), Individu(child_nn2)

def mutation(child, mr):
    for layer in child.nn.layers:
        layerSize = layer.outputShape[0]
        previousLayerSize = layer.inputShape[0]

        pm_biais = mr / layerSize
        pm_poids = mr / previousLayerSize

        mask_biais = np.random.rand(layerSize) < pm_biais
        mutations_biais = np.random.randn(layerSize) * 0.1
        layer.bias += mask_biais * mutations_biais

        mask_poids = np.random.rand(previousLayerSize, layerSize) < pm_poids
        mutations_poids = np.random.randn(previousLayerSize, layerSize) * 0.1
        layer.weights += mask_poids * mutations_poids

        