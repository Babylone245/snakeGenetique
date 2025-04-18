import numpy
from NeuralNetwork import *
from snake import *

def eval(sol, gameParams):
    score = 0
    for i in range(gameParams["nbGames"]):
        game = Game(gameParams["height"], gameParams["width"])
        while game.enCours:
            game.direction = sol.nn.predict(game.getFeatures())
            game.refresh()
        score += 1000*game.score+game.steps 
    sol.score = score / (gameParams["nbGames"]* gameParams["height"] * gameParams["width"] * 1000) 
    return sol.score

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
                child1, child2 = croisement(parent1, parent2, pc)
                child1 = mutation(child1, mr)
                child2 = mutation(child2, mr)
                eval(child1, gameParams)
                eval(child2, gameParams)
                enfants.append(child1)
                enfants.append(child2)
            population = selected_population + enfants
            population.sort(reverse=True, key=lambda sol: sol.score)
            print(f"Iteration {i}, Best score: {population[0].score}")
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
        for i in range(parent1_layer.weights.shape[0]): # Multi tableau par des scalaires
            for j in range(parent1_layer.weights.shape[1]):  
                alpha = np.random.rand()
                child1_layer.weights[i, j] = alpha * parent1_layer.weights[i, j] + (1 - alpha) * parent2_layer.weights[i, j]
                child2_layer.weights[i, j] = (1 - alpha) * parent1_layer.weights[i, j] + alpha * parent2_layer.weights[i, j]

        for j in range(parent1_layer.bias.shape[0]): 
            alpha = np.random.rand()
            child1_layer.bias[j] = alpha * parent1_layer.bias[j] + (1 - alpha) * parent2_layer.bias[j]
            child2_layer.bias[j] = (1 - alpha) * parent1_layer.bias[j] + alpha * parent2_layer.bias[j]
    return Individu(child_nn1), Individu(child_nn2)

def mutation(child, mr):
    for layer in child.nn.layers:
        layerSize = layer.outputShape[0]
        pm_biais = mr / layerSize
        previousLayerSize = layer.inputShape[0]
        pm_poids = mr / previousLayerSize

        for n in range(layerSize):
            if np.random.rand() < pm_biais:
                layer.bias[n] += np.random.randn() * 0.1
        for n_p in range(previousLayerSize):
            for n in range(layerSize):
                if np.random.rand() < pm_poids:
                    layer.weights[n_p][n] += np.random.randn() * 0.1

    return child

        