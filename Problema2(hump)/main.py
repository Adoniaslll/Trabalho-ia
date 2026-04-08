import random
from statistics import mean

# ========================================
# Problema: Three-Hump Camel (CB3)
# f(x) = 2*x1^2 - 1.05*x1^4 + x1^6/6 + x1*x2 + x2^2
# Domínio: -5 <= x1, x2 <= 5
# Optimal Value: f(x*) = 0 em (0, 0)
# ========================================

DOMAIN = (-5.0, 5.0)
DIMENSION = 2
OPTIMAL_VALUE = 0.0


POPULATION_SIZE = 50
MAX_GENERATIONS = 1000
CROSSOVER_RATE = 0.8
MUTATION_RATE = 0.2
MUTATION_STEP = 0.3
TOURNAMENT_SIZE = 3
RUNS = 100
SUCCESS_TOLERANCE = 0.01


def fitness(individual):
    x1, x2 = individual
    return 2 * (x1 ** 2) - 1.05 * (x1 ** 4) + (x1 ** 6) / 6 + x1 * x2 + (x2 ** 2)


def clamp(value, lower, upper):
    return max(lower, min(upper, value))


def create_individual():
    low, high = DOMAIN
    return [random.uniform(low, high) for _ in range(DIMENSION)]


def create_population():
    return [create_individual() for _ in range(POPULATION_SIZE)]


def evaluate_population(population):
    scored = []
    for ind in population:
        fit = fitness(ind)
        scored.append((ind, fit))
    scored.sort(key=lambda item: item[1])
    return scored


def tournament_selection(scored_population):
    candidates = random.sample(scored_population, TOURNAMENT_SIZE)
    winner = min(candidates, key=lambda item: item[1])
    return winner[0][:]


def crossover(parent1, parent2):
    if random.random() > CROSSOVER_RATE:
        return parent1[:], parent2[:]

    child1 = []
    child2 = []
    for g1, g2 in zip(parent1, parent2):
        alpha = random.random()
        c1 = alpha * g1 + (1 - alpha) * g2
        c2 = alpha * g2 + (1 - alpha) * g1
        child1.append(c1)
        child2.append(c2)
    return child1, child2


def mutate(individual):
    low, high = DOMAIN
    for i in range(DIMENSION):
        if random.random() < MUTATION_RATE:
            individual[i] += random.uniform(-MUTATION_STEP, MUTATION_STEP)
            individual[i] = clamp(individual[i], low, high)
    return individual


def run_ga():
    nfe = 0
    population = create_population()

    best_individual = None
    best_fitness = float("inf")

    for generation in range(1, MAX_GENERATIONS + 1):
        scored = evaluate_population(population)
        nfe += len(population)

        current_best, current_best_fit = scored[0][0][:], scored[0][1]
        if current_best_fit < best_fitness:
            best_individual = current_best
            best_fitness = current_best_fit

        if abs(best_fitness - OPTIMAL_VALUE) < SUCCESS_TOLERANCE:
            return {
                "success": True,
                "generation": generation,
                "nfe": nfe,
                "best_individual": best_individual,
                "best_fitness": best_fitness,
            }

        new_population = [current_best]

        while len(new_population) < POPULATION_SIZE:
            parent1 = tournament_selection(scored)
            parent2 = tournament_selection(scored)
            child1, child2 = crossover(parent1, parent2)
            new_population.append(mutate(child1))
            if len(new_population) < POPULATION_SIZE:
                new_population.append(mutate(child2))

        population = new_population

    return {
        "success": abs(best_fitness - OPTIMAL_VALUE) < SUCCESS_TOLERANCE,
        "generation": MAX_GENERATIONS,
        "nfe": nfe,
        "best_individual": best_individual,
        "best_fitness": best_fitness,
    }


def main():
    results = [run_ga() for _ in range(RUNS)]
    success_count = sum(1 for r in results if r["success"])
    success_rate = (success_count / RUNS) * 100
    average_nfe = mean(r["nfe"] for r in results)
    best_run = min(results, key=lambda r: r["best_fitness"])

    print("=== Algoritmo Genético Simples - Three-Hump Camel ===")
    print(f"Execuções: {RUNS}")
    print(f"População: {POPULATION_SIZE}")
    print(f"Gerações máximas: {MAX_GENERATIONS}")
    print(f"Taxa de crossover: {CROSSOVER_RATE}")
    print(f"Taxa de mutação: {MUTATION_RATE}")
    print(f"Critério de sucesso: |fBest - fÓtimo| < {SUCCESS_TOLERANCE}")
    print()
    print(f"Taxa de sucesso (SR): {success_rate:.2f}%")
    print(f"NFE médio: {average_nfe:.2f}")
    print(f"Melhor fitness encontrado: {best_run['best_fitness']:.6f}")
    print(f"Melhor indivíduo encontrado: {best_run['best_individual']}")


if __name__ == "__main__":
    main()
