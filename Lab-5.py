import sys
import random

# Helper function to interpret the file line in the format we need
def lineRead(file: str) -> [str]:
    line = file.readline().strip().replace(' ', '').split(',')
    return line

# Helper function to generate samples from any univariate distribution using inversion sampling
def generate_sample_from_distribution(pdf_func, domain):
    cumulative_probabilities = [pdf_func(value) for value in domain]
    total_probability = sum(cumulative_probabilities)
    cumulative_probabilities = [prob / total_probability for prob in cumulative_probabilities]

    rand_num = random.random()
    cumulative_prob = 0.0
    for value, prob in zip(domain, cumulative_probabilities):
        cumulative_prob += prob
        if rand_num < cumulative_prob:
            return value
    return None

    # Function to perform Prior Sampling
def prior_sampling(variableDomainsDict, ParentDictionary, CPTDict):
    sample = {}
    for variable in variableDomainsDict:
        parents = ParentDictionary[variable]
        if not parents:
            sample[variable] = random.choice(variableDomainsDict[variable])
        else:
            parent_values = tuple(sample[parent] for parent in parents)
            probabilities = CPTDict[variable][parent_values]
            sample[variable] = random.choices(variableDomainsDict[variable], weights=probabilities)[0]
    return sample

# Function to perform Rejection Sampling
def rejection_sampling(variableDomainsDict, ParentDictionary, CPTDict, query, evidence, num_samples):
    count_query_true = 0
    for _ in range(num_samples):
        sample = prior_sampling(variableDomainsDict, ParentDictionary, CPTDict)
        evidence_match = all(sample[var] == val for var_val in evidence for var, val in var_val.items())
        query_match = all(sample[var] == val for var_val in query for var, val in var_val.items())
        if evidence_match and query_match:
            count_query_true += 1
    return count_query_true / num_samples

# Function to perform Likelihood Weighting
def likelihood_weighting(variableDomainsDict, ParentDictionary, query_var, query_val, evidence, num_samples):
    count_query_true = 0
    weighted_sum = 0
    for _ in range(num_samples):
        sample = {}
        weight = 1.0
        for variable in variableDomainsDict:
            parents = ParentDictionary[variable]
            if not parents:
                sample[variable] = random.choice(variableDomainsDict[variable])
            else:
                parent_values = tuple(sample[parent] for parent in parents)
                probabilities = CPTDict[variable][parent_values]
                sample[variable] = random.choices(variableDomainsDict[variable], weights=probabilities)[0]
            if variable in evidence:
                weight *= CPTDict[variable][parent_values][variableDomainsDict[variable].index(evidence[variable])]
        weighted_sum += weight
        if sample[query_var] == query_val:
            count_query_true += weight
    return count_query_true / weighted_sum

# Function to perform Gibbs Sampling
def gibbs_sampling(variableDomainsDict, ParentDictionary, query_var, query_val, evidence, num_samples):
    sample = {variable: random.choice(variableDomainsDict[variable]) for variable in variableDomainsDict}
    count_query_true = 0
    for _ in range(num_samples):
        for variable in variableDomainsDict:
            parents = ParentDictionary[variable]
            if not parents:
                sample[variable] = random.choice(variableDomainsDict[variable])
            else:
                parent_values = tuple(sample[parent] for parent in parents)
                probabilities = CPTDict[variable][parent_values]
                sample[variable] = random.choices(variableDomainsDict[variable], weights=probabilities)[0]
            if variable in evidence:
                sample[variable] = evidence[variable]
        if sample[query_var] == query_val:
            count_query_true += 1
    return count_query_true / num_samples

# Main function to read input and perform sampling methods
def main():
    # Read input from the command line argument
    filename = sys.argv[1]
    file = open(filename, "r")

    # Read Bayesian Network structure
    bayesNetsDetails = lineRead(file)
    N = int(bayesNetsDetails[0])
    variableDomainsDict = {details[0]: details[1:] for details in [lineRead(file) for _ in range(N)]}
    ParentDictionary = {}
    CPTDict = {}
    for _ in range(N):
        relation = file.readline().replace(' ', '').strip().split('|')
        variable = relation[0]
        parents = relation[1].split(',') if relation[1] else []
        ParentDictionary[variable] = tuple(parents)
        numberOfTerms = len(variableDomainsDict[variable])
        for parent in parents:
            numberOfTerms *= len(variableDomainsDict[parent])
        CPTParentDict = {}
        for _ in range(numberOfTerms):
            probabilities = lineRead(file)
            CPTParentDict[tuple(probabilities[:-1])] = [float(prob) for prob in probabilities[-1].split('/')]
        CPTDict[variable] = CPTParentDict
    print(CPTDict)

    # Handle the query of form ([{query}], [evidence])
    queryLine = file.readline()
    queryLine = queryLine.replace(' ', '').strip()[queryLine.index('('):-1].split('|')
    query = (
        [{term.split('=')[0]: term.split('=')[1]} for term in queryLine[0].split(',')],  # Query
        [{term.split('=')[0]: term.split('=')[1]} for term in queryLine[1].split(',')],  # Evidence
    )
    # Number of samples for sampling methods
    num_samples = 10000

    # Perform computations using different sampling methods
    print("Rejection Sampling Result:", rejection_sampling(variableDomainsDict, ParentDictionary, CPTDict, query[0], query[1], num_samples))
    print("Likelihood Weighting Result:", likelihood_weighting(variableDomainsDict, ParentDictionary, query[0][0], query[0][1], query[1], num_samples))
    print("Gibbs Sampling Result:", gibbs_sampling(variableDomainsDict, ParentDictionary, query[0][0], query[0][1], query[1], num_samples))

if __name__ == "__main__":
    main()