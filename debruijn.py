#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  5 07:03:48 2025

@author: simon
"""
import random
import networkx as nx

# Create all possible (n-1)-tuples
def generate_tuples(alphabet, length, current=""):
    if length == 0:
        return [current]
    result = []
    for char in alphabet:
        result.extend(generate_tuples(alphabet, length-1, current + char))
    return result

def random_debruijn_sequence(alphabet, n, repeating=False):
    """
    Generate a complete De Bruijn sequence of order n over the given alphabet using random sampling.

    Args:
        alphabet: List of characters in the alphabet
        n: Order of the De Bruijn sequence

    Returns:
        A complete De Bruijn sequence as a string
    """
    # if isinstance(alphabet, str):
    #     alphabet = alphabet.split()

    k = len(alphabet)

    # For a de Bruijn sequence of order n, we need to create a graph where:
    # - Nodes are (n-1)-tuples
    # - Edges represent the n-tuples (transitions)



    # Generate all (n-1)-tuples as nodes
    nodes = generate_tuples(alphabet, n-1)

    # Create the de Bruijn graph
    G = nx.DiGraph()

    # Add edges between nodes
    for node in nodes:
        for char in alphabet:
            # The source is the current (n-1)-tuple
            source = node
            # The target is the (n-1)-tuple formed by removing the first character
            # and appending the new character
            target = source[1:] + char
            # Add the edge with the character as a label
            G.add_edge(source, target, label=char)

    # Perform a random Eulerian circuit
    # We'll use Hierholzer's algorithm with randomization

    # Start with any node (we'll pick the first one for simplicity)
    current_path = []
    circuit = []

    # Start with a random node
    current = random.choice(nodes)

    # Hierholzer's algorithm with randomization
    while True:
        # If the current node has no outgoing edges, add it to the circuit
        if G.out_degree(current) == 0:
            circuit.append(current)
            if not current_path:
                break
            # Backtrack to the previous node with unused edges
            current = current_path.pop()
        else:
            # Push the current node to the stack
            current_path.append(current)
            # Choose a random neighbor
            neighbors = list(G.successors(current))
            next_node = random.choice(neighbors)
            # Get the edge label
            edge_data = G.get_edge_data(current, next_node)
            # Remove the edge
            G.remove_edge(current, next_node)
            # Move to the next node
            current = next_node

    # Reverse the circuit to get the correct order
    circuit.reverse()

    # Extract the sequence from the circuit
    # The sequence is formed by the last character of each node in the circuit
    sequence = circuit[0]
    for node in circuit[1:]:
        sequence += node[-1]

    sequence = sequence  # don't create loop

    # Verify the sequence length
    expected_length = k**n + n-1
    if len(sequence) != expected_length:
        # If the sequence is not the expected length, there's an issue
        raise ValueError(f"Generated sequence length {len(sequence)} does not match expected length {expected_length}")

    if repeating == True:
        return sequence[:-n+1]

    return sequence

# Example usage and verification
if __name__ == "__main__":
    # Generate a De Bruijn sequence of order 3 over the alphabet {0,1,2,3}
    alphabet = ['0', '1', '2', '3']
    n = 4
    sequence = random_debruijn_sequence(alphabet, n, repeating=True)
    print(f"Random De Bruijn sequence of order {n} over {alphabet}:")
    print(sequence)
    print(f"Length: {len(sequence)}")

    # Verify that all possible n-tuples appear exactly once
    created_tuples = set()
    for i in range(len(sequence)-n+1):
        # Use modular arithmetic to handle the cyclic nature of the sequence
        tuple_str = ''.join(sequence[(i + j)] for j in range(n))
        created_tuples.add(tuple_str)
    all_possible_tuples = set(generate_tuples(alphabet, n))

    print(f"Number of unique {n}-tuples: {len(created_tuples)}")
    print(f"Expected number: {len(all_possible_tuples)}")

    # Check if any n-tuples are missing
    missing_tuples = all_possible_tuples - created_tuples

    if missing_tuples:
        print(f"Missing tuples: {missing_tuples}")
    else:
        print("All tuples are present exactly once!")
