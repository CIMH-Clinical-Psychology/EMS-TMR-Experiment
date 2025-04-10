#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  5 07:03:48 2025

@author: simon
"""
import random
import networkx as nx
import numpy as np
import warnings
import pandas as pd
import math


# Create all possible (n-1)-tuples

def assign_retrieval_other_old(original_df):
    """this is a Gemini2.5pro translated version of Julis code to balance
    the stimuli that can be found here: https://gitlab.zi.local/klips/panosam-code/anticlust_ems_tmr/-/blob/main/01_retrieval_lists.R"""
    counts_df = original_df.groupby(['category', 'cued'], observed=False).size().reset_index(name='n')
    categories_list = list(set(original_df.category))
    cued_categories = list(set(original_df[original_df.cued].category))

    x = original_df.copy()
    x['other_old'] = pd.NA

    for i, row in counts_df.iterrows():
        current_category = row['category']
        current_cued_status = row['cued']
        n_to_sample_total = row['n']

        sample_cats = [cat for cat in categories_list if cat != current_category]
        num_sample_cats = len(sample_cats)

        for cat in sample_cats:
            assigned_stims_so_far = x['other_old'].dropna().tolist()

            if cat in cued_categories:
                n_needed_per_group = math.floor(n_to_sample_total / num_sample_cats / 2)

                available_cued_indices = x.index[
                    (x['category'] == cat) &
                    x['cued'] &
                    ~x['correct'].isin(assigned_stims_so_far)
                ].tolist()
                available_uncued_indices = x.index[
                    (x['category'] == cat) &
                    ~x['cued'] &
                    ~x['correct'].isin(assigned_stims_so_far)
                ].tolist()

                sampled_cued_indices = random.sample(available_cued_indices, k=n_needed_per_group)
                sampled_uncued_indices = random.sample(available_uncued_indices, k=n_needed_per_group)

                temp_sample_indices = sampled_cued_indices + sampled_uncued_indices
                random.shuffle(temp_sample_indices)
                temp_stims = x.loc[temp_sample_indices, 'correct'].tolist()

            else: # cat is not cued
                n_needed = math.floor(n_to_sample_total / num_sample_cats)
                available_indices = x.index[
                    (x['category'] == cat) &
                    ~x['correct'].isin(assigned_stims_so_far) # All are uncued implicitly if cat not in cued_cats
                ].tolist()

                temp_sample_indices = random.sample(available_indices, k=n_needed)
                random.shuffle(temp_sample_indices)
                temp_stims = x.loc[temp_sample_indices, 'correct'].tolist()

            assign_to_indices = x.index[
                (x['category'] == current_category) &
                (x['cued'] == current_cued_status) &
                x['other_old'].isna()
            ].tolist()

            assign_to_sample = random.sample(assign_to_indices, k=len(temp_stims))
            x.loc[assign_to_sample, 'other_old'] = temp_stims
    new_df = x
    empty_slots_indices = new_df.index[new_df['other_old'].isna() & (new_df['correct'] != "none")]
    assigned_stims = new_df['other_old'].dropna().tolist()
    to_assign_df = new_df[~new_df['correct'].isin(assigned_stims) & (new_df['correct'] != "none")]
    remaining_stims_to_assign = to_assign_df['correct'].tolist()

    if len(empty_slots_indices) == len(remaining_stims_to_assign):
        random.shuffle(remaining_stims_to_assign)
        new_df.loc[empty_slots_indices, 'other_old'] = remaining_stims_to_assign

    # Derive other_cat and handle potential conflicts
    new_df['other_cat'] = new_df['other_old'].astype(str).str.replace('/.*$', '', regex=True)
    new_df['other_cat'] = new_df['other_cat'].replace({'<NA>': 'none', 'nan': 'none'}) # Handle potential string representations of NA
    new_df.loc[new_df['other_old'].isna(), 'other_cat'] = 'none' # Ensure actual NAs become 'none'

    filter_condition = new_df['category'] != "none"
    while (new_df.loc[filter_condition, 'category'] == new_df.loc[filter_condition, 'other_cat']).any():
        random.shuffle(remaining_stims_to_assign)
        new_df.loc[empty_slots_indices, 'other_old'] = remaining_stims_to_assign
        new_df['other_cat'] = new_df['other_old'].astype(str).str.replace('/.*$', '', regex=True)
        new_df['other_cat'] = new_df['other_cat'].replace({'<NA>': 'none', 'nan': 'none'})
        new_df.loc[new_df['other_old'].isna(), 'other_cat'] = 'none'

    return new_df

def generate_tuples(alphabet, length, current=""):
    if length == 0:
        return [current]
    result = []
    for char in alphabet:
        result.extend(generate_tuples(alphabet, length-1, current + char))
    return result

def longest_streak(iterable):
    max_streak = 1
    current_streak = 1
    current_item = iterable[0]

    for item in iterable[1:]:
        if item == current_item:
            current_streak += 1
        else:
            current_streak = 1
            current_item = item

        max_streak = max(max_streak, current_streak)

    return max_streak

def place_distractors(items, proba=0.2, block_surrounding=1):
    """given a list of items places the distractor positions such that
    each item is equally often a distractor"""
    items = np.array(items)
    uniques, counts = np.unique(items, return_counts=True)
    assert np.std(counts)==0, 'not all items are equally often'
    per_item = np.round(counts[0]*proba)
    if (per_item * 1/proba)!=counts[0]:
        warnings.warn(f'Cant make probability exactly {proba}, is now {per_item/counts[0]}')

    is_distractor = np.zeros(len(items))
    p = np.array([1/len(items)]*len(items))
    idx = np.arange(len(items))

    for item in uniques:
        for n in range(int(per_item)):
            idx_sub = idx[items==item]
            p_sub = p[items==item]
            assert len(idx_sub)==counts[0]  # sanity check
            ix = np.random.choice(idx_sub, p=p_sub/p_sub.sum())
            is_distractor[ix] = True

            # now normalize p vector such that neighbouring items are less likely
            p[ix] = 0
            if block_surrounding:
                for offset in range(1, block_surrounding+1):
                    if ix-offset >= 0:
                        p[ix-offset] **= 3
                    if ix+offset < len(p):
                        p[ix+offset] **= 3

    assert sum(is_distractor)==(len(uniques)*per_item)
    return is_distractor.astype(int)


def shift(seq, n):
    """np.roll for lists"""
    if not isinstance(seq, list):
        seq = list(seq)
    n = n % len(seq)
    return seq[n:] + seq[:n]


def random_debruijn_sequence(alphabet, n, repeating=True, check_transitions=False):
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
        sequence = sequence[:-n+1]

    # sanity check that each transition is present exactly once
    if check_transitions:
        step2_transitions = generate_tuples(alphabet, 2)
        transitions = list(zip(sequence[:-1], sequence[1:], strict=True))
        transitions = [''.join(x) for x in transitions]
        uniques, counts = np.unique(transitions, return_counts=True, axis=0)
        assert set(counts)=={1}, \
            f'{dict(zip(uniques, counts))=} not equal'
        assert set(step2_transitions) == set(uniques)
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
