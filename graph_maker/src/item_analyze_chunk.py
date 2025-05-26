import spacy
from spacy.tokens import Span, Token
from pymorphy3 import MorphAnalyzer
import statistics
import networkx as nx
from community import best_partition
from functools import lru_cache
import numpy as np
import pandas as pd
import pickle
import os
import time
import re
from scipy.stats import skew, kurtosis, norm, entropy
from collections import Counter, defaultdict
from joblib import Parallel, delayed
import argparse 

try:
    from utils import KEY_DEP_RATIO_PAIRS 
except ImportError:
    print("CRITICAL: utils.py with KEY_DEP_RATIO_PAIRS not found. Ratios will be missing.")
    KEY_DEP_RATIO_PAIRS = []

parser = argparse.ArgumentParser(description="Process a chunk of text items.")
parser.add_argument("--task_id", type=int, required=True, help="Current task ID (0-indexed).")
parser.add_argument("--total_chunks", type=int, required=True, help="Total number of chunks.")
args = parser.parse_args()

TASK_ID = args.task_id
TOTAL_CHUNKS = args.total_chunks

# logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - CONSOLIDATION - %(levelname)s - %(message)s') # Заменено на print

SPACY_MODEL = "ru_core_news_lg" 
GENERIC_ENTITY_MAP = {'PER': 'Человек', 'LOC': 'Место', 'ORG': 'Организация', 'GPE': 'Место', 'FAC': 'Объект', 'NORP': 'Национальность', 'PRODUCT': 'Продукт', 'EVENT': 'Событие', 'WORK_OF_ART': 'Произведение'} 
morph = MorphAnalyzer() 

ANEK_FILE = '/home/mmnima/jokes/graph_maker/jokes_generated.txt' 
NOT_ANEKS_FILE = '/home/mmnima/jokes/graph_maker/sentences_generated.txt' 

OUTPUT_DIR_ANALYSIS = 'analysis_results_LONG_CHOPPED_LLM' 
INPUT_DIR_GRAPH = '/home/mmnima/jokes/graph_maker/graph_data' 

ITEM_CORPUS_FRACTION = 1.0 

GLOBAL_WORKERS = 1 

if not os.path.exists(OUTPUT_DIR_ANALYSIS): 
    os.makedirs(OUTPUT_DIR_ANALYSIS) 

JOKE_STATS_FILENAME_CHUNK = os.path.join(OUTPUT_DIR_ANALYSIS, f'joke_statistics_task_{TASK_ID}.csv')
SENTENCE_STATS_FILENAME_CHUNK = os.path.join(OUTPUT_DIR_ANALYSIS, f'sentence_statistics_task_{TASK_ID}.csv')

GENERAL_GRAPH_GLOBAL = None 
GENERAL_NODES_GLOBAL = None 
COMMUNITIES_MAP_GLOBAL = {} 
GENERAL_EDGES_SET_GLOBAL = set() 
GLOBAL_BG_CENTRALITY_DEGREE = {} 
GLOBAL_BG_CENTRALITY_BETWEENNESS = {} 
GLOBAL_BG_CENTRALITY_CLOSENESS = {} 
GLOBAL_BG_CENTRALITY_EIGENVECTOR = {} 
GLOBAL_EDGE_WEIGHT_PERCENTILES = {} 
GLOBAL_HUBS_NODES = [] 

graphml_filename = '/home/mmnima/jokes/graph_maker/graph_data/general_on_long_graph_1400.graphml' 
global_data_filename = '/home/mmnima/jokes/graph_maker/graph_data/global_on_long_graph_data.pkl' 

try:
    if os.path.exists(graphml_filename): 
        GENERAL_GRAPH_GLOBAL = nx.read_graphml(graphml_filename) 
        print(f"General graph loaded from {graphml_filename}") 
    else:
        print(f"Graph file not found: {graphml_filename}. Exiting.") 
        exit()
    if os.path.exists(global_data_filename): 
        with open(global_data_filename, 'rb') as f: 
            global_data = pickle.load(f) 
            if not isinstance(global_data, dict): 
                print(f"Loaded global data from {global_data_filename} is not a dictionary. Exiting.") 
                exit()
            GENERAL_NODES_GLOBAL = global_data.get('nodes', set()) 
            if not isinstance(GENERAL_NODES_GLOBAL, set): 
                print(f"Loaded 'nodes' from {global_data_filename} is not a set. Exiting.") 
                exit()
            COMMUNITIES_MAP_GLOBAL = global_data.get('communities', {}) 
            GLOBAL_BG_CENTRALITY_DEGREE = global_data.get('bg_degree', {}) 
            GLOBAL_BG_CENTRALITY_BETWEENNESS = global_data.get('bg_betweenness', {}) 
            GLOBAL_BG_CENTRALITY_CLOSENESS = global_data.get('bg_closeness', {}) 
            GLOBAL_BG_CENTRALITY_EIGENVECTOR = global_data.get('bg_eigenvector', {}) 
        print(f"Global graph data loaded from {global_data_filename}") 
    else:
        print(f"Required global graph data file not found: {global_data_filename}. Exiting.") 
        exit()
except Exception as e:
    print(f"Failed to load global graph data: {e}") 
    exit()

if GENERAL_GRAPH_GLOBAL is None or GENERAL_GRAPH_GLOBAL.number_of_nodes() < 2: 
    print("Loaded general graph is too small. Exiting.") 
    exit()
if GENERAL_NODES_GLOBAL is None: 
    print("GENERAL_NODES_GLOBAL is None. Exiting.") 
    exit()
if not COMMUNITIES_MAP_GLOBAL: 
     print("Community detection data not loaded or empty. Community-based metrics may be zero.") 

try:
    nlp = spacy.load(SPACY_MODEL, disable=["textcat"]) 
    print(f"Модель SpaCy '{SPACY_MODEL}' успешно загружена.") 
except OSError:
    print(f"Модель SpaCy '{SPACY_MODEL}' не найдена. Exiting.") 
    exit()

if GENERAL_GRAPH_GLOBAL and GENERAL_GRAPH_GLOBAL.number_of_edges() > 0: 
    all_edge_weights = [data.get('weight', 1.0) for u, v, data in GENERAL_GRAPH_GLOBAL.edges(data=True)] 
    if all_edge_weights: 
        GLOBAL_EDGE_WEIGHT_PERCENTILES['p25'] = np.percentile(all_edge_weights, 25) 
        GLOBAL_EDGE_WEIGHT_PERCENTILES['p50'] = np.percentile(all_edge_weights, 50) 
        GLOBAL_EDGE_WEIGHT_PERCENTILES['p75'] = np.percentile(all_edge_weights, 75) 
        print("Global edge weight percentiles calculated.") 
    else:
        print("No edge weights found in global graph for percentile calculation.") 

if GLOBAL_BG_CENTRALITY_DEGREE: 
    top_n_hubs = 100 
    GLOBAL_HUBS_NODES = sorted(GLOBAL_BG_CENTRALITY_DEGREE, key=GLOBAL_BG_CENTRALITY_DEGREE.get, reverse=True)[:top_n_hubs] 
    print(f"Top {len(GLOBAL_HUBS_NODES)} global hubs identified.") 
else:
    print("Global degree centrality not available, cannot identify global hubs.") 


def parse_morph(text): 
    try:
        return morph.parse(text) 
    except Exception as e:
        print(f"MorphAnalyzer failed for '{text}': {e}") 
        return []

def calculate_iqr(data): 
    if not data: return 0 
    numeric_data = [x for x in data if isinstance(x, (int, float)) and np.isfinite(x)] 
    if len(numeric_data) < 2: return 0 
    try:
        q1, q3 = np.percentile(numeric_data, [25, 75]) 
        return q3 - q1 
    except Exception as e:
        print(f"Could not calculate IQR for data: {e}") 
        return 0

def sanitize_string_for_xml(s): 
    if not isinstance(s, str): 
        return s 
    s = re.sub(r'[\x00-\x1F\x7F-\x9F]', '', s) 
    return s 

def head_in_named_entity(doc, span): 
    if not span: 
        return None 
    head = span.root if span.root else (span[0] if span else None) 
    if head: 
        morph_info = [part.split('=')[1] for part in str(head.morph).split('|') if '=' in part] 
        return head, morph_info, head.head, head.dep_ 
    return None 

def normalize_noun_phrase(doc, np_obj): 
    if isinstance(np_obj, spacy.tokens.Span) and np_obj.label_ in GENERIC_ENTITY_MAP: 
        return sanitize_string_for_xml(GENERIC_ENTITY_MAP[np_obj.label_]) 
    
    if isinstance(np_obj, spacy.tokens.Token): 
        if np_obj.pos_ in ['ADJF', 'ADV', 'VERB', 'NUM']: 
            return sanitize_string_for_xml(np_obj.lemma_) 
        if np_obj.pos_ in ['NOUN', 'PROPN']: 
            ana_results = parse_morph(np_obj.text) 
            return sanitize_string_for_xml(ana_results[0].normal_form if ana_results else np_obj.lemma_) 
        return sanitize_string_for_xml(np_obj.lemma_) 
        
    elif isinstance(np_obj, spacy.tokens.Span): 
        head_info = head_in_named_entity(doc, np_obj) 
        if head_info is None or head_info[0] is None: 
            return sanitize_string_for_xml(np_obj.root.lemma_ if np_obj.root else np_obj.text) 
        
        head, _, _, _ = head_info 
        if head.pos_ in ['NOUN', 'PROPN']: 
            ana_results = parse_morph(head.text) 
            normalized_head = ana_results[0].normal_form if ana_results else head.lemma_ 
        else:
            normalized_head = head.lemma_ 
        
        if head.pos_ not in ['NOUN', 'PROPN']: 
            return sanitize_string_for_xml(np_obj.lemma_) 
        
        tokens_lemmas = [sanitize_string_for_xml(token.lemma_) for token in np_obj] 
        if head.i - np_obj.start < len(tokens_lemmas): 
            tokens_lemmas[head.i - np_obj.start] = sanitize_string_for_xml(normalized_head) 
        return " ".join(tokens_lemmas) 
    
    return sanitize_string_for_xml(str(np_obj) if hasattr(np_obj, 'text') else str(np_obj)) 

def contains_only_russian_or_latin_letters(text): 
    return not re.search(r'[^\s\w.,!?;:\-_()ЁёА-яA-Za-z0-9]', text) 

def get_syntactic_relations(doc): 
    relations = [] 
    
    dependencies_to_keep = { 
        'ROOT', 'nsubj', 'nsubj:pass', 'obj', 'iobj', 'obl', 'obl:agent',
        'amod', 'advmod', 'nmod', 'acl', 'acl:relcl', 'advcl', 'ccomp', 'xcomp',
        'conj', 'cc', 'compound', 'parataxis', 'discourse', 'orphan'
    }
    
    chunks = {} 
    for ent in doc.ents: 
        head_info = head_in_named_entity(doc, ent) 
        if head_info: 
            head_token, morph_info, parent_token, dep_type = head_info 
            normalized_text = normalize_noun_phrase(doc, ent) 
            chunks[head_token.i] = { 
                'token': head_token,
                'text': normalized_text,
                'pos': head_token.pos_,
                'morph': [sanitize_string_for_xml(m) for m in morph_info], 
                'is_ent': True,
                'span': ent
            }
    
    for token in doc: 
        if token.i not in chunks: 
            if token.pos_ in ['NOUN', 'PROPN', 'VERB', 'ADJF', 'ADV', 'NUM', 'PRON']: 
                normalized_text = normalize_noun_phrase(doc, token) 
                chunks[token.i] = { 
                    'token': token,
                    'text': normalized_text,
                    'pos': sanitize_string_for_xml(token.pos_), 
                    'morph': [sanitize_string_for_xml(part.split('=')[1]) for part in str(token.morph).split('|') if '=' in part], 
                    'is_ent': False,
                    'span': None
                }
    
    resolved_pronouns = {} 
    for i in sorted(chunks.keys()): 
        chunk_data = chunks[i] 
        token = chunk_data['token'] 
        if token.pos_ == 'PRON' and token.i not in resolved_pronouns: 
            pronoun_gender = token.morph.get('Gender') 
            pronoun_number = token.morph.get('Number') 
            
            for j in sorted(chunks.keys(), reverse=True): 
                if j < i: 
                    antecedent_data = chunks[j] 
                    antecedent_token = antecedent_data['token'] 
                    if antecedent_token.pos_ in ['NOUN', 'PROPN']: 
                        antecedent_gender = antecedent_token.morph.get('Gender') 
                        antecedent_number = antecedent_token.morph.get('Number') 
                        
                        if (pronoun_gender is None or antecedent_gender is None or pronoun_gender == antecedent_gender) and \
                           (pronoun_number is None or antecedent_number is None or pronoun_number == antecedent_number): 
                            resolved_pronouns[token.i] = antecedent_data['text'] 
                            chunk_data['text'] = antecedent_data['text'] 
                            break 
    
    for token in doc: 
        head = token.head 
        
        if token.dep_ not in dependencies_to_keep: 
            continue 
        
        src_node_idx = head.i if head.i in chunks else head.head.i if head.head.i in chunks else -1 
        tgt_node_idx = token.i if token.i in chunks else token.head.i if token.head.i in chunks else -1 
        
        src_concept_data = chunks.get(src_node_idx) 
        tgt_concept_data = chunks.get(tgt_node_idx) 

        if src_concept_data and tgt_concept_data and src_concept_data['text'] and tgt_concept_data['text']: 
            src_concept = src_concept_data['text'] 
            tgt_concept = tgt_concept_data['text'] 
            
            edge_label = sanitize_string_for_xml(token.dep_) 
            edge_type = 'dependency' 
            
            if token.dep_ == 'conj': 
                if head.dep_ == 'nsubj' and token.dep_ == 'conj' and head.head == token.head: 
                    edge_label = sanitize_string_for_xml('conj_subj') 
                    edge_type = 'custom' 
            
            if src_concept and tgt_concept: 
                relations.append((src_concept, edge_label, tgt_concept, edge_type)) 

    return relations 

def get_syntactic_depth(doc): 
    if not doc or not list(doc.sents): return 0 
    max_depth_val = 0 
    for sent in doc.sents: 
        roots = [token for token in sent if token.dep_ == 'ROOT'] 
        if not roots: 
            roots = [sent.root] 
        for root in roots: 
            try:
                q = [(root, 1)] 
                visited = {root} 
                current_max = 0 
                while q: 
                    token, depth = q.pop(0) 
                    current_max = max(current_max, depth) 
                    for child in token.children: 
                        if child not in visited: 
                            visited.add(child) 
                            q.append((child, depth + 1)) 
                max_depth_val = max(max_depth_val, current_max) 
            except Exception as e:
                print(f"Error calculating syntactic depth for sentence starting at {sent.start_char}: {e}") 
                max_depth_val = max(max_depth_val, -1) 
    return max_depth_val 


def analyze_item(doc, item_id, item_type): 
    try:
        print(f"Starting analysis for {item_type} ID: {item_id}") 

        general_graph = GENERAL_GRAPH_GLOBAL 
        general_nodes = GENERAL_NODES_GLOBAL 
        communities_map = COMMUNITIES_MAP_GLOBAL 
        
        bg_degree = GLOBAL_BG_CENTRALITY_DEGREE 
        bg_betweenness = GLOBAL_BG_CENTRALITY_BETWEENNESS 
        bg_closeness = GLOBAL_BG_CENTRALITY_CLOSENESS 
        bg_eigenvector = GLOBAL_BG_CENTRALITY_EIGENVECTOR 
        global_edge_weight_percentiles = GLOBAL_EDGE_WEIGHT_PERCENTILES 
        global_hubs_nodes = GLOBAL_HUBS_NODES 

        if not doc or len(doc) == 0: 
            print(f"Item {item_id}: Doc is empty.") 
            return None 
        
        print(f"Analyzing {item_type} ID: {item_id}, Text: '{doc.text[:100]}...'") 

        item_relations_raw = get_syntactic_relations(doc) 
        if not item_relations_raw: 
            print(f"Item {item_id}: No relations extracted, skipping.") 
            return None 
        
        item_concepts = set() 
        item_relation_details = [] 
        relation_type_counts = {} 

        for rel_tuple in item_relations_raw: 
            if len(rel_tuple) == 4: 
                src_concept, edge_label, tgt_concept, edge_type = rel_tuple 
                
                item_concepts.add(src_concept) 
                item_concepts.add(tgt_concept) 
                item_relation_details.append((src_concept, edge_label, tgt_concept, edge_type)) 

                relation_type_counts[edge_label] = relation_type_counts.get(edge_label, 0) + 1 
            else:
                print(f"Item {item_id}: Skipping relation with unexpected format: {rel_tuple}") 

        if not item_concepts: 
            print(f"Item {item_id}: No concepts extracted, skipping.") 
            return None 

        valid_item_concepts = item_concepts.intersection(general_nodes) 
        
        stats = { 
            'item_id' : item_id, 
            'item_text': doc.text 
        }

        if not valid_item_concepts: 
            print(f"Item {item_id}: No concepts overlap with general graph, returning basic stats.") 
            stats.update({ 
                'subgraph_components': 0, 'subgraph_avg_degree': 0, 'subgraph_density': 0,
                'subgraph_diameter': -1, 'mean_betweenness_subgraph': 0, 'mean_closeness_subgraph': 0,
                'mean_clustering_coefficient_subgraph': 0, 'degree_assortativity_subgraph': 0,
                'mean_eigenvector_centrality_subgraph': 0, 'min_degree_subgraph': 0,
                'max_degree_subgraph': 0, 'median_degree_subgraph': 0, 'iqr_degree_subgraph': 0,
                'dependency_label_entropy': 0.0, 'ratio_core_modifying_relations': 0.0,
                'proportion_unique_relation_types': 0.0, 'mean_node_strength_subgraph': 0.0,
                'global_efficiency_subgraph': 0.0, 'local_efficiency_subgraph': 0.0,
                'num_bridges_subgraph': 0, 'ratio_bridges_subgraph': 0.0,
                'num_articulation_points_subgraph': 0, 'ratio_articulation_points_subgraph': 0.0,
                'modularity_subgraph': 0.0, 'num_inter_community_edges_subgraph': 0,
                'ratio_inter_community_edges_subgraph': 0.0, 'cycle_count_subgraph': 0,
                'cycle_density_subgraph': 0.0,
                'bridging_relation_ratio': 0.0, 'shortest_path_general_mean': 0.0,
                'shortest_path_general_median': 0.0, 'shortest_path_general_min': 0.0,
                'shortest_path_general_max': 0.0, 'shortest_path_general_iqr': 0.0,
                'mean_bridged_path_length': 0.0, 'max_bridged_path_length': 0.0,
                'joke_nodes_mean_bg_degree_centrality': 0.0, 'joke_nodes_mean_bg_betweenness_centrality': 0.0,
                'joke_nodes_mean_bg_closeness_centrality': 0.0, 'joke_nodes_mean_bg_eigenvector_centrality': 0.0,
                'community_count': 0, 'concept_contrast': 0.0,
                'semantic_heterogeneity_mean_dist': 0.0, 'semantic_heterogeneity_median_dist': 0.0,
                'semantic_heterogeneity_iqr_dist': 0.0, 'mean_edge_weight_in_bg_subgraph': 0.0,
                'proportion_rare_edges_in_bg_subgraph': 0.0, 'mean_centrality_bridging_concepts_degree': 0.0,
                'mean_centrality_bridging_concepts_betweenness': 0.0, 'mean_centrality_bridging_concepts_closeness': 0.0,
                'mean_centrality_bridging_concepts_eigenvector': 0.0, 'mean_semantic_specificity_bg': 0.0,
                'mean_dist_to_nearest_global_hub': 0.0
            })
            if KEY_DEP_RATIO_PAIRS: 
                for label1, label2 in KEY_DEP_RATIO_PAIRS: 
                    stats[f'ratio_{label1}_to_{label2}'] = 0.0 
            return stats 
        
        induced_subgraph = general_graph.subgraph(valid_item_concepts) 
        induced_undirected_view = induced_subgraph.to_undirected(as_view=True) 
        
        num_nodes_subgraph = induced_subgraph.number_of_nodes() 
        num_edges_subgraph = induced_subgraph.number_of_edges() 
        density_subgraph = nx.density(induced_subgraph) if num_nodes_subgraph > 1 else 0 
        num_components_subgraph = 0 
        diameter_subgraph = -1 
        avg_degree_subgraph = 0 
        min_degree_subgraph, max_degree_subgraph, median_degree_subgraph, iqr_degree_subgraph = 0, 0, 0, 0 

        if num_nodes_subgraph > 0: 
            try:
                if induced_undirected_view.number_of_nodes() > 0: 
                   num_components_subgraph = nx.number_connected_components(induced_undirected_view) 
                   if num_components_subgraph == 1 and induced_undirected_view.number_of_nodes() > 1: 
                       try:
                           diameter_subgraph = nx.diameter(induced_undirected_view) 
                       except nx.NetworkXNoPath: 
                           diameter_subgraph = 0 if induced_undirected_view.number_of_nodes() == 1 else -1 
                       except Exception as e: 
                           print(f"Item {item_id}: Unexpected error in subgraph diameter calculation: {e}") 
                           diameter_subgraph = -1 
                   elif induced_undirected_view.number_of_nodes() == 1: 
                       diameter_subgraph = 0 
                   else:
                       diameter_subgraph = -1 
                else:
                   num_components_subgraph = 0 
                   diameter_subgraph = -1 
            except Exception as e:
                print(f"Item {item_id}: Could not calculate components/diameter for subgraph: {e}") 
                num_components_subgraph = induced_undirected_view.number_of_nodes() if induced_undirected_view else 0 
                diameter_subgraph = -1 
            
            degrees = [d for n, d in induced_subgraph.degree()] 
            avg_degree_subgraph = sum(degrees) / num_nodes_subgraph if num_nodes_subgraph > 0 else 0 
            if degrees: 
                min_degree_subgraph = min(degrees) 
                max_degree_subgraph = max(degrees) 
                median_degree_subgraph = np.median(degrees) 
                iqr_degree_subgraph = calculate_iqr(degrees) 


        mean_betweenness_subgraph, mean_closeness_subgraph, mean_eigenvector_centrality_subgraph = 0, 0, 0 
        avg_clustering_coef_subgraph = 0 
        degree_assortativity_subgraph = 0 
        

        if num_nodes_subgraph > 1: 
            try:
                betweenness_subgraph = nx.betweenness_centrality(induced_subgraph, normalized=True, weight='weight') 
                mean_betweenness_subgraph = np.mean(list(betweenness_subgraph.values())) if betweenness_subgraph else 0 
            except Exception as e: print(f"Item {item_id}: Betweenness centrality on subgraph failed: {e}") 
            
            try:
                closeness_subgraph = nx.closeness_centrality(induced_subgraph, distance='weight') 
                mean_closeness_subgraph = np.mean(list(closeness_subgraph.values())) if closeness_subgraph else 0 
            except Exception as e: print(f"Item {item_id}: Closeness centrality on subgraph failed: {e}") 
            
            try:
                if induced_undirected_view.number_of_nodes() > 0: 
                    avg_clustering_coef_subgraph = nx.average_clustering(induced_undirected_view) 
            except Exception as e: print(f"Item {item_id}: Average clustering failed: {e}") 
            
            try:
                 if induced_undirected_view.number_of_edges() > 0: 
                     degrees_u = [d for n, d in induced_undirected_view.degree()] 
                     if len(set(degrees_u)) > 1: 
                         degree_assortativity_subgraph = nx.degree_assortativity_coefficient(induced_undirected_view) 
            except Exception as e: print(f"Item {item_id}: Degree assortativity failed: {e}") 
            
            try: 
                
                eigenvector_centrality_subgraph = None
                mean_eigenvector_centrality_subgraph = np.mean(list(eigenvector_centrality_subgraph.values())) if eigenvector_centrality_subgraph else 0 
            except Exception as e: print(f"Item {item_id}: Eigenvector centrality failed for subgraph: {e}") 
            
        mean_node_strength_subgraph = 0.0 
        if num_nodes_subgraph > 0: 
            try:
                if nx.is_weighted(induced_subgraph): 
                    strengths = [induced_subgraph.degree(node, weight='weight') for node in induced_subgraph.nodes()] 
                    mean_node_strength_subgraph = np.mean(strengths) if strengths else 0.0 
            except Exception as e: print(f"Item {item_id}: Mean node strength calculation failed: {e}") 

        global_efficiency_subgraph = 0.0 
        if num_nodes_subgraph > 1: 
            try:
                global_efficiency_subgraph = nx.global_efficiency(induced_undirected_view) 
            except Exception as e: print(f"Item {item_id}: Global efficiency calculation failed: {e}") 

        local_efficiency_subgraph = 0.0 
        
        if num_nodes_subgraph > 0: 

            try:
                local_efficiency_subgraph = nx.local_efficiency(induced_undirected_view) 
            except Exception as e: print(f"Item {item_id}: Local efficiency calculation failed: {e}") 


        num_bridges_subgraph = 0 
        ratio_bridges_subgraph = 0.0 
        num_articulation_points_subgraph = 0 
        ratio_articulation_points_subgraph = 0.0 
        if induced_undirected_view.number_of_edges() > 0: 
            try:
                bridges = list(nx.bridges(induced_undirected_view)) 
                num_bridges_subgraph = len(bridges) 
                ratio_bridges_subgraph = num_bridges_subgraph / induced_undirected_view.number_of_edges() 
            except Exception as e: print(f"Item {item_id}: Bridges calculation failed: {e}") 
        
        if induced_undirected_view.number_of_nodes() > 0 : 
            try:
                articulation_points = list(nx.articulation_points(induced_undirected_view)) 
                num_articulation_points_subgraph = len(articulation_points) 
                ratio_articulation_points_subgraph = num_articulation_points_subgraph / induced_undirected_view.number_of_nodes() if induced_undirected_view.number_of_nodes() > 0 else 0.0 
            except Exception as e: print(f"Item {item_id}: Articulation points calculation failed: {e}") 


        modularity_subgraph = 0.0 
        num_inter_community_edges_subgraph = 0 
        ratio_inter_community_edges_subgraph = 0.0 
        if induced_undirected_view.number_of_nodes() > 1 and induced_undirected_view.number_of_edges() > 0: 
            try:
                partition = best_partition(induced_undirected_view) 
                communities_for_modularity = defaultdict(set) 
                for node, comm_id in partition.items(): 
                    communities_for_modularity[comm_id].add(node) 
                
                if communities_for_modularity and len(communities_for_modularity) > 1:  
                    modularity_subgraph = nx.community.modularity(induced_undirected_view, communities_for_modularity.values()) 
                    
                    inter_community_edges = 0 
                    for u, v in induced_undirected_view.edges(): 
                        if partition.get(u) != partition.get(v): 
                            inter_community_edges += 1 
                    num_inter_community_edges_subgraph = inter_community_edges 
                    ratio_inter_community_edges_subgraph = inter_community_edges / induced_undirected_view.number_of_edges() if induced_undirected_view.number_of_edges() > 0 else 0.0 
            except Exception as e: print(f"Item {item_id}: Modularity/Inter-community edges calculation failed: {e}") 

        cycle_count_subgraph = 0 
        cycle_density_subgraph = 0.0 
        if induced_undirected_view.number_of_nodes() > 2: 
            try:
                cycles = list(nx.cycle_basis(induced_undirected_view)) 
                cycle_count_subgraph = len(cycles) 
                if induced_undirected_view.number_of_edges() > 0: 
                    cycle_density_subgraph = cycle_count_subgraph / induced_undirected_view.number_of_edges() 
            except Exception as e: print(f"Item {item_id}: Cycle calculation failed: {e}") 


        shortest_path_lengths_s_t = [] 
        bridging_relation_count = 0 
        bridged_path_lengths = [] 
        
        undirected_general_graph_view_in_analyze_item = general_graph.to_undirected(as_view=True) 

        unique_src_concepts_for_path = set() 
        for src, _, tgt, _ in item_relation_details: 
            if src in undirected_general_graph_view_in_analyze_item and tgt in undirected_general_graph_view_in_analyze_item: 
                unique_src_concepts_for_path.add(src) 
        
        all_sssp_lengths = {} 
        if unique_src_concepts_for_path and undirected_general_graph_view_in_analyze_item.number_of_nodes() > 0: 
            try:
                for s_node in unique_src_concepts_for_path: 
                    if undirected_general_graph_view_in_analyze_item.has_node(s_node): 
                         all_sssp_lengths[s_node] = nx.single_source_shortest_path_length(undirected_general_graph_view_in_analyze_item, s_node) 
            except Exception as e:
                print(f"Item {item_id}: SSSP calculation failed for sources: {e}") 
                all_sssp_lengths = {} 

        bridging_concepts = set() 
        for src, edge_label, tgt, edge_type in item_relation_details: 
            if src in general_nodes and tgt in general_nodes: 
                if not general_graph.has_edge(src, tgt): 
                    bridging_relation_count += 1 
                    bridging_concepts.add(src) 
                    bridging_concepts.add(tgt) 
                    try:
                        if undirected_general_graph_view_in_analyze_item.has_node(src) and undirected_general_graph_view_in_analyze_item.has_node(tgt): 
                            path_len = all_sssp_lengths.get(src, {}).get(tgt) 
                            if path_len is not None: 
                                if path_len > 1: 
                                    bridged_path_lengths.append(path_len) 
                    except Exception as e:
                        print(f"Item {item_id}: Error getting shortest path for bridging relation ({src} -> {tgt}): {e}") 
                
                
                if src in all_sssp_lengths and tgt in all_sssp_lengths[src]: 
                    path_len = all_sssp_lengths[src][tgt] 
                    shortest_path_lengths_s_t.append(path_len) 

        mean_bridged_path_length = np.mean(bridged_path_lengths) if bridged_path_lengths else 0.0 
        max_bridged_path_length = np.max(bridged_path_lengths) if bridged_path_lengths else 0.0 


        path_stats = { 
            'mean': np.mean(shortest_path_lengths_s_t) if shortest_path_lengths_s_t else 0, 
            'median': np.median(shortest_path_lengths_s_t) if shortest_path_lengths_s_t else 0, 
            'min': min(shortest_path_lengths_s_t) if shortest_path_lengths_s_t else 0, 
            'max': max(shortest_path_lengths_s_t) if shortest_path_lengths_s_t else 0, 
            'iqr': calculate_iqr(shortest_path_lengths_s_t) 
        }


        item_communities = set(communities_map.get(c, -1) for c in valid_item_concepts if c in communities_map) 
        num_item_communities = len(item_communities) - (1 if -1 in item_communities else 0) 
        concept_contrast = num_item_communities / len(valid_item_concepts) if valid_item_concepts else 0 


        joke_nodes_mean_bg_degree_centrality = 0 
        joke_nodes_mean_bg_betweenness_centrality = 0 
        joke_nodes_mean_bg_closeness_centrality = 0 
        joke_nodes_mean_bg_eigenvector_centrality = 0 

        if valid_item_concepts: 
            joke_nodes_mean_bg_degree_centrality = np.mean([bg_degree.get(node, 0) for node in valid_item_concepts]) 
            joke_nodes_mean_bg_betweenness_centrality = np.mean([bg_betweenness.get(node, 0) for node in valid_item_concepts]) 
            joke_nodes_mean_bg_closeness_centrality = np.mean([bg_closeness.get(node, 0) for node in valid_item_concepts]) 
            joke_nodes_mean_bg_eigenvector_centrality = np.mean([bg_eigenvector.get(node, 0) for node in valid_item_concepts]) 


        semantic_heterogeneity_dists = [] 
        if len(valid_item_concepts) > 1: 
            nodes_list = list(valid_item_concepts) 
            for i in range(len(nodes_list)): 
                for j in range(i + 1, len(nodes_list)): 
                    n1, n2 = nodes_list[i], nodes_list[j] 
                    if undirected_general_graph_view_in_analyze_item.has_node(n1) and undirected_general_graph_view_in_analyze_item.has_node(n2): 
                        try:
                            dist = nx.shortest_path_length(undirected_general_graph_view_in_analyze_item, n1, n2) 
                            semantic_heterogeneity_dists.append(dist) 
                        except nx.NetworkXNoPath: 
                            semantic_heterogeneity_dists.append(float('inf')) 
                        except Exception as e: 
                            print(f"Item {item_id}: Error calculating semantic heterogeneity for {n1}-{n2}: {e}") 
        
        finite_semantic_dists = [d for d in semantic_heterogeneity_dists if d != float('inf')] 
        semantic_heterogeneity_mean_dist = np.mean(finite_semantic_dists) if finite_semantic_dists else 0.0 
        semantic_heterogeneity_median_dist = np.median(finite_semantic_dists) if finite_semantic_dists else 0.0 
        semantic_heterogeneity_iqr_dist = calculate_iqr(finite_semantic_dists) 


        mean_edge_weight_in_bg_subgraph = 0.0 
        proportion_rare_edges_in_bg_subgraph = 0.0 
        subgraph_edge_weights = [] 
        if induced_subgraph.number_of_edges() > 0: 
            for u, v, data in induced_subgraph.edges(data=True): 
                weight = data.get('weight', 1.0) 
                subgraph_edge_weights.append(weight) 
            if subgraph_edge_weights: 
                mean_edge_weight_in_bg_subgraph = np.mean(subgraph_edge_weights) 
                if 'p25' in global_edge_weight_percentiles: 
                    rare_threshold = global_edge_weight_percentiles['p25'] 
                    rare_edges_count = sum(1 for w in subgraph_edge_weights if w <= rare_threshold) 
                    proportion_rare_edges_in_bg_subgraph = rare_edges_count / len(subgraph_edge_weights) 


        mean_centrality_bridging_concepts_degree = 0.0 
        mean_centrality_bridging_concepts_betweenness = 0.0 
        mean_centrality_bridging_concepts_closeness = 0.0 
        mean_centrality_bridging_concepts_eigenvector = 0.0 

        if bridging_concepts: 
            mean_centrality_bridging_concepts_degree = np.mean([bg_degree.get(node, 0) for node in bridging_concepts]) 
            mean_centrality_bridging_concepts_betweenness = np.mean([bg_betweenness.get(node, 0) for node in bridging_concepts]) 
            mean_centrality_bridging_concepts_closeness = np.mean([bg_closeness.get(node, 0) for node in bridging_concepts]) 
            mean_centrality_bridging_concepts_eigenvector = np.mean([bg_eigenvector.get(node, 0) for node in bridging_concepts]) 


        mean_semantic_specificity_bg = 0.0 
        if valid_item_concepts: 
            specificity_scores = [1.0 / (bg_degree.get(node, 0) + 1e-9) for node in valid_item_concepts] 
            mean_semantic_specificity_bg = np.mean(specificity_scores) if specificity_scores else 0.0 


        mean_dist_to_nearest_global_hub = 0.0 
        if valid_item_concepts and global_hubs_nodes and undirected_general_graph_view_in_analyze_item.number_of_nodes() > 0: 
            dists_to_hubs = [] 
            for item_node in valid_item_concepts: 
                if undirected_general_graph_view_in_analyze_item.has_node(item_node): 
                    min_dist_to_hub = float('inf') 
                    for hub_node in global_hubs_nodes: 
                        if undirected_general_graph_view_in_analyze_item.has_node(hub_node): 
                            try:
                                dist = nx.shortest_path_length(undirected_general_graph_view_in_analyze_item, item_node, hub_node) 
                                min_dist_to_hub = min(min_dist_to_hub, dist) 
                            except nx.NetworkXNoPath: 
                                pass 
                            except Exception as e: 
                                print(f"Item {item_id}: Error calculating dist to hub for {item_node}-{hub_node}: {e}") 
                    if min_dist_to_hub != float('inf'): 
                        dists_to_hubs.append(min_dist_to_hub) 
            mean_dist_to_nearest_global_hub = np.mean(dists_to_hubs) if dists_to_hubs else 0.0 


        dep_labels_in_item = [r[1] for r in item_relation_details] 
        label_counts = Counter(dep_labels_in_item) 
        total_labels_in_item = len(dep_labels_in_item) 

        dependency_label_entropy = 0.0 
        if total_labels_in_item > 0: 
            probabilities = [count / total_labels_in_item for count in label_counts.values()] 
            dependency_label_entropy = entropy(probabilities, base=2) 

        CORE_DEPS = {'nsubj', 'obj', 'iobj', 'ccomp', 'xcomp', 'csubj', 'aux', 'aux:pass', 'expl'} 
        MODIFYING_DEPS = {'amod', 'nmod', 'advmod', 'acl', 'acl:relcl', 'advcl', 'cc', 'compound', 'parataxis', 'discourse', 'orphan', 'appos', 'det', 'case', 'punct', 'mark', 'flat', 'fixed', 'goeswith', 'list', 'reparandum', 'vocative', 'nummod'} 

        core_count = sum(label_counts.get(dep, 0) for dep in CORE_DEPS) 
        modifying_count = sum(label_counts.get(dep, 0) for dep in MODIFYING_DEPS) 

        ratio_core_modifying_relations = 0.0 
        if modifying_count > 0: 
            ratio_core_modifying_relations = core_count / modifying_count 
        elif core_count > 0: 
            ratio_core_modifying_relations = 1000.0 
        
        if KEY_DEP_RATIO_PAIRS: 
            for label1, label2 in KEY_DEP_RATIO_PAIRS: 
                count1 = label_counts.get(label1, 0) 
                count2 = label_counts.get(label2, 0) 
                if count2 > 0: 
                    stats[f'ratio_{label1}_to_{label2}'] = count1 / count2 
                elif count1 > 0: 
                    stats[f'ratio_{label1}_to_{label2}'] = 1000.0 
                else: 
                    stats[f'ratio_{label1}_to_{label2}'] = 0.0 

        proportion_unique_relation_types = 0.0 
        if total_labels_in_item > 0: 
            proportion_unique_relation_types = len(label_counts) / total_labels_in_item 


        bridging_relation_ratio = bridging_relation_count / len(item_relation_details) if len(item_relation_details) > 0 else 0 


        id_key_name = 'item_id' 
        if item_type == 'joke': 
            id_key_name = 'joke_id' 
        elif item_type == 'sentence': 
            id_key_name = 'sentence_id' 
        
        final_stats_dict = {id_key_name: item_id, 'item_text': doc.text}


        final_stats_dict.update({ 
            
            'subgraph_components': num_components_subgraph,
            'subgraph_avg_degree': avg_degree_subgraph,
            'subgraph_density': density_subgraph,
            'subgraph_diameter': diameter_subgraph,
            'mean_betweenness_subgraph': mean_betweenness_subgraph,
            'mean_closeness_subgraph': mean_closeness_subgraph,
            'mean_clustering_coefficient_subgraph': avg_clustering_coef_subgraph,
            'degree_assortativity_subgraph': degree_assortativity_subgraph,
            'mean_eigenvector_centrality_subgraph': mean_eigenvector_centrality_subgraph,
            'min_degree_subgraph': min_degree_subgraph,
            'max_degree_subgraph': max_degree_subgraph,
            'median_degree_subgraph': median_degree_subgraph, 
            'iqr_degree_subgraph': iqr_degree_subgraph,
            'mean_node_strength_subgraph': mean_node_strength_subgraph,
            'global_efficiency_subgraph': global_efficiency_subgraph,
            'local_efficiency_subgraph': local_efficiency_subgraph,
            'num_bridges_subgraph': num_bridges_subgraph,
            'ratio_bridges_subgraph': ratio_bridges_subgraph,
            'num_articulation_points_subgraph': num_articulation_points_subgraph,
            'ratio_articulation_points_subgraph': ratio_articulation_points_subgraph,
            'modularity_subgraph': modularity_subgraph,
            'num_inter_community_edges_subgraph': num_inter_community_edges_subgraph,
            'ratio_inter_community_edges_subgraph': ratio_inter_community_edges_subgraph,
            'cycle_count_subgraph': cycle_count_subgraph,
            'cycle_density_subgraph': cycle_density_subgraph,

            'bridging_relation_ratio': bridging_relation_ratio,
            'shortest_path_general_mean': path_stats['mean'],
            'shortest_path_general_median': path_stats['median'],
            'shortest_path_general_min': path_stats['min'],
            'shortest_path_general_max': path_stats['max'],
            'shortest_path_general_iqr': path_stats['iqr'],
            'mean_bridged_path_length': mean_bridged_path_length,
            'max_bridged_path_length': max_bridged_path_length,
            
            'joke_nodes_mean_bg_degree_centrality': joke_nodes_mean_bg_degree_centrality,
            'joke_nodes_mean_bg_betweenness_centrality': joke_nodes_mean_bg_betweenness_centrality, 
            'joke_nodes_mean_bg_closeness_centrality': joke_nodes_mean_bg_closeness_centrality,
            'joke_nodes_mean_bg_eigenvector_centrality': joke_nodes_mean_bg_eigenvector_centrality,
            'community_count': num_item_communities,
            'concept_contrast': concept_contrast,

            'semantic_heterogeneity_mean_dist': semantic_heterogeneity_mean_dist,
            'semantic_heterogeneity_median_dist': semantic_heterogeneity_median_dist,
            'semantic_heterogeneity_iqr_dist': semantic_heterogeneity_iqr_dist,
            'mean_edge_weight_in_bg_subgraph': mean_edge_weight_in_bg_subgraph,
            'proportion_rare_edges_in_bg_subgraph': proportion_rare_edges_in_bg_subgraph,
            'mean_centrality_bridging_concepts_degree': mean_centrality_bridging_concepts_degree,
            'mean_centrality_bridging_concepts_betweenness': mean_centrality_bridging_concepts_betweenness,
            'mean_centrality_bridging_concepts_closeness': mean_centrality_bridging_concepts_closeness,
            'mean_centrality_bridging_concepts_eigenvector': mean_centrality_bridging_concepts_eigenvector,
            'mean_semantic_specificity_bg': mean_semantic_specificity_bg,
            'mean_dist_to_nearest_global_hub': mean_dist_to_nearest_global_hub,

            'dependency_label_entropy': dependency_label_entropy,
            'ratio_core_modifying_relations': ratio_core_modifying_relations,
            'proportion_unique_relation_types': proportion_unique_relation_types
        })
        
        if KEY_DEP_RATIO_PAIRS: 
             for label1, label2 in KEY_DEP_RATIO_PAIRS: 
                final_stats_dict[f'ratio_{label1}_to_{label2}'] = stats.get(f'ratio_{label1}_to_{label2}', 0.0) 

        return final_stats_dict 

    except Exception as e:
        print(f"CRITICAL ERROR within analyze_item for item {item_id}: {e}") 
        return None 




print(f"Starting chunk processing for Task ID: {TASK_ID} / {TOTAL_CHUNKS}")


anek_lines_analysis = [] 
start_idx_aneks_global = -1 
if os.path.exists(ANEK_FILE): 
    try:
        with open(ANEK_FILE, 'r', encoding='utf-8') as f: 
            all_anek_lines_full = [line.strip() for line in f if line.strip()] 
        
        lines_to_read_total_aneks = int(len(all_anek_lines_full) * ITEM_CORPUS_FRACTION) 
        all_anek_lines_fractioned = all_anek_lines_full[:lines_to_read_total_aneks] 
        
        num_total_fractioned_aneks = len(all_anek_lines_fractioned) 
        lines_per_chunk = (num_total_fractioned_aneks + TOTAL_CHUNKS - 1) // TOTAL_CHUNKS 
        
        start_idx_aneks_global = TASK_ID * lines_per_chunk 
        end_idx_aneks_chunk = min(start_idx_aneks_global + lines_per_chunk, num_total_fractioned_aneks) 
        
        anek_lines_analysis = all_anek_lines_fractioned[start_idx_aneks_global:end_idx_aneks_chunk] 
        print(f"Task {TASK_ID} [ANEKS]: Reading lines {start_idx_aneks_global}-{end_idx_aneks_chunk-1} from fractioned corpus ({num_total_fractioned_aneks} total). Got {len(anek_lines_analysis)} aneks.")
    except Exception as e:
        print(f"Error reading {ANEK_FILE}: {e}") 
else:
    print(f"{ANEK_FILE} not found.") 

sentence_corpus_lines_analysis = [] 
start_idx_sentences_global = -1
if os.path.exists(NOT_ANEKS_FILE): 
    try:
        with open(NOT_ANEKS_FILE, 'r', encoding='utf-8') as f: 
            all_sentence_lines_full = [line.strip() for line in f if line.strip()] 

        lines_to_read_total_sentences = int(len(all_sentence_lines_full) * ITEM_CORPUS_FRACTION) 
        all_sentence_lines_fractioned = all_sentence_lines_full[:lines_to_read_total_sentences] 

        num_total_fractioned_sentences = len(all_sentence_lines_fractioned) 
        lines_per_chunk_s = (num_total_fractioned_sentences + TOTAL_CHUNKS - 1) // TOTAL_CHUNKS 
        
        start_idx_sentences_global = TASK_ID * lines_per_chunk_s 
        end_idx_sentences_chunk = min(start_idx_sentences_global + lines_per_chunk_s, num_total_fractioned_sentences) 

        sentence_corpus_lines_analysis = all_sentence_lines_fractioned[start_idx_sentences_global:end_idx_sentences_chunk] 
        print(f"Task {TASK_ID} [SENTENCES]: Reading lines {start_idx_sentences_global}-{end_idx_sentences_chunk-1} from fractioned corpus ({num_total_fractioned_sentences} total). Got {len(sentence_corpus_lines_analysis)} sentences.")
    except Exception as e:
        print(f"Error reading {NOT_ANEKS_FILE}: {e}") 
else:
    print(f"{NOT_ANEKS_FILE} not found.") 

all_items_texts_for_pipe = [] 
all_items_ids = [] 
all_items_types = [] 

if anek_lines_analysis: 
    all_items_texts_for_pipe.extend(anek_lines_analysis) 
    all_items_ids.extend([f"joke_{start_idx_aneks_global + i}" for i in range(len(anek_lines_analysis))]) 
    all_items_types.extend(['joke'] * len(anek_lines_analysis)) 
if sentence_corpus_lines_analysis: 
    all_items_texts_for_pipe.extend(sentence_corpus_lines_analysis) 
    all_items_ids.extend([f"sentence_{start_idx_sentences_global + i}" for i in range(len(sentence_corpus_lines_analysis))]) 
    all_items_types.extend(['sentence'] * len(sentence_corpus_lines_analysis)) 

if not all_items_texts_for_pipe: 
    print(f"Task {TASK_ID}: No items to process for this chunk. Exiting gracefully.") 
    pd.DataFrame().to_csv(JOKE_STATS_FILENAME_CHUNK)
    pd.DataFrame().to_csv(SENTENCE_STATS_FILENAME_CHUNK)
    exit() 

print(f"Task {TASK_ID}: Starting SpaCy processing of {len(all_items_texts_for_pipe)} items (n_process={GLOBAL_WORKERS}).") 
processed_docs = [] 
try:
    pipe_n_process_chunk = GLOBAL_WORKERS 
    
    
    
    processed_docs_iterator = nlp.pipe(all_items_texts_for_pipe, n_process=pipe_n_process_chunk, batch_size=10) 
    for i, doc in enumerate(processed_docs_iterator): 
        processed_docs.append(doc) 
        if (i + 1) % 1000 == 0 or (i + 1) == len(all_items_texts_for_pipe): 
            print(f"Task {TASK_ID}: SpaCy processed {i+1}/{len(all_items_texts_for_pipe)} items.") 
except Exception as e:
    print(f"Task {TASK_ID}: SpaCy pipe processing error: {e}") 
    exit() 

print(f"Task {TASK_ID}: Starting analysis of {len(processed_docs)} items using joblib (n_jobs={GLOBAL_WORKERS}).") 
analysis_results = [] 
item_args_list = [] 
for i, doc in enumerate(processed_docs): 
    item_args_list.append((doc, all_items_ids[i], all_items_types[i])) 

results = Parallel(n_jobs=GLOBAL_WORKERS)( 
    delayed(analyze_item)(doc, item_id, item_type) 
    for doc, item_id, item_type in item_args_list 
)
analysis_results = [r for r in results if r is not None] 
print(f"Task {TASK_ID}: Analysis complete. Got {len(analysis_results)} valid results.") 


joke_statistics_list = [r for r in analysis_results if 'joke_id' in r] 
sentence_statistics_list = [r for r in analysis_results if 'sentence_id' in r] 

if joke_statistics_list: 
    stats_df_jokes_chunk = pd.DataFrame(joke_statistics_list) 
    if 'joke_id' in stats_df_jokes_chunk.columns: 
        stats_df_jokes_chunk = stats_df_jokes_chunk.set_index('joke_id') 
        stats_df_jokes_chunk.to_csv(JOKE_STATS_FILENAME_CHUNK) 
        print(f"Task {TASK_ID}: Joke statistics for chunk saved to '{JOKE_STATS_FILENAME_CHUNK}' ({len(stats_df_jokes_chunk)} rows)") 
    else:
        print(f"Task {TASK_ID}: 'joke_id' not in joke stats. Saving empty.") 
        pd.DataFrame().to_csv(JOKE_STATS_FILENAME_CHUNK)
else:
    print(f"Task {TASK_ID}: No joke statistics for this chunk. Saving empty.") 
    pd.DataFrame().to_csv(JOKE_STATS_FILENAME_CHUNK)

if sentence_statistics_list: 
    stats_df_sentences_chunk = pd.DataFrame(sentence_statistics_list) 
    if 'sentence_id' in stats_df_sentences_chunk.columns: 
        stats_df_sentences_chunk = stats_df_sentences_chunk.set_index('sentence_id') 
        stats_df_sentences_chunk.to_csv(SENTENCE_STATS_FILENAME_CHUNK) 
        print(f"Task {TASK_ID}: Sentence statistics for chunk saved to '{SENTENCE_STATS_FILENAME_CHUNK}' ({len(stats_df_sentences_chunk)} rows)") 
    else:
        print(f"Task {TASK_ID}: 'sentence_id' not in sentence stats. Saving empty.") 
        pd.DataFrame().to_csv(SENTENCE_STATS_FILENAME_CHUNK)
else:
    print(f"Task {TASK_ID}: No sentence statistics for this chunk. Saving empty.") 
    pd.DataFrame().to_csv(SENTENCE_STATS_FILENAME_CHUNK)

print(f"Task {TASK_ID}: Chunk processing finished successfully.")