import logging
import os
import time
import pickle
import networkx as nx
from community import best_partition
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from utils import build_graph_from_relations, sanitize_string_for_xml

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

OUTPUT_DIR_GRAPH = 'graph_data'
OUTPUT_DIR_LONG_RELATIONS = os.path.join(OUTPUT_DIR_GRAPH, 'long_relations')
LONG_RELATIONS_FILE = os.path.join(OUTPUT_DIR_LONG_RELATIONS, 'long_corpus_relations.pkl')
OUTPUT_DIR_SHORT_RELATIONS = os.path.join(OUTPUT_DIR_GRAPH, 'short_relations')
SHORT_RELATIONS_FILE = os.path.join(OUTPUT_DIR_SHORT_RELATIONS, 'short_corpus_relations.pkl')

GRAPH_OUTPUT_FILE = os.path.join('graph_data_X', 'general_graph.pkl')
NODES_OUTPUT_FILE = os.path.join('graph_data_X', 'general_nodes.pkl')
EDGES_OUTPUT_FILE = os.path.join('graph_data_X', 'general_edges_set.pkl')

def calculate_degree_centrality(graph):
    logging.info("Calculating global degree centrality in parallel process...")
    try:
        return nx.degree_centrality(graph)
    except Exception as e:
        logging.error(f"Failed to calculate global degree centrality in parallel: {e}", exc_info=True)
        return {}

def calculate_betweenness_centrality(graph):
    logging.info("Calculating global betweenness centrality in parallel process...")
    try:
        if nx.get_edge_attributes(graph, 'weight'):
            return nx.betweenness_centrality(graph, weight='weight', normalized=True)
        else:
            return nx.betweenness_centrality(graph, normalized=True)
    except Exception as e:
        logging.error(f"Failed to calculate global betweenness centrality in parallel: {e}", exc_info=True)
        return {}

def calculate_closeness_centrality(graph):
    logging.info("Calculating global closeness centrality in parallel process...")
    try:
        if nx.get_edge_attributes(graph, 'weight'):
            return nx.closeness_centrality(graph, distance='weight')
        else:
            return nx.closeness_centrality(graph)
    except Exception as e:
        logging.error(f"Failed to calculate global closeness centrality in parallel: {e}", exc_info=True)
        return {}

def calculate_eigenvector_centrality(graph):
    logging.info("Calculating global eigenvector centrality in parallel process...")
    if not nx.is_connected(graph):
         logging.warning("Graph is disconnected, skipping eigenvector centrality in parallel process.")
         return {}
    try:
        if nx.get_edge_attributes(graph, 'weight'):
            return nx.eigenvector_centrality(graph, weight='weight', max_iter=2000, tol=1e-06)
        else:
            return nx.eigenvector_centrality(graph, max_iter=2000, tol=1e-06)
    except Exception as e:
        logging.error(f"Failed to calculate global eigenvector centrality in parallel: {e}", exc_info=True)
        return {}

def save_pickle_task(data, filename):
    try:
        with open(filename, 'wb') as f:
            pickle.dump(data, f)
        logging.info(f"Data saved to {filename}")
        return True, filename
    except Exception as e:
        logging.error(f"Failed to save data to {filename}: {e}", exc_info=True)
        return False, filename

def save_graphml_task(graph, filename):
    try:
        nx.write_graphml(graph, filename)
        logging.info(f"GraphML saved to {filename}")
        return True, filename
    except Exception as e:
        logging.error(f"Failed to save GraphML to {filename}: {e}", exc_info=True)
        return False, filename

def main():
    os.makedirs(OUTPUT_DIR_GRAPH, exist_ok=True)
    os.makedirs(OUTPUT_DIR_LONG_RELATIONS, exist_ok=True)
    os.makedirs(OUTPUT_DIR_SHORT_RELATIONS, exist_ok=True)

    GLOBAL_BG_CENTRALITY_DEGREE = {}
    GLOBAL_BG_CENTRALITY_BETWEENNESS = {}
    GLOBAL_BG_CENTRALITY_CLOSENESS = {}
    GLOBAL_BG_CENTRALITY_EIGENVECTOR = {}
    COMMUNITIES_MAP_GLOBAL = {}

    all_relations_list = []

    if os.path.exists(LONG_RELATIONS_FILE):
        logging.info(f"Loading long corpus relations from {LONG_RELATIONS_FILE}...")
        try:
            with open(LONG_RELATIONS_FILE, 'rb') as f:
                long_relations = pickle.load(f)
                all_relations_list.extend(long_relations)
                logging.info(f"Loaded {len(long_relations)} relations from long corpus.")
        except Exception as e:
            logging.error(f"Error loading long corpus relations from {LONG_RELATIONS_FILE}: {e}", exc_info=True)
    else:
        logging.warning(f"Long corpus relations file not found: {LONG_RELATIONS_FILE}. Skipping.")

    if os.path.exists(SHORT_RELATIONS_FILE):
        logging.info(f"Loading short corpus relations from {SHORT_RELATIONS_FILE}...")
        try:
            with open(SHORT_RELATIONS_FILE, 'rb') as f:
                short_relations = pickle.load(f)
                all_relations_list.extend(short_relations)
                logging.info(f"Loaded {len(short_relations)} relations from short corpus.")
        except Exception as e:
            logging.error(f"Error loading short corpus relations from {SHORT_RELATIONS_FILE}: {e}", exc_info=True)
    else:
        logging.warning(f"Short corpus relations file not found: {SHORT_RELATIONS_FILE}. Skipping.")

    if not all_relations_list:
        logging.error("No relations loaded from either corpus. Cannot build graph. Exiting.")
        exit()

    logging.info(f"Total relations loaded: {len(all_relations_list)}. Starting graph construction...")

    try:
        general_graph, general_nodes_initial, general_edges_set_initial = build_graph_from_relations(all_relations_list)
        logging.info(f"Graph built: {general_graph.number_of_nodes()} nodes, {general_graph.number_of_edges()} edges.")
        logging.info(f"Initial graph built: {general_graph.number_of_nodes()} nodes, {general_graph.number_of_edges()} edges.")

        logging.info("Removing self-loops from the general graph...")
        self_loops = list(nx.selfloop_edges(general_graph))
        general_graph.remove_edges_from(self_loops)
        logging.info(f"Removed {len(self_loops)} self-loops.")

        general_nodes = set(general_graph.nodes())
        general_edges_set = set((u, v, data.get('label'), data.get('type')) for u, v, data in general_graph.edges(data=True))

        logging.info("Starting graph filtering...")
        initial_nodes_after_loops = general_graph.number_of_nodes()
        initial_edges_after_loops = general_graph.number_of_edges()
        logging.info(f"Graph after self-loop removal: {initial_nodes_after_loops} nodes, {initial_edges_after_loops} edges.")
        '''
        edges_to_remove = [(u, v) for u, v, data in general_graph.edges(data=True) if data.get('weight', 0) < 10]
        general_graph.remove_edges_from(edges_to_remove)
        logging.info(f"Removed {len(edges_to_remove)} edges with weight < 10.")

        nodes_to_remove = [node for node in list(general_graph.nodes()) if general_graph.degree(node) <= 0]
        general_graph.remove_nodes_from(nodes_to_remove)
        logging.info(f"Removed {len(nodes_to_remove)} nodes with degree <= 1.")
        '''
        nodes_after_degree_filter = general_graph.number_of_nodes()
        edges_after_degree_filter = general_graph.number_of_edges()
        logging.info(f"Graph after weight and degree filtering: {nodes_after_degree_filter} nodes, {edges_after_degree_filter} edges.")

        logging.info("Keeping only the largest connected component...")
        if general_graph.number_of_nodes() > 0:
            try:
                undirected_graph_view = general_graph.to_undirected(as_view=True)
                components = list(nx.connected_components(undirected_graph_view))
                logging.info(f"Found {len(components)} connected components.")
                if components:
                    largest_component_nodes = max(components, key=len)
                    logging.info(f"Size of the largest component: {len(largest_component_nodes)}")
                    general_graph = general_graph.subgraph(largest_component_nodes).copy()
                    logging.info(f"Graph reduced to the largest component: {general_graph.number_of_nodes()} nodes, {general_graph.number_of_edges()} edges.")
                else:
                    logging.warning("No connected components found in the graph after filtering.")
                    general_graph = nx.DiGraph()
            except Exception as e:
                logging.error(f"Error while finding or keeping largest connected component: {e}", exc_info=True)
                general_graph = nx.DiGraph()
        else:
            logging.warning("Graph is empty after filtering, cannot find connected components.")

        logging.info(f"Final filtered graph: {general_graph.number_of_nodes()} nodes, {general_graph.number_of_edges()} edges.")

        logging.info("Calculating community structure...")
        if best_partition is None:
            logging.warning("Community detection skipped because 'python-louvain' is not installed.")
        elif general_graph.number_of_nodes() > 1:
            try:
                undirected_graph_for_community = general_graph.to_undirected(as_view=True)
                if undirected_graph_for_community.number_of_nodes() > 1 :
                    logging.info("Starting community detection (best_partition)...")
                    start_comm_time = time.time()
                    if nx.get_edge_attributes(undirected_graph_for_community, 'weight'):
                        COMMUNITIES_MAP_GLOBAL = best_partition(undirected_graph_for_community, weight='weight')
                    else:
                        COMMUNITIES_MAP_GLOBAL = best_partition(undirected_graph_for_community)
                    end_comm_time = time.time()
                    logging.info(f"Community detection complete ({end_comm_time - start_comm_time:.2f}s). Found {len(set(COMMUNITIES_MAP_GLOBAL.values()))} communities.")
                else:
                    logging.warning("Graph not suitable for community detection (e.g. disconnected or too small). Skipping.")
            except Exception as e:
                logging.error(f"Community detection failed: {e}", exc_info=True)
        else:
            logging.warning("Graph has insufficient nodes after filtering, skipping community detection.")

        logging.info("Calculating global graph centrality metrics (in parallel)...")
        undirected_graph_global_view = general_graph.to_undirected(as_view=True)

        if undirected_graph_global_view.number_of_nodes() > 1:
            max_workers = os.cpu_count()
            logging.info(f"Using up to {max_workers} workers for centrality calculations.")
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                futures = {}
                futures[executor.submit(calculate_degree_centrality, undirected_graph_global_view)] = "degree"
                futures[executor.submit(calculate_betweenness_centrality, undirected_graph_global_view)] = "betweenness"
                futures[executor.submit(calculate_closeness_centrality, undirected_graph_global_view)] = "closeness"

                if nx.is_connected(undirected_graph_global_view):
                     futures[executor.submit(calculate_eigenvector_centrality, undirected_graph_global_view)] = "eigenvector"
                else:
                    logging.warning("Global graph is disconnected, skipping eigenvector centrality calculation submission.")
                    GLOBAL_BG_CENTRALITY_EIGENVECTOR = {}

                for future in as_completed(futures):
                    metric_name = futures[future]
                    try:
                        result = future.result()
                        if metric_name == "degree": GLOBAL_BG_CENTRALITY_DEGREE = result; logging.info("Global degree centrality calculated.")
                        elif metric_name == "betweenness": GLOBAL_BG_CENTRALITY_BETWEENNESS = result; logging.info("Global betweenness centrality calculated.")
                        elif metric_name == "closeness": GLOBAL_BG_CENTRALITY_CLOSENESS = result; logging.info("Global closeness centrality calculated.")
                        elif metric_name == "eigenvector": GLOBAL_BG_CENTRALITY_EIGENVECTOR = result; logging.info("Global eigenvector centrality calculated.")
                    except Exception as e:
                        logging.error(f"Error calculating {metric_name} centrality in parallel: {e}", exc_info=True)
            logging.info("Global centrality metrics calculation complete.")
        else:
            logging.warning("Global graph has less than 2 nodes after filtering, skipping global centrality calculations.")

        logging.info(f"Saving global graph data and individual files to '{OUTPUT_DIR_GRAPH}'...")
        graphml_filename = os.path.join(OUTPUT_DIR_GRAPH, "general_graph_1400.graphml")
        global_data_filename = os.path.join(OUTPUT_DIR_GRAPH, "global_graph_data.pkl")
        
        global_data_payload = {
            'nodes': set(general_graph.nodes()),
            'edges_set': set((u, v, data.get('label'), data.get('type')) for u, v, data in general_graph.edges(data=True)),
            'communities': COMMUNITIES_MAP_GLOBAL,
            'bg_degree': GLOBAL_BG_CENTRALITY_DEGREE,
            'bg_betweenness': GLOBAL_BG_CENTRALITY_BETWEENNESS,
            'bg_closeness': GLOBAL_BG_CENTRALITY_CLOSENESS,
            'bg_eigenvector': GLOBAL_BG_CENTRALITY_EIGENVECTOR,
        }

        with ThreadPoolExecutor(max_workers=5) as executor:
            save_futures = []
            save_futures.append(executor.submit(save_graphml_task, general_graph, graphml_filename))
            save_futures.append(executor.submit(save_pickle_task, global_data_payload, global_data_filename))
            save_futures.append(executor.submit(save_pickle_task, general_graph, GRAPH_OUTPUT_FILE))
            save_futures.append(executor.submit(save_pickle_task, general_nodes, NODES_OUTPUT_FILE))
            save_futures.append(executor.submit(save_pickle_task, general_edges_set, EDGES_OUTPUT_FILE))

            for future in as_completed(save_futures):
                success, filename_saved = future.result()
                if success:
                    logging.info(f"Successfully completed saving {filename_saved}")
                else:
                    logging.error(f"Saving failed for {filename_saved}")
        
        logging.info("All saving tasks submitted/completed. Graph building and data saving complete.")
        logging.info("Graph construction and saving finished successfully.")

    except Exception as e:
        logging.error(f"Error in main processing block: {e}", exc_info=True)

if __name__ == "__main__":
    main()