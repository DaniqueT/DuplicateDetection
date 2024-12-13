#%%
import json
import scipy.cluster.hierarchy as sch
from scipy.cluster.hierarchy import fcluster
import re
import random
from collections import defaultdict
import numpy as np
import itertools
from scipy.spatial.distance import squareform
import sympy



def load_dataset(file_path):
    with open(file_path, 'r') as f:
        try:
            data = json.load(f)  
        except json.JSONDecodeError as e:
            print(f"Error loading JSON: {e}")
            return []

    products = []
    for model_id, product_list in data.items():
        if not isinstance(product_list, list):
            print(f"Invalid format for modelID {model_id}, skipping.")
            continue

        for product in product_list:
            if not isinstance(product, dict):
                print(
                    f"Invalid product entry under modelID {model_id}, skipping.")
                continue

            product['modelID'] = model_id
            products.append(product)

    return products


def clean_data(products):
    def normalize_units(text):
        if not isinstance(text, str):
            return text
        transformations = [
            (r'^[-!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~\s]+', ''),
            (r'[-!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~\s]+$', ''),
            (r'("|-?inch(es)?|\s?inch(es)?)', 'inch'),
            (r'(hertz|hz|hz\b|-hz)', 'hz'),
            (r'\s+', ' '), 
            (r'\belectronics\b', ''),  
            (r'\btv\b', ''),  
            (r'\bpansonic\b', 'panasonic'),
        ]
        text = text.lower()
        for pattern, replacement in transformations:
            text = re.sub(pattern, replacement, text)
        return text.strip()

    for product in products:
        product['title'] = normalize_units(product['title'])

        product['featuresMap'] = {
            key: normalize_units(value) for key, value in product['featuresMap'].items()
        }

    return products


def get_distinct_brands(products):
    distinct_brands = set()  

    for product in products:
        features_map = product.get('featuresMap', {})
        brand = features_map.get('Brand')
        if brand: 
            distinct_brands.add(brand)

    for product in products:
        features_map = product.get('featuresMap', {})
        brand = features_map.get('Brand')

        if brand:
            product['brand'] = brand
        else:
            for brand_word in distinct_brands:
                for words in product['title'].split():
                    if brand_word == words:
                        product['brand'] = brand_word
    for product in products:
        brand = product.get('brand')
        if not brand:
            product['brand'] = None

    return sorted(distinct_brands)


def generate_shingles(item):
    model_word_pattern_title = r'\b(\d+[a-zA-Z]+[a-zA-Z0-9]*|\d+\.\d+[a-zA-Z]+|[a-zA-Z]+\d+[a-zA-Z0-9]*)\b'
    shingles = re.findall(model_word_pattern_title, item['title'])
    
    brand = item.get('brand')
    if brand:
        shingles.append(brand)

    return shingles


def generate_allshingles(products):
    all_shingles = set()
    for item in products:
        item['shingles'] = generate_shingles(item)
        all_shingles.update(item['shingles'])

    shingles_vocab = all_shingles
    shingle_to_index = {shingle: idx for idx,
                        shingle in enumerate(shingles_vocab)}
    return (shingles_vocab, shingle_to_index)


def generate_OneHotVector(products, shingle_to_index):
    for item in products:
        one_hot_vector = set(shingle_to_index[shingle] for shingle in item['shingles'] if shingle in shingle_to_index)
        item['shingles_one_hot'] = one_hot_vector

def minhash_signature(sparse_one_hot, hash_functions, p):
    signature = []
    for a, b in hash_functions:
        min_hash = min((a * idx + b) % p for idx in sparse_one_hot)
        signature.append(min_hash)
    return signature

def generate_hash_functions(num_hashes, p):
    hash_functions = []
    for _ in range(num_hashes):
        a = random.randint(1, p - 1)
        b = random.randint(0, p - 1)  
        hash_functions.append((a, b))
    return hash_functions


def banding_method(products, num_bands, rows_per_band):
    buckets = [defaultdict(list)
               for _ in range(num_bands)]  

    for idx, item in enumerate(products):
        signature = item['minhash_signature']
        for band_idx in range(num_bands):
            start = band_idx * rows_per_band
            end = start + rows_per_band
            band = tuple(signature[start:end])
            buckets[band_idx][band].append(idx)

    candidate_pairs = set()
    for bucket in buckets:
        for indices in bucket.values():
            if len(indices) > 1:
                candidate_pairs.update(itertools.combinations(indices, 2))
           
    filtered_candidates = set()
    for a, b in candidate_pairs:
        product1 = products[a]
        product2 = products[b]
        brand1 = product1['brand']
        brand2 = product2['brand']

        if(product1['shop'] != product2['shop']):
            if(brand1 is None or brand2 is None or brand1 ==brand2):
                filtered_candidates.add((a, b))
         
    return filtered_candidates

def generate_qgrams(text, q=3):
    text = re.sub(r'\W', '', text.lower())
    return {text[i:i+q] for i in range(len(text) - q + 1)}

def KVP_similarity(product1, product2, KVPthreshold):
    keys1 = " ".join(product1['featuresMap'].keys())
    keys2 = " ".join(product2['featuresMap'].keys())
    qgrams1Keys = generate_qgrams(keys1, 3)
    qgrams2Keys = generate_qgrams(keys2, 3)
    keysSimilarity = len(qgrams1Keys.intersection(
        qgrams2Keys))/len(qgrams1Keys.union(qgrams2Keys))

    if(keysSimilarity > KVPthreshold):
        values1 = " ".join(product1['featuresMap'].values())
        values2 = " ".join(product2['featuresMap'].values())
        qgrams1Values = generate_qgrams(values1, 3)
        qgrams2Values = generate_qgrams(values2, 3)
        return len(qgrams1Values.intersection(qgrams2Values))/len(qgrams1Values.union(qgrams2Values))

    return -1

def HSM_similarity(product1, product2):
    set1 = set(product1['featuresMap'].values())
    set2 = set(product2['featuresMap'].values())
    return len(set1.intersection(set2))/len(set1.union(set2))


def TMWM_similarity(product1, product2):
    shingle1 = set(product1['shingles'])
    shingle2 = set(product2['shingles'])

    return len(shingle1.intersection(shingle2))/len(shingle1.union(shingle2))


def msm_similarity(products, ida, idb, KVPthreshold, mu, θ1):
    product1 = products[ida]
    product2 = products[idb]
    KVPSim = KVP_similarity(product1, product2, KVPthreshold)
    HSMSim = HSM_similarity(product1, product2)
    TMWMSim = TMWM_similarity(product1, product2)

    if(KVPSim == -1):
        θ2 = 1 - θ1
        return θ1 * HSMSim + θ2*TMWMSim
    else:
        θ2 = 1 - mu - θ1
        return mu*KVPSim + θ1*HSMSim + θ2 * TMWMSim


def hierarchical_clustering(products, candidate_pairs, KVPthreshold, threshold, mu, θ1):

    num_products = len(products)
    distance_matrix = np.full((num_products, num_products), 1000.0)
    
    np.fill_diagonal(distance_matrix, 0.0)

    for a, b in candidate_pairs:
        distance = 1 - msm_similarity(products, a, b, KVPthreshold, mu, θ1)
        distance_matrix[a, b] = distance
        distance_matrix[b, a] = distance  # Since distance_matrix is symmetric

    condensed_distance_matrix = squareform(distance_matrix)

    Z = sch.linkage(condensed_distance_matrix, method='complete')
    clusters = fcluster(Z, t=threshold, criterion='distance')

    return clusters


def evaluate_clusters_with_pq_pc_f1(products, finalpairs, candidate_pairs):
    ground_truth = defaultdict(set)
    for idx, product in enumerate(products):
        if product['modelID']:
            ground_truth[product['modelID']].add(idx)

    total_duplicates = sum(len(items) * (len(items) - 1) //
                           2 for items in ground_truth.values())

    duplicates_found_MSM = sum(
        1 for p1, p2 in finalpairs
        if products[p1]['modelID'] == products[p2]['modelID']
    )
    duplicates_found_LSH = sum(
        1 for p1, p2 in candidate_pairs
        if products[p1]['modelID'] == products[p2]['modelID']
    )

    total_comparisons_MSM = len(finalpairs)
    total_comparisons_LSH = len(candidate_pairs)

    pair_quality = duplicates_found_LSH / total_comparisons_LSH if total_comparisons_LSH > 0 else 0
    pair_completeness = duplicates_found_LSH / \
        total_duplicates if total_duplicates > 0 else 0
    f1_star = (
        2 * (pair_quality * pair_completeness) /
        (pair_quality + pair_completeness)
        if pair_quality + pair_completeness > 0
        else 0
    )

    TP = duplicates_found_MSM
    FP = total_comparisons_MSM - TP
    FN = total_duplicates - duplicates_found_MSM

    precision = TP/(TP + FP) if (TP + FP) > 0 else 0
    recall = TP/(TP + FN)
    f1_score = (2 * precision * recall)/(precision + recall) if(precision + recall) >0 else 0
    return {
        "F1*-Measure": f1_star,
        "Pair Quality (PQ)": pair_quality,
        "Pair Completeness (PC)": pair_completeness,
        "Comparisons_LSH": total_comparisons_LSH,
        "Duplicates found LSH": duplicates_found_LSH,
        "Total products": len(products),
        "F1-score": f1_score,
        "Precision": precision,
        "Recall": recall,
        "Duplicates Found MSM": duplicates_found_MSM,
        "Comparisons MSM": total_comparisons_MSM,
        "Total Duplicates": total_duplicates,
    }


def main(products, hash_functions, p, num_bands, KVPthreshold,threshold, mu, θ1):
    
    shingles_vocab, shingles_to_index = generate_allshingles(products)
    generate_OneHotVector(products, shingles_to_index)

    for item in products:
        one_hot_vector = item['shingles_one_hot']
        item['minhash_signature'] = minhash_signature(
            one_hot_vector, hash_functions, p)

    num_bands = num_bands  
    rows_per_band = int(len(hash_functions)/num_bands)

    candidate_pairs = banding_method(products, num_bands, rows_per_band)

    clusters = hierarchical_clustering(
        products, candidate_pairs, KVPthreshold=KVPthreshold,  threshold=threshold, mu=mu, θ1=θ1)

    cluster_results = []
    for cluster_id in set(clusters):
        cluster_results.append(
            [i for i, c in enumerate(clusters) if c == cluster_id])

    finalpairs = set()
    for cluster in cluster_results:
        if len(cluster) > 1:
            for pair in itertools.combinations(cluster, 2):
                finalpairs.add(pair)


    evaluation_results = evaluate_clusters_with_pq_pc_f1(products, finalpairs, candidate_pairs)
    return evaluation_results

#%%

def bootstrap_split(products):
    n = len(products)
    sampled_indices = [random.randint(0, n - 1) for _ in range(n)]
    unique_train_indices = set(sampled_indices)
    train_data = [products[i] for i in unique_train_indices]
    
    all_indices = set(range(n))
    test_indices = all_indices - unique_train_indices
    test_data = [products[i] for i in test_indices]
    
    return train_data, test_data

final_results_per_split = []

for i in range(5): 
    print(f"iteration {i + 1}")

    file_path = 'C:/Users/daniq/iCloudDrive/Erasmus University Rotterdam/Master/Block 2/Computer Science/TVs-all-merged.json'
    products = load_dataset(file_path)
    products = clean_data(products)
    distinct_brands = get_distinct_brands(products)
    num_hashes = 200 
    p = sympy.randprime(num_hashes, 10*num_hashes)
    hash_functions = generate_hash_functions(num_hashes, p)

    train_data, test_data = bootstrap_split(products)

    all_test_results = defaultdict(list)
    
    KVPthreshold_values = np.arange(0.1, 0.6, 0.1) 
    threshold_values = np.arange(0.3, 0.8, 0.1)
    mu_values = np.arange(0.1, 0.5, 0.1)
    theta1_values = np.arange(0.1, 0.5, 0.1)
    
    total = (
      len(KVPthreshold_values) * 
      len(threshold_values) * 
      len(mu_values) * 
      len(theta1_values) 
  )

    for num_bands in [2, 4, 5, 10, 20, 25, 50, 100]:
        print(f"Evaluating for num_bands = {num_bands}")

        results_list = []
        k= 0
       
        for KVPthreshold, threshold, mu, theta1 in itertools.product(
            KVPthreshold_values, threshold_values, mu_values, theta1_values
        ): 
            k += 1
            print(f"Option {k} from {total} options")
            results = main(
                products=train_data,
                hash_functions= hash_functions,
                p = p,
                num_bands=num_bands,
                KVPthreshold=KVPthreshold,
                threshold=threshold,
                mu=mu,
                θ1=theta1
            )
            results_list.append({
                "F1-score": results.get('F1-score'),
                "KVPthreshold": KVPthreshold,
                "threshold": threshold,
                "mu": mu,
                "θ1": theta1,
                "results": results
            })

        best_result = max(results_list, key=lambda x: x['F1-score'])
        print(f"Best parameters for num_bands = {num_bands}: {best_result}")
        
        print("Evaluating test_data")
        test_results = main(
            products=test_data,
            hash_functions = hash_functions,
            p = p,
            num_bands=num_bands,
            KVPthreshold=best_result["KVPthreshold"],
            threshold=best_result["threshold"],
            mu=best_result["mu"],
            θ1=best_result["θ1"],
        )
        print(f"Test results F1-score = {test_results['F1-score']}")
        all_test_results[num_bands].append(test_results)
    final_results_per_split.append(all_test_results)

print("Calculating final results")
overall_metrics = defaultdict(float)

for split_result in final_results_per_split:
    for num_bands, test_results_list in split_result.items():
        for test_results in test_results_list:
            for key, value in test_results.items():
                if value is not None:
                    overall_metrics[(num_bands, key)] += value

overall_average_metrics = {
    (num_bands, key): value / 5
    for (num_bands, key), value in overall_metrics.items()
}
