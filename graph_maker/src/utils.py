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
import logging
import os
import time
import re
from joblib import Parallel, delayed
from scipy.stats import skew, kurtosis, norm

KEY_DEP_RATIO_PAIRS = [
    ('ROOT', 'nsubj'),
    ('ROOT', 'nsubj:pass'),
    ('ROOT', 'obj'),
    ('ROOT', 'iobj'),
    ('ROOT', 'obl'),
    ('ROOT', 'obl:agent'),
    ('ROOT', 'amod'),
    ('ROOT', 'advmod'),
    ('ROOT', 'nmod'),
    ('ROOT', 'acl'),
    ('ROOT', 'acl:relcl'),
    ('ROOT', 'advcl'),
    ('ROOT', 'ccomp'),
    ('ROOT', 'xcomp'),
    ('ROOT', 'conj'),
    ('ROOT', 'cc'),
    ('ROOT', 'compound'),
    ('ROOT', 'parataxis'),
    ('ROOT', 'discourse'),
    ('ROOT', 'orphan'),
    ('nsubj', 'nsubj:pass'),
    ('nsubj', 'obj'),
    ('nsubj', 'iobj'),
    ('nsubj', 'obl'),
    ('nsubj', 'obl:agent'),
    ('nsubj', 'amod'),
    ('nsubj', 'advmod'),
    ('nsubj', 'nmod'),
    ('nsubj', 'acl'),
    ('nsubj', 'acl:relcl'),
    ('nsubj', 'advcl'),
    ('nsubj', 'ccomp'),
    ('nsubj', 'xcomp'),
    ('nsubj', 'conj'),
    ('nsubj', 'cc'),
    ('nsubj', 'compound'),
    ('nsubj', 'parataxis'),
    ('nsubj', 'discourse'),
    ('nsubj', 'orphan'),
    ('nsubj:pass', 'obj'),
    ('nsubj:pass', 'iobj'),
    ('nsubj:pass', 'obl'),
    ('nsubj:pass', 'obl:agent'),
    ('nsubj:pass', 'amod'),
    ('nsubj:pass', 'advmod'),
    ('nsubj:pass', 'nmod'),
    ('nsubj:pass', 'acl'),
    ('nsubj:pass', 'acl:relcl'),
    ('nsubj:pass', 'advcl'),
    ('nsubj:pass', 'ccomp'),
    ('nsubj:pass', 'xcomp'),
    ('nsubj:pass', 'conj'),
    ('nsubj:pass', 'cc'),
    ('nsubj:pass', 'compound'),
    ('nsubj:pass', 'parataxis'),
    ('nsubj:pass', 'discourse'),
    ('nsubj:pass', 'orphan'),
    ('obj', 'iobj'),
    ('obj', 'obl'),
    ('obj', 'obl:agent'),
    ('obj', 'amod'),
    ('obj', 'advmod'),
    ('obj', 'nmod'),
    ('obj', 'acl'),
    ('obj', 'acl:relcl'),
    ('obj', 'advcl'),
    ('obj', 'ccomp'),
    ('obj', 'xcomp'),
    ('obj', 'conj'),
    ('obj', 'cc'),
    ('obj', 'compound'),
    ('obj', 'parataxis'),
    ('obj', 'discourse'),
    ('obj', 'orphan'),
    ('iobj', 'obl'),
    ('iobj', 'obl:agent'),
    ('iobj', 'amod'),
    ('iobj', 'advmod'),
    ('iobj', 'nmod'),
    ('iobj', 'acl'),
    ('iobj', 'acl:relcl'),
    ('iobj', 'advcl'),
    ('iobj', 'ccomp'),
    ('iobj', 'xcomp'),
    ('iobj', 'conj'),
    ('iobj', 'cc'),
    ('iobj', 'compound'),
    ('iobj', 'parataxis'),
    ('iobj', 'discourse'),
    ('iobj', 'orphan'),
    ('obl', 'obl:agent'),
    ('obl', 'amod'),
    ('obl', 'advmod'),
    ('obl', 'nmod'),
    ('obl', 'acl'),
    ('obl', 'acl:relcl'),
    ('obl', 'advcl'),
    ('obl', 'ccomp'),
    ('obl', 'xcomp'),
    ('obl', 'conj'),
    ('obl', 'cc'),
    ('obl', 'compound'),
    ('obl', 'parataxis'),
    ('obl', 'discourse'),
    ('obl', 'orphan'),
    ('obl:agent', 'amod'),
    ('obl:agent', 'advmod'),
    ('obl:agent', 'nmod'),
    ('obl:agent', 'acl'),
    ('obl:agent', 'acl:relcl'),
    ('obl:agent', 'advcl'),
    ('obl:agent', 'ccomp'),
    ('obl:agent', 'xcomp'),
    ('obl:agent', 'conj'),
    ('obl:agent', 'cc'),
    ('obl:agent', 'compound'),
    ('obl:agent', 'parataxis'),
    ('obl:agent', 'discourse'),
    ('obl:agent', 'orphan'),
    ('amod', 'advmod'),
    ('amod', 'nmod'),
    ('amod', 'acl'),
    ('amod', 'acl:relcl'),
    ('amod', 'advcl'),
    ('amod', 'ccomp'),
    ('amod', 'xcomp'),
    ('amod', 'conj'),
    ('amod', 'cc'),
    ('amod', 'compound'),
    ('amod', 'parataxis'),
    ('amod', 'discourse'),
    ('amod', 'orphan'),
    ('advmod', 'nmod'),
    ('advmod', 'acl'),
    ('advmod', 'acl:relcl'),
    ('advmod', 'advcl'),
    ('advmod', 'ccomp'),
    ('advmod', 'xcomp'),
    ('advmod', 'conj'),
    ('advmod', 'cc'),
    ('advmod', 'compound'),
    ('advmod', 'parataxis'),
    ('advmod', 'discourse'),
    ('advmod', 'orphan'),
    ('nmod', 'acl'),
    ('nmod', 'acl:relcl'),
    ('nmod', 'advcl'),
    ('nmod', 'ccomp'),
    ('nmod', 'xcomp'),
    ('nmod', 'conj'),
    ('nmod', 'cc'),
    ('nmod', 'compound'),
    ('nmod', 'parataxis'),
    ('nmod', 'discourse'),
    ('nmod', 'orphan'),
    ('acl', 'acl:relcl'),
    ('acl', 'advcl'),
    ('acl', 'ccomp'),
    ('acl', 'xcomp'),
    ('acl', 'conj'),
    ('acl', 'cc'),
    ('acl', 'compound'),
    ('acl', 'parataxis'),
    ('acl', 'discourse'),
    ('acl', 'orphan'),
    ('acl:relcl', 'advcl'),
    ('acl:relcl', 'ccomp'),
    ('acl:relcl', 'xcomp'),
    ('acl:relcl', 'conj'),
    ('acl:relcl', 'cc'),
    ('acl:relcl', 'compound'),
    ('acl:relcl', 'parataxis'),
    ('acl:relcl', 'discourse'),
    ('acl:relcl', 'orphan'),
    ('advcl', 'ccomp'),
    ('advcl', 'xcomp'),
    ('advcl', 'conj'),
    ('advcl', 'cc'),
    ('advcl', 'compound'),
    ('advcl', 'parataxis'),
    ('advcl', 'discourse'),
    ('advcl', 'orphan'),
    ('ccomp', 'xcomp'),
    ('ccomp', 'conj'),
    ('ccomp', 'cc'),
    ('ccomp', 'compound'),
    ('ccomp', 'parataxis'),
    ('ccomp', 'discourse'),
    ('ccomp', 'orphan'),
    ('xcomp', 'conj'),
    ('xcomp', 'cc'),
    ('xcomp', 'compound'),
    ('xcomp', 'parataxis'),
    ('xcomp', 'discourse'),
    ('xcomp', 'orphan'),
    ('conj', 'cc'),
    ('conj', 'compound'),
    ('conj', 'parataxis'),
    ('conj', 'discourse'),
    ('conj', 'orphan'),
    ('cc', 'compound'),
    ('cc', 'parataxis'),
    ('cc', 'discourse'),
    ('cc', 'orphan'),
    ('compound', 'parataxis'),
    ('compound', 'discourse'),
    ('compound', 'orphan'),
    ('parataxis', 'discourse'),
    ('parataxis', 'orphan'),
    ('discourse', 'orphan')
]

GENERIC_ENTITY_MAP = {'PER': 'Человек', 'LOC': 'Место', 'ORG': 'Организация', 'GPE': 'Место', 'FAC': 'Объект', 'NORP': 'Национальность', 'PRODUCT': 'Продукт', 'EVENT': 'Событие', 'WORK_OF_ART': 'Произведение'}

try:
    morph = MorphAnalyzer()
    logging.info("Pymorphy3 MorphAnalyzer успешно инициализирован.")
except Exception as e:
    logging.error(f"Ошибка инициализации Pymorphy3: {e}", exc_info=True)

@lru_cache(maxsize=2048)
def parse_morph(text):
    """Кэшированный морфологический анализ текста с помощью Pymorphy3."""
    try:
        return morph.parse(text)
    except Exception as e:
        logging.warning(f"MorphAnalyzer не смог разобрать '{text}': {e}")
        return []

def sanitize_string_for_xml(s):
    """Очищает строку от недопустимых XML символов."""
    if not isinstance(s, str):
        return s
    s = re.sub(r'[\x00-\x1F\x7F-\x9F]', '', s)
    return s

@lru_cache(maxsize=1000)
def head_in_named_entity(doc, span):
    """Находит головной токен в именованной сущности (Span) и возвращает его информацию."""
    if not span:
        return None
    head = span.root if span.root else (span[0] if span else None)
    if head:
        morph_info = [part.split('=')[1] for part in str(head.morph).split('|') if '=' in part]
        return head, morph_info, head.head, head.dep_
    return None

def normalize_noun_phrase(doc, np_obj):
    """Нормализует фразу или токен (особенно существительные и сущности) к нормальной форме."""
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
             return sanitize_string_for_xml(np_obj.lemma_ if np_obj.root else np_obj.text)

        tokens_lemmas = [sanitize_string_for_xml(token.lemma_) for token in np_obj]
        if head.i - np_obj.start < len(tokens_lemmas):
            tokens_lemmas[head.i - np_obj.start] = sanitize_string_for_xml(normalized_head)
        return " ".join(tokens_lemmas)

    return sanitize_string_for_xml(str(np_obj) if hasattr(np_obj, 'text') else str(np_obj))

def contains_only_russian_or_latin_letters(text):
    """Проверяет, содержит ли текст только русские, латинские буквы, цифры, пробелы и основные знаки пунктуации."""
    return not re.search(r'[^\s\w.,!?;:\-_()ЁёА-яA-Za-z0-9]', text)


def get_syntactic_relations(doc):
    """Извлекает синтаксические отношения (связи) из обработанного SpaCy документа (Doc)."""
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

        src_node_idx = head.i if head.i in chunks else -1
        tgt_node_idx = token.i if token.i in chunks else -1

        if src_node_idx == -1 or tgt_node_idx == -1:
             continue

        src_concept_data = chunks.get(src_node_idx)
        tgt_concept_data = chunks.get(tgt_node_idx)

        if src_concept_data and tgt_concept_data and src_concept_data['text'] and tgt_concept_data['text']:
            src_concept = src_concept_data['text']
            tgt_concept = tgt_concept_data['text']

            edge_label = sanitize_string_for_xml(token.dep_)
            edge_type = 'dependency'

            if token.dep_ == 'conj' and head.dep_ == 'nsubj' and head.head == token.head:
                 edge_label = sanitize_string_for_xml('conj_subj')
                 edge_type = 'custom'

            if src_concept and tgt_concept:
                 relations.append((src_concept, edge_label, tgt_concept, edge_type))

    return relations

def build_graph_from_relations(relations_list):
    """Строит ориентированный граф NetworkX из списка извлеченных связей."""
    G = nx.DiGraph()
    edges_data = {}

    for rel_tuple in relations_list:
        if len(rel_tuple) == 4:
            src = rel_tuple[0]
            lbl = rel_tuple[1]
            tgt = rel_tuple[2]
            typ = rel_tuple[3]

            src_str = sanitize_string_for_xml(str(src))
            tgt_str = sanitize_string_for_xml(str(tgt))

            G.add_node(src_str)
            G.add_node(tgt_str)

            edge_key = (src_str, tgt_str, lbl, typ)
            if edge_key not in edges_data:
                 edges_data[edge_key] = {'count': 0}
            edges_data[edge_key]['count'] += 1
        else:
            logging.warning(f"Пропускается связь с неожиданным форматом: {rel_tuple}")

    unique_edges_for_set = set()
    for edge_key, data in edges_data.items():
        src, tgt, lbl, typ = edge_key
        weight = data['count']
        G.add_edge(src, tgt, label=lbl, type=typ, weight=weight)
        unique_edges_for_set.add(edge_key)

    return G, set(G.nodes()), unique_edges_for_set
