#!/usr/bin/env python
# -*- coding: utf-8 -*-
__author__ = 'Matteo'

import hashlib
from nltk import Tree
from cky import cky_and_viterbi, cky_parser
from util import translate_single_tag, get_pcfg, save_obj_to_file

# ATTENZIONE: funziona solo se l'albero è in chomsky normal form
# restituisce BOOLEAN, ALBERO dove boolean dice se ha trovato un verbo e albero è l'albero da spostare in testa
def find_deepleftfirst_verb(t):
    """
    Restituisce una versione XSV di un albero che rappresenta una frase SVO
    :param t: albero SVO in CNF
    :return: albero XSV
    """
    # se l'albero è None, ritorno False e None
    # ATTENZIONE: questo controllo non servirebbe
    if t == None:
        return False, None

    # se è un nodo "quasi foglia" ovvero è un nodo che ha un solo figlio
    # e il suo label è un verbo
    if len(t) == 1 and translate_single_tag(t.label()) == "VERB":
        # allora ho trovato un verbo
        return True, None
    elif len(t) == 1:
        # altrimenti è un altro tag ma non ci interessa
        return False, None

    # indice del figlio dell'albero che sto considerando
    index = 0
    # per ogni figlio dell'albero
    for child in t:
        # cerco un verbo (NON UNA VP, ma un vero e proprio verbo)
        # questo perché sarebbe troppo complicato estrarre la parte
        # "successiva" a VP: non sapremmo a quale livello di profondità
        # iniziare a cercare / fermarci (se consideriamo di risalire le
        # chiamate ricorsive dal basso)
        trovato, albero = find_deepleftfirst_verb(child)
        # se nel sotto albero figlio che sto considerando c'è il verbo
        # e non ho ancora costruito il pezzo di frase "successivo al verbo"
        # da spostare, allora
        if trovato and albero == None:
            # rimuovo dall'albero attuale tutti i figli che si trovano
            # "dopo" il figlio attuale: per farlo concateno questi
            # in una lista e poi uno ad uno li rimuovo dall'albero originale:
            # questo per mantenere il puntatore all'albero originale (t)
            temp_list = [n for n in t[index+1:]]
            for temp_c in temp_list:
                t.remove(temp_c)
            # ritorno True, perché ho trovato il primo verbo (con ricerca
            # deep - left) e ricostruisco il pezzo di frase successivo al verbo
            # a partire da quel livello di profondità.
            return True, Tree("X", temp_list)
            # return True, Tree(t.label(), temp_list)
        # se ho trovato il sottoalbero con il primo verbo della frase
        # ma ho già costruito un pezzo dell'albero (che rappresenta la frase
        # "successiva" al verbo che ho scelto) da inserire in testa alla verb phrase
        # definita dal primo albero in cui ho trovato il primo verbo della frase,
        # (NOTA BENE: in pratica sono a profondità minore dell'albero in cui ho trovato il verbo
        # e se esistono delle cose più a destra, sto per ricostruirle attaccandole a destra dell'albero
        # che ho attualmente)
        elif trovato:
            # se più a destra di dove sono non c'è più niente (la "lunghezza" dell'albero)
            # che è una lista è uguale all'indice del nodo attuale
            if index == len(t)-1:
                # allora ho ricostruito tutto l'albero da attaccare in testa in modo corretto
                return True, albero
            else:
                # rimuovo dall'albero attuale tutti i figli che si trovano
                # "dopo" il figlio attuale: per farlo concateno questi
                # in una lista e poi uno ad uno li rimuovo dall'albero originale:
                # questo per mantenere il puntatore all'albero originale (t)
                temp_list = [n for n in t[index+1:]]
                for temp_c in temp_list:
                    t.remove(temp_c)
                # PASSO DI COMPOSIZIONE: compongo l'albero che ho già
                # attualmente ('albero') con tutti i figli a destra
                # concateno sempre i figli dell'albero che ho creato
                # risalendo quindi la X compare solo l'ultima volta che lo creo
                # Così non sarà più in CNF. Ma...
                tree_composto = Tree("X", [n for n in albero]+temp_list)
#                tree_composto = Tree(albero.label(), [albero]+temp_list)
                return True, tree_composto

        index += 1

    return False, None

def create_xsvversion_of_tree(t):
    founded, sub_tree = find_deepleftfirst_verb(t)
    if sub_tree != None:
        t._label = "S1"
        temp = Tree("S", [sub_tree]+[t])
        return temp
    return t

def testing_svo_to_xsv(config, sentence, t = None):

    if t == None:
        grammar, test_set, training_set, all_sentences, terminal_symbols = get_pcfg(config, config.TREEBANKGOOGLE_FILE, "S", (9.0/10.0), limit = -1, treebank_modified_file = None)
        t = cky_and_viterbi(config, sentence.split(" "), sentence.split(" "), grammar, ["S", "NP"])
    if t != None:
        ot = t.copy(deep=True)
        nt = create_xsvversion_of_tree(t)
        return ot, nt
    else:
        config.LOGGER.info("Nessun albero trovato per la frase input!")
        return None, None