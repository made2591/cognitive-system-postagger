#!/usr/bin/env python
# -*- coding: utf-8 -*-
__author__ = 'Matteo'

import unicodedata, subprocess, sys, nltk
from nltk import Tree
from config import Config
from util import save_obj_to_file, get_obj_from_file, translate_single_tag
from viterbi import viterbi_alg, train_model

global viterbi_dump_for_cky

def cky_parser(config, words, pcfg_cnf_grammar, start_symbols):
    """
    Questo metodo esegue il parsing con CKY
    :param words: list => parole da analizzare
    :param pcfg_cnf_grammar: grammar => grammatica PCFG in CNF
    :param start_symbols: list => start symbol(s) eventualmente più di uno per
                          come è conformato il treebank
    :return: tree or none
    """
    # alias
    log = config.LOGGER
    log.debug("Inizio esecuzione di CKY per la sentence: "+str(words))
    # numero di parole
    n_words = len(words)
    # inizializzo la tabella tridimensionale
    table = [[dict() for x in range(0, n_words + 1)] for x in range(0, n_words)]
    # inizializzo il backpointer
    back = [[dict() for x in range(0, n_words + 1)] for x in range(0, n_words)]
    # inizializzo la diagonale: per ogni parola
    for j in range(1, n_words + 1):
        # alias
        parola = words[j - 1]
        # recupero tutte le regole della grammatica che producono quella parola
        rules = pcfg_cnf_grammar.productions(None, parola)
        # per ogni regola che ho recuperato
        for r in rules:
            log.debug("\tInserisco la regola '%s' per la parola '%s'", str(r), parola)
            # introduco nella posizione j-1, j la testa A e la probabilità che A -> parola
            table[j - 1][j][str(r.lhs())] = r.prob()
        # se non c'è nessuna regola lessicale per quella parola mando in qualsiasi regola lessicale
        if len(rules) == 0:
            log.debug("\tInserisco tutte le regole lessicali per la parola '%s'", parola)
            # per ogni regola della grammatica
            for p in pcfg_cnf_grammar.productions():
                # se è lessicale
                if p.is_lexical():
                    # aggiungo la sua testa come potenziale regola adatta a produrre la parola
                    table[j - 1][j][str(p.lhs())] = p.prob()

    # per ogni altra parola dopo la prima
    for j in range(1, n_words + 1):
        log.debug("Valuto indice j valore %s di %s: ", str(j), str(n_words))
        for i in reversed(range(0, j - 1)):
            log.debug("\tValuto indice i valore %s (%s di %s): ", str(i), str(abs(j-1-i)), str(j-1))
            # cerco tutte le regole A -> BC tali per cui B è in table[i][k] e C in table[k][j]
            # Per farlo scorro tutte le teste B in table[i][k]
            for k in range(i + 1, j):
                log.debug("\t\tValuto indice k valore %s (%s di %s): ", str(k), str(j-(i+1)), str(j))
                for B in table[i][k]:
                    # per ogni potenziale regola con corpo B* (B e qualcosa)
                    brules = pcfg_cnf_grammar.productions(None, nltk.Nonterminal(B))
                    # per ogni regola con testa C in table[k][j]
                    for C in table[k][j]:
                        # per ogni regola isolata cercando le regole B*
                        for BCrule in brules:
                            # se quella regola è una regola BC ovvero il secondo
                            # non terminale [1] è uguale alla testa C su cui stiamo iterando
                            if str(BCrule.rhs()[1]) == C:
                                log.debug("\t\t\tIsolo la regola '%s'", str(BCrule))
                                # recupero la testa della regola A -> BC che ho trovato:
                                # se non ho ancora inizializzato la sua probabilità, la metto a 0
                                if str(BCrule.lhs()) not in table[i][j].keys():
                                    table[i][j][str(BCrule.lhs())] = 0.0
                                # se la probabilità di A in table[i][j] è minore del prodotto della probabilità
                                # della regola che ho appena isolato (A -> BC) per la probabilità della regola
                                # B (che produce dalla parole i alla parola k) per la probabilità della regola
                                # C (che produce le restanti parole dalla k alla j) allora ho trovato una nuova
                                # configurazione per produrre A -> BC più probabile quindi...
                                if table[i][j][str(BCrule.lhs())] <= BCrule.prob() * table[i][k][B] * table[k][j][C]:
                                    log.debug("\t\t\tAumento la probabilità per la regola '%s'", str(BCrule))
                                    # aggiorno la probabilità di A in table[i][j] di produrre le parole dalla i alla j
                                    # con le regole che...
                                    table[i][j][str(BCrule.lhs())] = BCrule.prob() * table[i][k][B] * table[k][j][C]
                                    # le cui teste posso trovare poi in i,k e k,j tramite la struttura BACK pointer
                                    back[i][j][str(BCrule.lhs())] = tuple((k, B, C))
    # se nell'ultima colonna
    for i in table[0][n_words]:
        # esiste una chiave (una testa di una regola) che è presente nella lista di start symbols
        # accettati allora...
        if str(i) in start_symbols:
            # ricostruisco l'albero di parsing
            stringtree = build_parsing_tree(back, words, str(i))
            # se non ci sono estati errori di costruzione => NON DOVREBBE MAI SUCCEDERE
            # Se succede c'è qualche problema nel codice
            if "ERRORE" not in stringtree:
                # ritorno l'albero
                return nltk.Tree.fromstring(stringtree)
            else:
                log.error("Qualcosa è andato storto nella ricostruzione dell'albero di parsing")

    return None

def cky_and_viterbi(config, words, real_sentence, pcfg_cnf_grammar, start_symbols):
    """
    Questo algoritmo esegue il CKY con Viterbi. Per funzionare, richiede la sentence con i pos
    tag di google e la sentence originale con i terminali = parole
    :param words: parole (postag di google)
    :param real_sentence: (frase, parole vere)
    :param pcfg_cnf_grammar: grammar => grammatica PCFG in CNF
    :param start_symbols: list => start symbol(s) eventualmente più di uno per
                          come è conformato il treebank
    :return: tree or none
    """
    # alias
    log = config.LOGGER
    log.info("Inizio esecuzione di CKY + Viterbi per la sentence: "+str(real_sentence))
    # numero di parole
    n_words = len(words)
    # inizializzo la tabella tridimensionale
    table = [[dict() for x in range(0, n_words + 1)] for x in range(0, n_words)]
    # inizializzo il backpointer
    back = [[dict() for x in range(0, n_words + 1)] for x in range(0, n_words)]
    # inizializzo la diagonale: per ogni parola
    log.info("Eseguo Viterbi per ottenere i POSTag di google corretti per le parole della frase")
    result = viterbi_for_leaves(config, real_sentence)
    # isolo i postag
    pos_tag_google = result[1]
    log.info("Tag trovati da viterbi  : "+str(pos_tag_google))
    # per ogni regola della grammatica
    p = pcfg_cnf_grammar.productions()
    for j in range(1, n_words + 1):

        # se non c'è nessuna regola lessicale per quella parola mando in qualsiasi regola lessicale
        if pos_tag_google[j-1] == u"X":
            log.debug("\tInserisco tutte le regole lessicali per la parola '%s'", real_sentence[j-1])
            # per ogni regola della grammatica
            for r in pcfg_cnf_grammar.productions():
                # se è lessicale
                if r.is_lexical():
                    # aggiungo la sua testa come potenziale regola adatta a produrre la parola
                    table[j - 1][j][str(r.lhs())] = r.prob()
        else:
            #log.info("Parola: %s, risultato di viterbi: %s", real_sentence[j-1], pos_tag_google[j-1]+Config.POS_TAG_GOOGLE_CONSTANT)
            # per ogni regola di produzione della grammatica
            for r in p:
                # se è una regola lessicale che produce il POSTag predetto da Viterbi, allora aggiungo la testa della regola alla cella
                # della diagonale per quella parola
                if r.is_lexical() and r.rhs()[0].encode('utf-8') == pos_tag_google[j-1]+Config.POS_TAG_GOOGLE_CONSTANT.decode('utf-8'):
                    log.debug("\tRegola aggiunta alla diagonale: %s ", str(r))
                    table[j - 1][j][str(r.lhs())] = r.prob()

    # per ogni altra parola dopo la prima
    for j in range(1, n_words + 1):

        #log.info("\tValuto indice j valore %s di %s: ", str(j), str(n_words))
        for i in reversed(range(0, j - 1)):
            log.debug("\tValuto indice i valore %s (%s di %s): ", str(i), str(abs(j-1-i)), str(j-1))
            # cerco tutte le regole A -> BC tali per cui B è in table[i][k] e C in table[k][j]
            # Per farlo scorro tutte le teste B in table[i][k]
            for k in range(i + 1, j):
                log.debug("\t\tValuto indice k valore %s (%s di %s): ", str(k), str(j-(i+1)), str(j))
                for B in table[i][k]:
                    # per ogni potenziale regola con corpo B* (B e qualcosa)
                    brules = pcfg_cnf_grammar.productions(None, nltk.Nonterminal(B))
                    # per ogni regola con testa C in table[k][j]
                    for C in table[k][j]:
                        # per ogni regola isolata cercando le regole B*
                        for BCrule in brules:
                            # se quella regola è una regola BC ovvero il secondo
                            # non terminale [1] è uguale alla testa C su cui stiamo iterando
                            if str(BCrule.rhs()[1]) == C:
                                log.debug("\t\t\tIsolo la regola '%s'", str(BCrule))
                                # recupero la testa della regola A -> BC che ho trovato:
                                # se non ho ancora inizializzato la sua probabilità, la metto a 0
                                if str(BCrule.lhs()) not in table[i][j].keys():
                                    table[i][j][str(BCrule.lhs())] = 0.0
                                # se la probabilità di A in table[i][j] è minore del prodotto della probabilità
                                # della regola che ho appena isolato (A -> BC) per la probabilità della regola
                                # B (che produce dalla parole i alla parola k) per la probabilità della regola
                                # C (che produce le restanti parole dalla k alla j) allora ho trovato una nuova
                                # configurazione per produrre A -> BC più probabile quindi...
                                if table[i][j][str(BCrule.lhs())] <= BCrule.prob() * table[i][k][B] * table[k][j][C]:
                                    log.debug("\t\t\tAumento la probabilità per la regola '%s'", str(BCrule))
                                    # aggiorno la probabilità di A in table[i][j] di produrre le parole dalla i alla j
                                    # con le regole che...
                                    table[i][j][str(BCrule.lhs())] = BCrule.prob() * table[i][k][B] * table[k][j][C]
                                    # le cui teste posso trovare poi in i,k e k,j tramite la struttura BACK pointer
                                    back[i][j][str(BCrule.lhs())] = tuple((k, B, C))
    # se nell'ultima colonna
    for i in table[0][n_words]:
        # esiste una chiave (una testa di una regola) che è presente nella lista di start symbols
        # accettati allora...
        if str(i) in start_symbols:
            # ricostruisco l'albero di parsing
            stringtree = build_parsing_tree(back, words, str(i))
            # se non ci sono estati errori di costruzione => NON DOVREBBE MAI SUCCEDERE
            # Se succede c'è qualche problema nel codice
            if "ERRORE" not in stringtree:
                # ritorno l'albero
                return nltk.Tree.fromstring(stringtree)
            else:
                log.error("Qualcosa è andato storto nella ricostruzione dell'albero di parsing")

    return None

def viterbi_for_leaves(config, sentence):
    """
    Questo metodo wrappa l'esecuzione di viterbi per l'algoritmo CKY + Viterbi
    :param sentence: sentence in ingresso
    :return: probabilità + sequenza di postag più probabile
    """
    # alias
    log = config.LOGGER
    log.debug("Inizio esecuzione di Viterbi interna al CKY per la sentence: "+str(sentence))
    # imposto l'esecuzione ottimale
    Config.POS_TAG = 1
    Config.UNKNOW_WORD = 3
    Config.CAPITALMODE = 2
    Config.STRING_VERSION = str(Config.POS_TAG)+"."+str(Config.UNKNOW_WORD)+"."+str(Config.CAPITALMODE)
    log.debug("Carico il dump dell'ultimo training")
    viterbi_dump_for_cky = get_obj_from_file(config.DUMP_TRAINING)
    obj = viterbi_dump_for_cky
    transition_probs, emission_probs, single_words_distribution, start_probs = obj[0], obj[1], obj[2], obj[3]

    results = viterbi_alg(sentence, tuple(transition_probs.keys()), start_probs, transition_probs, emission_probs, single_words_distribution, 3)
    return results[0], results[1]

def build_parsing_tree(back, words, start_symbol):
    """
    Questo metodo ricostruisce l'albero di parsing a partire dalla matrice di backpointer generata dal CKY
    :param back: matrice di back pointer
    :param words: parole della frase
    :param start_symbols: list => start symbol(s) eventualmente più di uno per
                          come è conformato il treebank
    :return: stringa che rappresenta l'albero
    """
    # numero di parole
    n_words = len(words)
    # costruisco i sotto alberi in modo ricorsivo
    b = build_subtree(0, n_words, start_symbol, back, words)
    return b

def build_subtree(i, j, nonterminal_root, back, words):
    """
    Metodo ricorsivo
    :param i: indice
    :param j: indice
    :param nonterminal_root: root del sotto albero che stiamo valutando
    :param back: struttura back
    :param words: parole
    :return:
    """
    # se la radice dell'albero che stiamo considerando è dentro la struttura back all'indice i,j
    if nonterminal_root in back[i][j]:
        # allora il nodo che stiamo costruendo è il nodo table[i][j][nonterminal_root]
        node = back[i][j][nonterminal_root]
        # il non terminale B che produce i termini dall'indice i all'indice k costruisce
        # il sottoalbero che va da i a k
        b = build_subtree(i, node[0], str(node[1]), back, words)
        # il non terminale C che produce i termini dall'indice i all'indice k costruisce
        # il sottoalbero che va da k a j
        c = build_subtree(node[0], j, str(node[2]), back, words)
        return "(" + nonterminal_root + " " + b + " " + c + ")"
    elif i == j-1:
        # siamo arrivati a termine della frase quindi il nodo attuale produce l'ultima parola
        return "(" + nonterminal_root + " " + words[j - 1] + ")"
    else:
        # c'è qualcosa di storto
        return "ERRORE"

def print_cky_table(table):
    """
    Stamba una tabella: comodo per debuggare il CKY
    :param table: tabella bidimensionale
    :return: none
    """
    s = [[str(e.keys()) for e in row] for row in table]
    lens = [max(map(len, col)) for col in zip(*s)]
    fmt = '\t'.join('{{:{}}}'.format(x) for x in lens)
    table = [fmt.format(*row) for row in s]
    print '\n'.join(table)

def test_cky(config, gold_file, test_file, test_set, pcfg_cnf_grammar):
    """
    Testa il CKY nella sua versione base
    :param config: configurazione
    :param gold_file: file dove verranno salvati i gli alberi gold valutati
    :param test_file: file dove verranno salvati i gli alberi test generati
    :param test_set: test_set (list di sentence) da valutare
    :param pcfg_cnf_grammar: grammar => grammatica PCFG in CNF
    :return:
    """
    # alias
    log = config.LOGGER
    log.info("Inizio testing di CKY")
    # creo i puntatori ai due file di test e di log
    my_gold_file = open(gold_file, "w")
    my_test_file = open(test_file, "w")
    # contatore degli alberi trovati
    founded_tree = 0
    # alberi già calcolati
    counter_tested = 0
    # # stringa in
    # original_tree_string = ""
    # founded_tree_string = ""
    for s in test_set:

        counter_tested += 1
        # calcolo la percentuale di avanzamento
        percentage = float((float(counter_tested) / float(len(test_set))) * 100.0)
        if(float(percentage % 10) == 0.0):
            log.info("\tPercentuale avanzamento: %s%% (%s di %s)", str(int(percentage)), str(counter_tested), str(len(test_set)))

        # leggo l'albero
        t = Tree.fromstring(s.decode('utf-8'), remove_empty_top_bracketing=True)
        # copio l'albero in una struttura di supporto
        original_form = t.copy(deep=True)
        # converto l'albero in chomsky normal form
        nltk.treetransforms.chomsky_normal_form(original_form, factor='right', horzMarkov=1, vertMarkov=1, childChar='p', parentChar='u')
        # collaso le produzioni unarie
        t.collapse_unary(collapsePOS=True, collapseRoot=True)

        # converto l'albero in chomsky normal form
        nltk.treetransforms.chomsky_normal_form(t, factor='right', horzMarkov=1, vertMarkov=1, childChar='p', parentChar='u')
        log.debug("Trovati :"+str(founded_tree)+" su "+str(counter_tested)+" valutati di "+str(len(test_set))+" totali.")
        # trasformo in stringa l'albero originale
        original_tree = unicodedata.normalize('NFKD', t.pprint()).encode('ascii','ignore').replace("\n", "")
        # eseguo il CKY
        t = cky_parser(config, t.leaves(), pcfg_cnf_grammar, ["S", "NP"])
        # se ho trovato un albero
        if t != None:
            # lo trasformo in stringa
            tree_founded = unicodedata.normalize('NFKD', t.pprint()).encode('ascii','ignore').replace("\n", "")
            log.debug("Albero originale: "+original_tree)
            # original_tree_string += " ".join(original_tree.split())+"\n"
            # founded_tree_string += " ".join(tree_founded.split())+"\n"
            # incremento il numero di alberi trovati correttamente
            founded_tree += 1
            log.info("Albero trovato  : "+tree_founded)
            my_gold_file.write(unicode(original_tree)+"\n")
            my_gold_file.flush()
            my_test_file.write(unicode(tree_founded)+"\n")
            my_test_file.flush()
        else:
            log.info("Albero non trovato per sentence "+s)
            my_gold_file.write(unicode(original_tree)+"\n")
            my_gold_file.flush()
            my_test_file.write("()\n")
            my_test_file.flush()

    log.info("Fine testing di CKY")
    my_gold_file.close()
    my_test_file.close()

def test_cky_with_viterbi(config, pcfg_cnf_grammar, test_set, gold_file, test_file, original_testing_set):
    """
    Testa il CKY nella sua versione con Viterbi
    :param config: configurazione
    :param pcfg_cnf_grammar: grammar => grammatica PCFG in CNF
    :param test_set: test_set (list di sentence) da valutare
    :param gold_file: file dove verranno salvati i gli alberi gold valutati
    :param test_file: file dove verranno salvati i gli alberi test generati
    :param original_testing_set: test_set originale corrispettivo di quello presente in test_set ma con
           le parole al posto dei postag di google
    :return:
    """
    # alias
    log = config.LOGGER
    log.info("Inizio testing di CKY con Viterbi")
    # counter tested tree
    counter_tested = 0
    # apro i file di gold e test
    my_gold_file = open(gold_file, "w")
    my_test_file = open(test_file, "w")
    # per ogni sentence presente nel test set, scorro parallalelamente
    # la sentence con i tag di google e la sentence con le parole originali (da passare a viterbi)
    for sentence_with_google, real_sentence in zip(test_set, original_testing_set):
        # # try:
        # # aumento il numero di sentence elaborate
        counter_tested += 1
        # calcolo la percentuale di avanzamento
        percentage = float((float(counter_tested) / float(len(test_set))) * 100.0)
        if(float(percentage % 10) == 0.0):
            log.info("\tPercentuale avanzamento: %s%% (%s di %s)", str(int(percentage)), str(counter_tested), str(len(test_set)))

        # leggo l'albero con i tag di google e lo trasformo in CNF
        tree_with_google = Tree.fromstring(sentence_with_google.decode('utf-8'), remove_empty_top_bracketing=True)
        tree_with_google.collapse_unary(collapsePOS=True, collapseRoot=True)
        nltk.treetransforms.chomsky_normal_form(tree_with_google, factor='right', horzMarkov=1, vertMarkov=1, childChar='p', parentChar='u')
        # ne ricavo una versione a stringa
        original_tree = tree_with_google.pprint().replace("\n", "")
        log.debug("Albero originale con i postag di Google: "+original_tree)
        # leggo l'albero con le parole e lo trasformo in CNF
        tree_with_word = Tree.fromstring(real_sentence.decode('utf-8'), remove_empty_top_bracketing=True)
        tree_with_word.collapse_unary(collapsePOS=True, collapseRoot=True)
        nltk.treetransforms.chomsky_normal_form(tree_with_word, factor='right', horzMarkov=1, vertMarkov=1, childChar='p', parentChar='u')
        # ne ricavo una versione a stringa
        original_tree_word = tree_with_word.pprint().replace("\n", "")
        log.debug("Albero originale con le parole         : "+original_tree_word)
        # costruisco la sentence con parole e tag giusti
        word_with_tag_traslated = []
        # i tag e le parole
        real_pos = []
        real_word = []
        # per ogni produzione dell'albero
        production_of_tree = tree_with_word.productions()
        # per ogni parola nell'albero con le parole

        if len(tree_with_word.leaves()) < 25:

            for word in tree_with_word.leaves():
                # per ogni produzione nell'albero delle produzioni con i pos di google
                for production in production_of_tree:
                    # se la produzione è una produzione lessicale che produce la parola che voglio
                    if production.is_lexical() and production.rhs()[0] == word:
                        # allora popolo le tre liste
                        word_with_tag_traslated.append([unicodedata.normalize('NFKD', word).encode('ascii','ignore'), translate_single_tag(str(production.lhs()))])
                        real_word.append(unicodedata.normalize('NFKD', word).encode('ascii','ignore'))
                        real_pos.append(translate_single_tag(word_with_tag_traslated[-1][-1]))
                        break
            log.debug("Tag originali        : "+str(real_pos))
            log.debug("Parole della sentence: "+str(real_word))
            log.debug("Sentence con tag     : "+str(word_with_tag_traslated))
            # cerco di ricostruiscre l'albero con l'algoritmo di viterbi
            tree_founded = cky_and_viterbi(config, tree_with_google.leaves(), real_word, pcfg_cnf_grammar, ["S", "NP"])
            # se l'ho trovato
            if tree_founded != None:
                # lo denormalizzo
                tree_founded.un_chomsky_normal_form(expandUnary = False, childChar = "p", parentChar = "u")
                # ne ricavo una versione stringa
                tree_founded = tree_founded.pprint().replace("\n", "")
                log.info("Albero trovato: %s"+tree_founded)
                log.warning("Index_of_founded %s", str(counter_tested-1))
                # lo scrivo su file
                my_gold_file.write(original_tree+"\n")
                my_gold_file.flush()
                my_test_file.write(tree_founded+"\n")
                my_test_file.flush()
            else:
                log.info("Albero non trovato per sentence "+real_sentence)
                my_gold_file.write(original_tree+"\n")
                my_gold_file.flush()
                my_test_file.write("()\n")
                my_test_file.flush()
            # except Exception, e:
            #     print e
            #     my_gold_file.close()
            #     my_test_file.close()
            #     raw_input()

    log.info("Fine testing di CKY con Viterbi")
    my_gold_file.close()
    my_test_file.close()

def evaluate(evaluator_path, parameter_file, gold_file, test_file):
    """
    Lancia il valutatore evalb con parametro file gold e file test
    :param evaluator_path: string evalb path
    :param parameter_file: string parameter file
    :param gold_file: string gold_file
    :param test_file: string test_file
    :return:
    """
    subprocess.call([evaluator_path, "-p", parameter_file, gold_file, test_file])