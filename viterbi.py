#!/usr/bin/env python
# -*- coding: utf-8 -*-
from config import Config
import util, operator

__author__ = 'Matteo'

def viterbi_alg(obs, states, start_p, trans_p, emit_p, single_words_distribution, mode):
    """
    Questo algoritmo implementa Viterbi e restituisce la probabilità e la sequenza con probabilità massima
    :param obs: list di words (sentence)
    :param states: possibili stati (postag)
    :param start_p: probabilità iniziali dei tag
    :param trans_p: probabilità di transizione
    :param emit_p: probabilità di emissione
    :return: tuple (prob, path)
    """

    # costruisco la matrice di Viterbi: per ogni cella dell'array, ho una colonna di valori.
    # questa colonna è rappresentata tramite un dizionario chiave valore con chiave ogni possibile stato
    # e valore la probabilità che la parola di indice = # di colonna nella frase abbia quel tag associato
    viterbi_matrix = [{}]
    # mantiene il path
    path = {}
    # inizializzo Viterbi (prima colonna): per ogni possibile stato (riga)
    for state in states:
        # se la prima parola non è presente nelle parole emesse dallo stato attuale
        if obs[0] not in emit_p.keys():
            # inserisco la probabilità di emissione di quel tag per quella parola come la probabilità
            # di emissione media per quel tag
            emit_p[obs[0]] = {}
            for s in states:
                if int(mode) == 1:
                    # PROBABILITA' UNIFORME
                    emit_p[obs[0]][s] = 1.0/len(states)
                elif int(mode) == 2:
                    # PROBABILITA' LEGATA AL TAG
                    emit_p[obs[0]][s] = start_p[s]
                elif int(mode) == 3:
                    # PROBABILITA' LEGATA AL TAG
                    emit_p[obs[0]][s] = single_words_distribution[s]

        # calcolo il valore di viterbi per quella parola (la prima, infatti viterbi_matrix[0][state])
        # come il prodotto tra la prob. iniziale dello stato state e la prob di emissione della prima parola obs[0]
        viterbi_matrix[0][state] = start_p[state] * emit_p[obs[0]][state]
        # aggiungo allo stato attuale lo stato attuale come precedente (puramente algoritmico)
        path[state] = [state]
    # per ogni altra parola nella frase
    for t in range(1, len(obs)):
        #print print_viterbi_matrix(obs, viterbi_matrix)
        # aggiungo una colonna per la parola t-esima della frase
        viterbi_matrix.append({})
        # costruisco un temporaneo per tenere traccia del path più probabile di stati fino a quella parola
        newpath = {}
        # per ogni stato (postag o riga) all'interno dei possibili stati
        for state in states:
            # se la prima parola non è presente nelle parole emesse dallo stato attuale
            if obs[t] not in emit_p.keys():
                # inserisco la probabilità di emissione di quel tag per quella parola come la probabilità
                # di emissione media per quel tag
                emit_p[obs[t]] = {}
                for s in states:
                    if int(mode) == 1:
                        # PROBABILITA' UNIFORME
                        emit_p[obs[t]][s] = 1.0/len(states)
                    elif int(mode) == 2:
                        # PROBABILITA' LEGATA AL TAG
                        emit_p[obs[t]][s] = start_p[s]
                    elif int(mode) == 3:
                        # PROBABILITA' LEGATA AL TAG
                        emit_p[obs[t]][s] = single_words_distribution[s]

            # calcolo la tupla (probabilià di occorrenza, stato_precedente) massima tra quelle prodotte considerando ogni il prodotto
            # tra i 3 fattori:
            # - il valore di viterbi della parola precedente a quella attuale;
            # - la probabilità di transizione tra lo stato di quella parola e lo stato attuale;
            # - la probabilità di emissione dello stato attuale per la parola corrente;
            (max_prob, max_state) = max(
                (viterbi_matrix[t-1][previous_state] * trans_p[previous_state][state] * emit_p[obs[t]][state], previous_state) for previous_state in states
            )
            # ottengo la coppia massima e allora so che la probabilità per lo stato attuale per la parola attuale è data
            # dal prodotto massimo calcolato sopra
            viterbi_matrix[t][state] = max_prob
            # per lo stato che sto considerando, aggiungo in testa alla lista di transizione più probabile lo stato che sto considerando
            newpath[state] = path[max_state] + [state]
        # il path vecchio è sovrascritto dal nuovo dizionario (che per ogni stato, mantiene la sequenza più probabile
        # di transizioni a partire da quello stato)
        path = newpath
    # se l'osservazione è fatta di una sola parola inizializzo una variabile a 0
    index_of_word = 0
    # diversamente inizializzo la stessa variabile all'indice dell'ultima parola
    if len(obs) != 1: index_of_word = t
    # stampo la matrice di Viterbi
    # print_viterbi_matrix(obs, viterbi_matrix)
    # scelgo come path da restituire quello che ha come stato lo stato per cui il valore
    # di Viterbi è massimo sul'ultima parola: tanto la probabilità è crescente, quindi
    # poiché è un prodotto di probabilità sappiamo che il path per il percorso massimo
    # è identificato dalla chiave di dizionario per cui la probabilità è massima
    (max_prob, state) = max((viterbi_matrix[index_of_word][state], state) for state in states)
    # restituisco la probabilità della sequenza e il path percorso
    # index_of_state = 1
    # for s in path[state][1:]:
    #     if path[state][index_of_state-1] == "DET" and s != "NOUN":
    #         path[state][index_of_state-1] = "PRON"

    return (max_prob, path[state])

def train_model(config):
    """
    Esegue il training del modello per Viterbi
    :param config: configurazione
    :return:
    """
    # Alias
    log = config.LOGGER
    # Costruisco una struttura appropriata per rappresentare il corpora
    log.info("\t1 di 3) Costruisco una struttura appropriata per rappresentare il TRAINING SET")
    train_corpus_struct = -1
    if config.CAPITALMODE == 1:
        train_corpus_struct = util.create_struct_from_csv_corpus(config, testing_mode= 0, capital_word_mode=1)
    elif config.CAPITALMODE == 2:
        train_corpus_struct = util.create_struct_from_csv_corpus(config, testing_mode= 0, capital_word_mode=2)
    elif config.CAPITALMODE == 3:
        train_corpus_struct = util.create_struct_from_csv_corpus(config, testing_mode= 0, capital_word_mode=3)
    # Calcolo le probabilità di transizione
    log.info("\t2 di 3) Calcolo le probabilità di transizione")
    transition_probs = util.transition_probs(config, train_corpus_struct)
    # Calcolo le probabilità di emissione
    log.info("\t3 di 3) Calcolo le probabilità di emissione")
    emission_probs = util.emission_probs(config, transition_probs, train_corpus_struct)
    single_words_distribution = []

    if config.UNKNOW_WORD == 3:
        log.info("\t3 di 3) Passo aggiuntivo: calcolo la distribuzione degli hapax legomena che userò per parole nuove")
        single_words_distribution = util.get_single_words_distribution(train_corpus_struct, transition_probs.keys())

    # calcolo le probabilità di start
    start_probs = {}
    for t in transition_probs.keys():
        start_probs[t] = transition_probs[t]['count'] / float(len(emission_probs))

    # Restituisco le prime e le seconde
    return transition_probs, emission_probs, single_words_distribution, start_probs

def test_model(config, transitions_probs, emissions_probs, single_words_distribution):
    """
    Esegue il testing del modello per viterbi
    :param config: configurazione
    :param transitions_probs: probabilità di transizione
    :param emissions_probs: probabilità di emissione
    :param single_words_distribution: distribuzione hapax legomena
    :return:
    """
    # Alias
    log = config.LOGGER

    log.info("\t1 di 3) Costruisco una struttura appropriata per rappresentare il TEST SET")
    test_corpus_struct = -1
    if config.CAPITALMODE == 1:
        test_corpus_struct = util.create_struct_from_csv_corpus(config, testing_mode= 1, capital_word_mode=1)
    elif config.CAPITALMODE == 2:
        test_corpus_struct = util.create_struct_from_csv_corpus(config, testing_mode= 1, capital_word_mode=2)
    elif config.CAPITALMODE == 3:
        test_corpus_struct = util.create_struct_from_csv_corpus(config, testing_mode= 1, capital_word_mode=3)
    # Conto il numero di parole taggate
    total_pos_tagged = 0.0
    # Conto il numero di parole taggate correttamente
    total_pos_correct = 0.0
    # Indice della sentence attualmente analizzata
    counter_senteces = 0
    total_senteces = int(len(test_corpus_struct)-1)

    # errori
    error = {}
    for tag1 in transitions_probs:
        error[tag1] = {}
        for tag2 in transitions_probs:
            error[tag1][tag2] = {'total': 1, 'wrong': 0}

    log.info("\t2 di 3) Inizio la fase di test del POS Tagger sul test SET")
    log.info("\t        Numero di frasi nel test set: %s", str(total_senteces))

    # Per ogni frase del corpora
    for sentence in test_corpus_struct:

        if len(sentence) > 0:
            # tengo il conto della sentence attualmente testata
            counter_senteces += 1
            # calcolo la percentuale di avanzamento
            percentage = float((float(counter_senteces) / float(total_senteces)) * 100.0)
            if(float(percentage % 10) == 0.0):
                log.info("\t        Percentuale avanzamento: %s%% (%s di %s)", str(int(percentage)), str(counter_senteces), str(total_senteces))
            log.debug("\t\t        Eseguo Viterbi su frase numero: %s di %s", str(counter_senteces), str(total_senteces))

            ######################################################################
            ############### ESEGUO VITERBI sulla sentence attuale ################
            result = exec_viterbi_and_check(config, sentence, transitions_probs, emissions_probs, single_words_distribution)
            ######################################################################
            ######################################################################
            # loggo i risultati di viterbi
            dummy_temp_sentence = ""
            for dummy_temp in sentence:
                dummy_temp_sentence += dummy_temp[0]+" "
            log.debug("\t\t        Frase: %s", dummy_temp_sentence)
            log.debug("\t\t        Probabilità sequenza         : %s", str(result[0]))
            log.debug("\t\t        Sequenza più probabile       : %s", str(result[1]))
            log.debug("\t\t        Sequenza corretta            : %s", str(result[2]))

            # counter per ciclare su tag corretti
            dummy_correct_tag_position = 0
            # controllo la correttezza dei tag guardando i gold
            for predicted_tag in result[1]:
                # se un tag è stato previsto correttamente
                if predicted_tag == result[2][dummy_correct_tag_position]:
                    # incremento il numero di tag corretti
                    total_pos_correct += 1
                else:
                    if result[2][dummy_correct_tag_position] not in transitions_probs.keys():
                        log.error("ATTENZIONE: tag corretto %s non presente nel training set!!!!!!", result[2][dummy_correct_tag_position])
                        log.error("TAG NEL TRAINING SET: %s", str(transitions_probs.keys()))
                    log.debug("\t\t\t        Errore predizione        : '%s' predetto come %s invece di %s", sentence[dummy_correct_tag_position][0], predicted_tag, result[2][dummy_correct_tag_position])
                    error[predicted_tag][result[2][dummy_correct_tag_position]]['wrong'] += 1
                error[predicted_tag][result[2][dummy_correct_tag_position]]['total'] += 1
                # incremento il numero di tag predetti in totale
                total_pos_tagged += 1
                dummy_correct_tag_position += 1
            if(percentage == 0):
                log.info("\t\t        Percentuale di successo      : %s", str(total_pos_correct/total_pos_tagged*100))
            else:
                log.debug("\t\t        Percentuale di successo      : %s", str(total_pos_correct/total_pos_tagged*100))

    log.info("\t3 di 3) Fine fase di testing. Risultati:")
    log.info("          Frasi analizzate: %s", str(total_senteces))
    log.info("          Parole predette : %s", str(int(total_pos_tagged)))
    log.info("          Parole corrette : %s", str(int(total_pos_correct)))
    log.info("          Percentuale succ: %s", str(total_pos_correct/total_pos_tagged*100))
    get_first_n_common_error(log, error, 10)

def get_first_n_common_error(log, error, n):
    """
    Trova i primi n errori più comuni
    :param log: logger
    :param error: dizionario degli errori commessi
    :param n: numero di errori da considerare nella classifica
    :return: none
    """
    # errori
    errors = {}
    # per ogni tag nel dizionario degli errori
    for tag1 in error:
        # per ogni tag con cui è stato predetto
        for tag2 in error:
            # calcolo la quantità di errori commessi sul numero di previsioni fatte per quella coppia TAGREALE -> PREDIZIONE
            errors[tag1+" predetto in modo errato come "+tag2] = float(error[tag1][tag2]['wrong']) / float(error[tag1][tag2]['total'])
    # ordino gli errori
    errors = sorted(errors.items(), key=operator.itemgetter(1))
    # inverto la il dizionario
    errors = errors[::-1]
    number_of_error = 0
    # per ogni errore nella lista
    for e in errors:
        # se fa parte dei primi n
        if number_of_error < n:
            # loggo l'informazione con la percentuale di errore per quell'errore specifico
            log.info("Errore: %s con percentuale di errore %s", e[0], e[1])
        else:
            # mi fermo
            break
        # incremento il numero di errori della lista
        number_of_error += 1

def print_viterbi_matrix(obs, viterbi_matrix):
    """
    Questo metodo stampa una matrice di viterbi
    :param obs: list di words (una sentence)
    :param viterbi_matrix: list (matrice di Viterbi, lista di dizionari)
    :return: str una stringa che rappresenta la matrice di viterbi
    """
    # tabulazione di prima riga
    viterbi = "\t\t"
    # indice della parola in esame
    index_of_word = 0
    # per ogni parola
    for word in obs:
        # concateno il contenuto della parola e il suo indice
        viterbi += ("%15s" % (word+" ("+str(index_of_word)+")"))
        index_of_word += 1

    # concateno un nuova linea
    viterbi += "\n"
    # per ogni colonna riga della matrice di viterbi
    for array_column in viterbi_matrix[0]:
        # stampo il postag che quella riga rappresenta
        viterbi += ("%5s:  " % array_column)
        # per ogni parola nella matrice di viterbi (dalla prima all'ultima)
        for column_index in range(1, len(viterbi_matrix)):
            # concateno la probabilità che quella parola sia emessa con quel tag
            probs_column = viterbi_matrix[column_index]
            viterbi += ("%15f" % probs_column[array_column])
        # concateno un nuova linea
        viterbi += "\n"
    return viterbi


# Questa funzione è un wrapper per l'esecuzione di viterbi
def exec_viterbi_and_check(config, observed_sentence, transition_probs, emission_probs, single_words_distribution):
    """
    Wrapper per l'esecuzione di Viterbi
    :param config: configurazione
    :param observed_sentence: sentence
    :param transitions_probs: probabilità di transizione
    :param emissions_probs: probabilità di emissione
    :param single_words_distribution: distribuzione hapax legomena
    :return:
    """
    #TODO observed_sentence[0][0] = observed_sentence[0][0].lower()
    #observed_sentence[0][0] = observed_sentence[0][0].lower()

    # Alias
    log = config.LOGGER
    # Creo l'observation a partire dall'elenco di parole (strutturate con anche i tag)
    observation = []
    for w in observed_sentence:
        observation.append(w[0])
    tuple(observation)

    # Genero i possibili stati del grafo di transizione
    states = tuple(transition_probs.keys())

    # Calcolo la probabilità di partenza di ogni tag
    # (il numero di volte che compare diviso il numero di parole presenti nel corpora)
    start_probs = {}
    for t in transition_probs.keys():
        start_probs[t] = transition_probs[t]['count'] / float(len(emission_probs))

    # Recupero i tag
    obs_correct_pos = []
    for w in observed_sentence:
        obs_correct_pos.append(w[Config.POS_TAG])

    viterbi_result = []
    ###############################################################################
    #################### ESEGUO VITERBI sulla sentence attuale ####################
    if config.UNKNOW_WORD == 1:
        viterbi_result = viterbi_alg(observation, states, start_probs, transition_probs, emission_probs,
                                     single_words_distribution, 1)
    elif config.UNKNOW_WORD == 2:
        viterbi_result = viterbi_alg(observation, states, start_probs, transition_probs, emission_probs,
                                     single_words_distribution, 2)
    elif config.UNKNOW_WORD == 3:
        viterbi_result = viterbi_alg(observation, states, start_probs, transition_probs, emission_probs,
                                     single_words_distribution, 3)
    ###############################################################################
    ###############################################################################

    results = []
    for elem in viterbi_result:
        results.append(elem)
    results.append(obs_correct_pos)
    return results