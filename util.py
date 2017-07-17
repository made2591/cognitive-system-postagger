#!/usr/bin/env python
# -*- coding: utf-8 -*-
__author__ = 'Matteo'

import pickle, sys, nltk
from config import Config
from nltk import Tree, Nonterminal, Production
from string import punctuation

def get_pcfg(config, treebank_file, start_simbol, percentage_of_testing, limit = -1, treebank_modified_file = None):
    """
    Il seguente metodo carica una grammatica CF probabilistica a partire da un treebank.
    Prende come parametro un file contenente il treebank, lo start symbol della grammatica
    e la percentuale di file che dovrà essere usata come training o come testing (espressa come
    valore numero tra 0 e 1. Opzionalmente, è possibile limitare il treebank ad un numero
    di linee predefinito (e su di queste considerare la percentuale di training e di testing)
    e passare un file opzionale per l'esecuzione di CKY con Viterbi: il file di training
    per quanto riguarda l'estrazione della grammatica prevede la rimozione dei simboli terminali:
    per avere un riferimento a questi (alle sentence di cui vogliamo trovare i pos con viterbi)
    costruiamo anche un elenco di sentence con i terminali originali (e non rimpiazzati
    dal postag set di cui fa uso viterbi

    :param config: oggetto di configurazione
    :param treebank_file: string -> filename del treebank da cui estrarre la grammatica
    :param start_simbol: string -> start symbol della grammatica che vogliamo estrarre
    :param percentage_of_testing: float -> percentuale tra (0,1) di training (testing = 1-percentage_of_testing)
    :param limit: int -> numero di linee del file treebank_file da considerare
    :param treebank_modified_file: string -> filename del treebank modificato con i terminali = al postag set
                                   usato dall'algoritmo di viterbi
    :return:
    """
    # alias
    log = config.LOGGER
    # apro il file contenente il treebank
    f = open(treebank_file, "r")
    # leggo le linee
    training_set = f.readlines()
    # se è speicificato un limite del treebank specificato
    if limit != -1:
        training_set = training_set[:limit]
    # se è specificato il file del treebank modificato
    if treebank_modified_file != None:
        # leggo le linee del file modificato
        original_training_set = open(treebank_modified_file, "r").readlines()
        # se è specificato un limite limito anche il treebank modificato
        if limit != -1:
            original_training_set = original_training_set[:limit]
    # leggo tutte le sentence presenti nel file e le salvo in una lista
    all_sentences = list(training_set)
    # calcolo la percentuale di training e di testing
    percentage = int(len(training_set) * percentage_of_testing)
    # se è specificato un file, separto anche il testing delle frasi originali
    if treebank_modified_file != None:
        original_testing_set = original_training_set[percentage:]
    # separo il testing dal training set
    test_set = training_set[percentage:]
    # estraggo il training set
    training_set = training_set[:percentage]
    # creo una lista in cui concateno tutte le produzioni
    all_productions = []
    # creo una lista di simboli terminali
    terminal_symbols = []
    # per ogni frasi nel training set
    for l in training_set:
        # leggo l'albero
        t = Tree.fromstring(l.decode('utf-8'), remove_empty_top_bracketing=True)
        # collasso le produzioni unarie
        t.collapse_unary(collapsePOS=True, collapseRoot=True)
        # trasformo le produzioni in CNF
        nltk.treetransforms.chomsky_normal_form(t, factor='right', horzMarkov=1, vertMarkov=1, childChar='p', parentChar='u')
        # concateno le produzioni
        all_productions += t.productions()
        # concateno tutti i simboli terminali
        terminal_symbols += [i.lower().strip() for i in t.leaves()]
    # induco la grammatica probabilistica
    grammar = nltk.induce_pcfg(Nonterminal(start_simbol), all_productions)
    # se avevo specificato il file con il treebank originale
    if treebank_modified_file != None:
        # ritorno anche le sentence originali con i terminali = parole della sentence
        return grammar, test_set, training_set, all_sentences, terminal_symbols, original_testing_set
    # restituisco la grammatica, il testing set, il training set, tutte le frasi considerate e i simboli terminali
    return grammar, test_set, training_set, all_sentences, terminal_symbols


def transform_leaves_def(config, origin_file, destination_file):
    """
    Questo metodo prende un treebank e lo trasforma in un treebank che ha
    come simboli terminali i pos tag di google = al set di pos tag che restituisce viterbi
    :param origin_file: treebank file di origine
    :param destination_file: treebank file di destinazione
    :return: None
    """
    # alias
    log = config.LOGGER
    # apro il file di origine
    orig_sentences = open(origin_file, "r").readlines()
    # apro il file di destinazione
    transformed_tunn = open(destination_file, "w")
    # creo una stringa per concatenare i nuovi alberi
    transformed_sentences = ""
    # per ogni sentence nel file
    for sentence in orig_sentences:
        # separo per spazi
        sentence = sentence.split(" ")
        # indicizzo le parole
        index_of_word = 0
        # per ogni parola
        for word in sentence:
            # salvo la vecchia parola
            old_word = word
            log.debug("Parola originale  : "+str(old_word))
            # conto il numero di parentesi a destra e sinistra
            number_of_left_bracket = old_word.count("(")
            number_of_right_bracket = old_word.count(")")
            word = old_word.replace("(","").replace(")","")
            log.debug("Parola            : "+str(word))
            # creo un booleano
            term_leaf = False
            # se la paroal non è vuota, è upper o un simbolo di punteggiatura
            # e se la parola successiva è un terminale
            if len(word) > 0 and (word.isupper() or word in punctuation) and "(" in old_word and index_of_word < len(sentence) and ")" in sentence[index_of_word+1]:
                # salvo il terminale successivo
                old_terminal = sentence[index_of_word+1]
                log.debug("\tTerm originale  : "+str(old_terminal))
                number_of_left_bracket_term = old_terminal.count("(")
                number_of_right_bracket_term = old_terminal.count(")")
                # elimino le parentesi dal terminale
                terminal = sentence[index_of_word+1].replace("(","").replace(")","")
                log.debug("\tTerm            : "+str(old_terminal))
                # traduco il tag del penn con il tag di google corrispondente
                word = translate_single_tag(word)
                # concateno al pos del vecchio terminale una costante per distinguerlo dai pos del penn
                terminal = word+Config.POS_TAG_GOOGLE_CONSTANT
                # se c'erano parentesi le conatento a dx e sx
                if "(" in old_terminal:
                    s = ""
                    for i in range(0, number_of_left_bracket_term):
                        s += "("
                    sentence[index_of_word+1] = s+terminal
                if ")" in old_terminal:
                    s = ""
                    for i in range(0, number_of_right_bracket_term):
                        s += ")"
                    sentence[index_of_word+1] = terminal+s

                # mi salvo il fatto che ho modificato un terminale
                term_leaf = True

                log.debug("\tTerm trasformato  : "+str(sentence[index_of_word+1]))

            # stessa cosa per il non terminale
            if "(" in old_word:
                s = ""
                for i in range(0, number_of_left_bracket):
                    s += "("
                word = s+word

            if ")" in old_word:
                s = ""
                for i in range(0, number_of_right_bracket):
                    s += ")"
                word = word+s
            log.debug("Parola trasformata: "+str(word))

            # se non è l'ultimo elemento e non è stato modificato un terminale
            if old_word != sentence[-1] and not term_leaf:
                # concateno la nuova parola
                transformed_sentences += word+" "
            else:
                # diversamente concateno la vecchia parola
                transformed_sentences += old_word
                if index_of_word < len(sentence)-1:
                    transformed_sentences += " "
            index_of_word += 1

    transformed_tunn.write(transformed_sentences)
    transformed_tunn.flush()
    transformed_tunn.close()

def translate_single_tag(tag):
    """
    Preso un tag del penn risale alla giusta nomenclatura del pos tag di Google
    :param tag: string tag originale
    :return: string tag di Google
    """
    if tag in punctuation:
        tag = "."
    elif tag[0:2] == "NP":
        tag = "NOUN"
    elif tag[0:2] == "PP":
        tag = "PRT"
    elif tag[0:2] == "VP" or tag == "VAU" or tag == "VMA" or tag == "VMO" or tag == "VP":
        tag = "VERB"
    elif tag[0:3] == "ADJ":
        tag = "ADJ"
    elif tag[0:3] == "ADV":
        tag = "ADV"
    elif tag[0:3] == "ART":
        tag = "DET"
    elif tag[0:3] == "NOU":
        tag = "NOUN"
    elif tag[0:3] == "PRN":
        tag = "X"
    elif tag[0:3] == "PRO":
        tag = "PRON"
    elif tag[0:4] == "CONJ":
        tag = "CONJ"
    elif tag[0:4] == "DATE":
        tag = "NUM"
    elif tag[0:4] == "NUMR":
        tag = "NUM"
    elif tag[0:4] == "PRDT":
        tag = "PRON"
    elif tag[0:4] == "PREP":
        tag = "ADP"
    elif tag[0:5] == "PUNCT":
        tag = "."
    elif tag == "S+REDUC" or tag == "S-EXTPSBJ-" or tag == "SBAR" or tag == "SPECIAL" or tag == "VP+REDUC" or tag == "Vbar" or "-NONE-":
        tag = "X"

    return tag

def create_struct_from_csv_corpus(config, testing_mode = 0, capital_word_mode = 1):
    """
    Questo metodo fornisce una struttura dati a partire dal file specificato come parametro
    :param config_file: parametro di input per specificare il file sorgente
    :param rewrite: se vale 0, cerca un file in memoria da cui recuperare la struttura
                    se vale 1, sovrascrive, se trova, il file con il risultato della chiamata
    :return: un array contentente le informazioni codificate come segue

             All'indice k c'è la sentence k-esima, che altro non è che un
            [
                 array (frase) di array (parole) strutturati
                [
                    come segue
                    [
                        "parola_1"          :   posizione 0
                        "POS_Tag 2° colonna":   posizione 1
                        "POS_Tag 4° colonna":   posizione 2
                    ],
                    [
                        "parola_2"          :   posizione 0
                        "POS_Tag 2° colonna":   posizione 1
                        "POS_Tag 4° colonna":   posizione 2
                    ],
                    ...
                ],

                Il separatore di frase viene indicato con un array vuoto.
                [],
                ...
            ]

    """
    testing_mode = int(testing_mode)
    # Alias per il logger
    log = config.LOGGER

    # recupero il path del file contenente il training set (o il test set)
    origin_csv_file = config.TRAIN_FILE
    if testing_mode == 1:
        origin_csv_file = config.TEST_FILE

    # costruisco il dizionario che conterrà le frasi
    sentences = []
    # apro il file e lo processo
    log.info("Dimensione massima file del SO: %s", sizeof_fmt(sys.maxsize))
    with open(origin_csv_file, mode='r') as infile:
        lines = infile.readlines()
        # aggiungo una nuova frase
        sentences.append([])
        # per ogni linea del file
        row_count = 0
        word_count = 0
        for row in lines:
            row = row.decode('utf-8')
            row = row.split("\t")
            #log.debug("ROW: %s", str(row))
            row_count += 1
            # se ha più di 4 colonne, allora mi interessa (non è vuota)
            if len(row) > 4:
                # levo gli spazi di troppo
                row[1] = row[1].strip()
                if capital_word_mode == 1:
                    row[1] = row[1].lower()
                # introduco una miglioria: se la parola è la prima della frase
                # allora non considero la maiuscola, diversamente si.
                if len(sentences) > 0 and len(sentences[-1]) == 0 and capital_word_mode == 2:
                    row[1] = row[1].lower()
                word_count += 1
                # aggiungo all'ultima frase la parola strutturata
                # come un array di tre valori: parola, tag, tag_specializzato
                # log.debug("Ultima parola aggiunta: %s", str(sentences[-1]))
                sentences[-1].append([row[1], row[3], row[4]])
            else:
                # se la riga ha meno di 4 colonne, aggiungo un separatore:
                # significa che la frase è finita
                # log.debug("Ultima frase aggiunta: %s", str(sentences[-1]))
                sentences.append([])
                # log.debug("SENTENCE: %s", str(sentences[-2]))

        log.info("Numero di righe del file : %s", str(row_count))
        log.info("Numero di parole del file: %s", str(word_count))

    log.info("Separatori / Numero frasi: %s", str(int(len(sentences)-1)))
    log.info("Numero di parole medio per frase: ~ %s", str(int(float(word_count)/int(len(sentences)-1))))

    # terza euristica
    if capital_word_mode == 3:
        sentences = lower_or_upper(sentences)

    return sentences

def lower_or_upper(sentences):
    """
    Questo metodo sceglie per ogni sentence se considerare o meno la prima parola upper o lower
    :param sentences: list lista di sentences
    :return: lista di sentences modificata secondo l'euristica espressa nella relazione
    """
    already_checked = {}
    counter_sentence = 0
    for sentence in sentences:
        # print str(counter_sentence)+" di "+str(len(sentences))
        # prendo la prima parola, il suo contenuto e la capitalizzo con l'euristica secondo cui se compare
        # un maggior numero di volte lower o upper
        if len(sentence) > 0:
            if len(sentence[0]) > 0:
                word = sentence[0][0]
                if word.lower() not in already_checked.keys():
                    already_checked[word.lower()] = check_word_occurrence(word, sentences)
                sentence[0][0] = already_checked[word.lower()]
        counter_sentence += 1
    return sentences

def check_word_occurrence(word, sentences):
    """
    Questo metodo è un metodo d'appoggio che conta il numero di occorrenze della parola
    passata come parametro all'interno delle sentences, tutte le volte che non compare come prima parola:
    sulla base del numero di conteggi in cui questa compare lower o upper, restituisce la parola lower o upper
    :param word: string parola originale
    :param sentences: list lista di sentences
    :return: parola originale modificata secondo l'euristica
    """
    lower_count = 0
    upper_count = 0
    # per ogni frase
    for sentence in sentences:
        # per ogni parola in ogni frase (escluse le prime parole che sono ambigue per definizione)
        for a_word in sentence[1:]:
            a_word = a_word[0]
            # se la parola è uguale ignorando la capitalizzaione alla parola in esame
            if a_word.lower() == word.lower():
                # se compare maiuscola incremento il valore di comparse upper altrimenti lower
                if a_word.isupper(): upper_count += 1
                else: lower_count += 1
                # in ogni caso incremento il numero di comparse totali delle parola
    if lower_count > upper_count: return word.lower()
    else: return word

def get_single_words_distribution(sentences, states):
    """
    Calcola la distribuzione delle parole che compaiono una sola volta all'interno delle sentences
    e la restituisce sottoforma di un dizionario chiave = postag => valore = distribuzione di quel
    pos tag negli hapax legomena
    :param sentences:
    :param states:
    :return:
    """
    single_words = {}
    for s in sentences:
        for row in s:
            # salvo le parole che compaiono una sola volta e il loro pos tag
            if row[0] not in single_words.keys():
                single_words[row[0]] = row[Config.POS_TAG]
            else:
                single_words.pop(row[0], None)
    sws = {}
    for s in states:
        sws[s] = 0.0
        for w, pt in single_words.iteritems():
            if pt == s:
                sws[s] += 1.0
        sws[s] /= len(single_words.keys())

    return sws

def transition_probs(config, structured_corpus):

    """
    Prende come parametro una configurazione (l'oggetto istanziato per personalizzare i
    dei file in cui verranno salvate le computazioni, i logger, etc) e una struttura
    nella forma della risposta create_struct_from_csv_corpus

            [
                [
                    [
                        "parola_1": posizione 0
                        "POS_Tag 2° colonna": posizione 1
                        "POS_Tag 4° colonna": posizione 2
                    ],
                    [
                        "parola_2": posizione 0
                        "POS_Tag 2° colonna": posizione 1
                        "POS_Tag 4° colonna": posizione 2
                    ],
                    ...
                ],
                [],
                ...
            ]

    :param config: oggetto di configurazione
    :param structured_corpus: struttura restituita dal metodo create_struct_from_csv_corpus
    :param rewrite: se vale 0, cerca un file in memoria da cui recuperare la struttura
                    se vale 1, scrive e sovrascrive il file con il risultato della chiamata
    :return: restituisce una struttura del tipo

            {
                "POS_Tag" :
                        {
                            "count" : #numero #numero di comparse del tag nella struttura postag_corpora,
                            #contiene tutte le probabilità di transizione da un TAG precedente al successivo
                            POS_Tag : P_t1-i,
                            POS_Tag : P_t2-i,
                        },
                ...
            }

            che rappresenta la probabilità di transizione da un POS_Tag verso tutti i possibili altri

    """
    # Alias
    log = config.LOGGER

    # Costruisce la struttura dati come è definita nella documentazione sopra
    transition_probs = {}
    dummy_temp_tags = []
    for sentence in structured_corpus:
        for word in sentence:
            dummy_temp_tags.append(word[Config.POS_TAG])
    dummy_temp_tags = list(set(dummy_temp_tags))

    for actual_tag in dummy_temp_tags:
        transition_probs[actual_tag] = {}
        transition_probs[actual_tag]['count'] = 0
        for previous_tag in dummy_temp_tags:
            transition_probs[actual_tag][previous_tag] = 0
    log.info("INIT:         Creata struttura per tutti i POS Tag (semplici): %s", str(dummy_temp_tags))

    # calcola il campo 'count' per ogni POS_TAG nella struttura
    for sentence in structured_corpus:
        for word in sentence:
            transition_probs[word[Config.POS_TAG]]['count'] += 1
    log.info("STEP 1:       Calcolati i counter per tutti POS Tag (semplici)")

    # Trova le probabilità di transizione
    log.info("STEP 2        Inizio il calcolo delle probabilità di transizione")
    log.info("STEP 2.1:     Calcolo il numero di occorrenze per coppie di tag")
    number_of_sentence = 0
    total_sentences = int(len(structured_corpus))
    for sentence in structured_corpus:
        log.debug("                 Frase %s di %s: %s", str(number_of_sentence), str(total_sentences), str(sentence))
        word_position = 0
        for word in sentence:
            all_possible_postags = transition_probs.keys()
            for actual_tag in all_possible_postags:
                # per presa una parola in una frase controllo
                # se quella parola ha assegnato il tag che sto
                # prendendo in considerazione
                if actual_tag == word[Config.POS_TAG]:
                    # se si, allora se non è la prima parola della frase
                    # (in tal caso, non è preceduta da nessuna parola),
                    # andiamo ad incrementare la probabilità che dato il tag della parola precedente,
                    # la nostra actual word abbia il tag assegnatoli in questa frase
                    if word_position > 0:
                        transition_probs[word[Config.POS_TAG]][sentence[word_position-1][Config.POS_TAG]] += 1
            word_position += 1
        if len(sentence) > 0:
            number_of_sentence += 1

    log.info("STEP 2.2:     Calcolo la probabilità dividendo per il numero di occorenze")

    #TODO: DEVO NORMALIZZARE GLI ZERI QUI? NON CREDO
    for actual_tag in transition_probs.keys():
        for previous_tag in transition_probs[actual_tag].keys():
            if previous_tag != 'count':
                transition_probs[actual_tag][previous_tag] /= float(transition_probs[previous_tag]['count'])

    # stampa di debug dell'intera struttura
    log.info("STEP 2:       Finito calcolo delle probabilità di transizione")

    # restituisco la struttura
    return transition_probs

def emission_probs(config, transition_probs, structured_corpus):
    """
    Calcola le probabilità di emissione di una parola con dato tag
    :param config: oggetto di configurazione
    :param transition_probs: oggetto che rappresenta le probabilità di transizione
    :param structured_corpus: struttura restituita dal metodo create_struct_from_csv_corpus
    :return:
    """
    # Alias
    log = config.LOGGER

    log.info("INIT:         Creo struttura per tutte le parole")
    log.info("STEP 0:       Costruisco le strutture d'appoggio")
    # Costruisce la struttura dati come è definita nella documentazione sopra
    emission_probs = {}
    # Strutture d'appoggio
    dummy_words = []
    dummy_singles = []
    # Passo tutte le parole: mi salvo la parola (singola) in singles e
    # la parola con le info sui tag in un altra struttura (words)
    for sentence in structured_corpus:
        for word in sentence:
            dummy_words.append(word)
            dummy_singles.append(word[0])

    log.info("STEP 0:       Inizializzo i counter per ogni parola")
    # Passo le parole una volta sola e inizializzo la struttura di modo da non
    # dover far i controlli sotto per non reinizializzare i counter
    dummy_singles = sorted(list(set(dummy_singles)))
    for word in dummy_singles:
        emission_probs[word] = {}
        emission_probs[word] = {'count' : 0}
        for a_tag in transition_probs:
            emission_probs[word][a_tag] = 0

    log.info("STEP 1:       Conto il numero di assegnamenti per ogni tag per ogni parola e il numero di comparse della parola")
    for word in dummy_words:
        emission_probs[word[0]]['count'] += 1
        emission_probs[word[0]][word[Config.POS_TAG]] += 1

    log.info("STEP 2:       Calcolo le probabilità di emissione sfruttando i conteggi dei tag della struttura transition_probs")
    for word in emission_probs.keys():
        for emission_prob_for_tag in emission_probs[word].keys():
            if emission_prob_for_tag != 'count':
                emission_probs[word][emission_prob_for_tag] /= float(transition_probs[emission_prob_for_tag]['count'])

    return emission_probs

def save_obj_to_file(obj, file_name):

    """
    Questo metodo salva un oggetto obj nel file file_name e lo restituisce.
    Se ci sono errori, ritorna -1
    :param obj: oggetto da salvare
    :param file_name: file in cui salvare l'oggetto
    """
    try:
        filehandler = open(file_name, "wb")
        pickle.dump(obj, filehandler)
        filehandler.close()
    except Exception, e:
        print e
        return -1

def get_obj_from_file(file_name):
    """
    Questo metodo carica un oggetto dal file_name e lo restituisce.
    Se ci sono errori, ritorna -1
    :param file_name: file da cui caricare l'oggetto
    :return: oggetto caricato dal file
    """
    try:
        file = open(file_name,'rb')
        object_file = pickle.load(file)
        file.close()
        return object_file
    except Exception, e:
        print e
        return -1

def sizeof_fmt(num, suffix='B'):
    """
    Metodo per stampare correttamente la dimensione in byte
    :param num: numero
    :param suffix: suffisso
    :return: stringa human form
    """
    for unit in ['','Ki','Mi','Gi','Ti','Pi','Ei','Zi']:
        if abs(num) < 1024.0:
            return "%3.1f%s%s" % (num, unit, suffix)
        num /= 1024.0
    return "%.1f%s%s" % (num, 'Yi', suffix)