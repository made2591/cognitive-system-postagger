#!/usr/bin/env python
# -*- coding: utf-8 -*-
import hashlib
import os
from scipy.sparse import rand
from xsv import testing_svo_to_xsv

__author__ = 'Matteo'

from nltk import Tree
from viterbi import test_model, train_model
from cky import cky_and_viterbi, test_cky, evaluate, test_cky_with_viterbi
from util import get_pcfg, transform_leaves_def, get_obj_from_file, save_obj_to_file
from config import Config


def main(v, config):
    log = config.LOGGER
    os.system('clear')

    if v == 1:

        while True:
            print """
    ####################################################################################
    #######################     ESERCITAZIONE 1 - VITERBI     ##########################
    ####################################################################################
    ##                                                                                ##
    ## Scelta del pos tag:                                                            ##
    ##    1 => uso i pos tag della prima colonna   (Google).                          ##
    ##    2 => uso i pos tag della seconda colonna (Penn Tree Bank).                  ##
    ## Gestione delle unknown words:                                                  ##
    ##    1 => assumo distribuzione uniforme tra i vari tag.                          ##
    ##    2 => assumo distribuzione legata al training set.                           ##
    ##    3 => (migliore) assumo distribuzione uguale a quella fornita dalla          ##
    ##         distribuzione delle parole che compaiono 1 volta nel training set.     ##
    ## Gestione delle maiuscole e minuscole:                                          ##
    ##    1 => Prendo tutte le parole e faccio lower su tutte.                        ##
    ##    2 => Prendo tutte le parole e faccio lower solo su quelle a inizio frase.   ##
    ##    3 => (comportamento analogo a 2) risolvo l'ambiguità contando               ##
    ##         quante volte una parola a inizio frase compare in mezzo alla frase     ##
    ##         compare in mezzo alla frase capital e non se compare più volte         ##
    ##         non capital, allora la metto non capital, se no la metto capital.      ##
    ## ------------------------------------------------------------------------------ ##
    ## Riesegui training (develop mode - presentation mode):                          ##
    ##    0 => NO. Salta il training sfruttando il dump dell'ultimo training.         ##
    ##    1 => SI: Riscrive le strutture, ignorando le computazioni precedenti.       ##
    ## ------------------------------------------------------------------------------ ##
    ## Durante la selezione:                                                          ##
    ##    q => Torna al menù di scelta dell'esercitazione.                            ##
    ##                                                                                ##
    ####################################################################################
        """

            try:
                dummy = raw_input("POST da utilizzare      (default 1): ")
                if len(dummy) == 0: dummy = 1
                if dummy == "q": return
                dummy = int(dummy)
                Config.POS_TAG     = dummy
                if dummy != 1 and dummy != 2: raise BaseException

                dummy = raw_input("Gestione unknown w      (default 3): ")
                if len(dummy) == 0: dummy = 3
                if dummy == "q": return
                dummy = int(dummy)
                Config.UNKNOW_WORD = dummy
                if dummy != 1 and dummy != 2 and dummy != 3: raise BaseException

                dummy = raw_input("Gestione maiuscole      (default 2): ")
                if len(dummy) == 0: dummy = 2
                if dummy == "q": return
                dummy = int(dummy)
                Config.CAPITALMODE = dummy
                if dummy != 1 and dummy != 2 and dummy != 3: raise BaseException

                dummy = raw_input("Esegui il training (default 0 (NO)): ")
                if dummy.isalpha() and ("s" in dummy.lower() or "y" in dummy.lower()): dummy = 1
                elif dummy.isalpha() and "n" in dummy.lower(): dummy = 0
                elif dummy == "q": return
                elif len(dummy) == 0: dummy = 0
                elif dummy.isdigit(): dummy = int(dummy)
                train_mode = int(dummy)
                if train_mode != 0 and train_mode != 1: raise BaseException

                Config.STRING_VERSION = str(Config.POS_TAG)+"."+str(Config.UNKNOW_WORD)+"."+str(Config.CAPITALMODE)
                log.info("####################################################")
                log.info("## CONFIGURAZIONE (postag.unknown.capital): "+Config.STRING_VERSION+" ##")
                log.info("####################################################")
                log.info("=======>>>> INIZIO TRAINING <<<<=======")
                if train_mode == 1:
                    transition_probs, emission_probs, single_words_distribution, start_probs = train_model(config)
                    save_obj_to_file([transition_probs, emission_probs, single_words_distribution, start_probs], config.DUMP_TRAINING)
                else:
                    log.info("Carico il dump dell'ultimo training")
                    obj = get_obj_from_file(config.DUMP_TRAINING)
                    transition_probs, emission_probs, single_words_distribution, start_probs = obj[0], obj[1], obj[2], obj[3]
                    log.info("Dump caricato")
                log.info("=======>>>> FINE   TRAINING <<<<=======\n")
                log.info("=======>>>> INIZIO TESTING  <<<<=======")
                test_model(config, transition_probs, emission_probs, single_words_distribution)
                log.info("=======>>>> FINE   TESTING  <<<<=======\n")
                go_on()
            except Exception, e:
                print "Opzione non valida", e

    elif v == 2:

        while True:
            print """
    ####################################################################################
    ##########################    ESERCITAZIONE 2 - CKY    #############################
    ####################################################################################
    ##                                                                                ##
    ## Scelta della versione di CKY:                                                  ##
    ##    1  => CKY senza l'esecuzione di Viterbi.                                    ##
    ##    2  => CKY con l'esecuzione di Viterbi.                                      ##
    ## Percentuale di testing e training:                                             ##
    ##  ( 0.0, 1.0 )                                                                  ##
    ##       => Esempio: 0.9 = 90% training, 10% testing.                             ##
    ## Limita il numero di linee del file:                                            ##
    ##    0  => Prendo tutte le linee del file.                                       ##
    ##    >1 => Prendo x linee del file del treebank.                                 ##
    ## ------------------------------------------------------------------------------ ##
    ## Riesegui testing (develop mode - presentation mode):                           ##
    ##    0  => NO: Salta il testing e mostra i risultati sfruttando il dump          ##
    ##              dell'ultimo test.                                                 ##
    ##    1  => SI: Riesegue il testing, ignorando le computazioni precedenti e       ##
    ##              salvando i risultati su file per valutazioni successive.          ##
    ## ------------------------------------------------------------------------------ ##
    ## Durante la selezione:                                                          ##
    ##    q  => Torna al menù di scelta dell'esercitazione.                           ##
    ##                                                                                ##
    ####################################################################################
        """
            try:
                dummy = raw_input("Versione di CKY da eseguire    (default 1): ")
                if len(dummy) == 0: dummy = 1
                if dummy == "q": return
                dummy = int(dummy)
                cky_mode           = dummy
                if cky_mode != 0 and cky_mode != 1 and cky_mode != 2: raise BaseException

                dummy = raw_input("Percentuale training/testing (default 0.9): ")
                if len(dummy) == 0: dummy = 0.9
                if dummy == "q": return
                dummy = float(dummy)
                percentage = dummy

                dummy = raw_input("Limite il file di treebank (default 0 (NO): ")
                if len(dummy) == 0: dummy = -1
                if dummy == "q": return
                dummy = int(dummy)
                limit = dummy

                dummy = raw_input("Riesegui per il testing   (default 0 (NO)): ")
                if dummy.isalpha() and ("s" in dummy.lower() or "y" in dummy.lower()): dummy = 1
                elif dummy.isalpha() and "n" in dummy.lower(): dummy = 0
                elif dummy == "q": return
                elif len(dummy) == 0: dummy = 0
                elif dummy.isdigit(): dummy = int(dummy)
                testing_mode = int(dummy)
                if testing_mode != 0 and testing_mode != 1: raise BaseException

                if cky_mode == 1:

                    log.info("####################################################")
                    log.info("################### ESECUZIONE #####################")
                    log.info("####################################################")
                    log.info("Ottengo la grammatica")
                    pcfg_cnf_grammar, test_set, training_set, all_sentences, terminal_symbols = get_pcfg(config, config.TREEBANK_FILE, "S", percentage, limit)
                    log.info("=======>>>> INIZIO TESTING  <<<<=======")
                    log.info("Frasi di training: %s", str(len(training_set)))
                    log.info("Frasi di testing : %s", str(len(test_set)))
                    #test_set = [test_set[4], test_set[8], test_set[11], test_set[16], test_set[19], test_set[24], test_set[25], test_set[32], test_set[42], test_set[43], test_set[49], test_set[51], test_set[52], test_set[55], test_set[58], test_set[60], test_set[68], test_set[69], test_set[71], test_set[74], test_set[76], test_set[81], test_set[84], test_set[87], test_set[88], test_set[91], test_set[93], test_set[95], test_set[98], test_set[99], test_set[100], test_set[102], test_set[105], test_set[109]]
                    if testing_mode == 1:
                        log.info("Creo gold e test file...potrebbero volerci ORE")
                        test_cky(config, config.CKY_WITHOUT_VITERBI_GOLD_FILE, config.CKY_WITHOUT_VITERBI_TEST_FILE, test_set, pcfg_cnf_grammar)
                        log.info("File correttamente creati")
                    else:
                        log.info("Eseguo la valutazione")
                        evaluate(config.CKY_EVALUATOR_PATH, config.EVALUATOR_PARAMETER_FILE, config.CKY_WITHOUT_VITERBI_GOLD_FILE, config.CKY_WITHOUT_VITERBI_TEST_FILE)
                        log.info("Valutazione completata")
                    log.info("=======>>>> FINE   TESTING  <<<<=======\n")
                    go_on()
                elif cky_mode == 2:

                    log.info("####################################################")
                    log.info("################### ESECUZIONE #####################")
                    log.info("####################################################")

                    log.info("Trasformo il treebank con i POS di Google al posto dei terminali")
                    transform_leaves_def(config, config.TREEBANK_FILE, config.TREEBANKGOOGLE_FILE)
                    log.info("Treebank trasformato")
                    log.info("Ottengo la grammatica")
                    pcfg_cnf_grammar, test_set, training_set, all_sentences, terminal_symbols, original_testing_set = get_pcfg(config, config.TREEBANKGOOGLE_FILE, "S", percentage, limit, config.TREEBANK_FILE)
                    log.info("=======>>>> INIZIO TESTING  <<<<=======")
                    log.info("Frasi di training: %s", str(len(training_set)))
                    log.info("Frasi di testing : %s", str(len(test_set)))
                    if testing_mode == 1:
                        log.info("Creo gold e test file...potrebbero volerci ORE")
                        test_cky_with_viterbi(config, pcfg_cnf_grammar, test_set, config.CKY_WITH_VITERBI_GOLD_FILE, config.CKY_WITH_VITERBI_TEST_FILE, original_testing_set)
                        log.info("File correttamente creati")
                    else:
                        log.info("Eseguo la valutazione")
                        evaluate(config.CKY_EVALUATOR_PATH, config.EVALUATOR_PARAMETER_FILE, config.CKY_WITH_VITERBI_GOLD_FILE, config.CKY_WITH_VITERBI_TEST_FILE)
                        log.info("Valutazione completata")
                    log.info("=======>>>> FINE   TESTING  <<<<=======\n")
                    go_on()
                else:
                    print "Opzione non valida"
            except Exception, e:
                print "Opzione non valida", e

    elif v == 3:

        while True:
            print """
    ####################################################################################
    ######################    ESERCITAZIONE 3 - SVO to XSV    ##########################
    ####################################################################################
    ##                                                                                ##
    ## Scelta della frase:                                                            ##
    ##    0  => Inserisci una frase generica.                                         ##
    ##    1  => Prova l'esecuzione con qualche frase di testing.                      ##
    ##    2  => Seleziona una frase a caso dal treebank e prova con quella            ##
    ## Disegna GRAFICAMENTE entrambi gli alberi:                                      ##
    ##    0  => Non disegna nella finestra grafica nessun albero.                     ##
    ##    1  => Genera l'albero manipolato ma non quello originale.                   ##
    ##    2  => Genera l'albero originale e quello manipolato. (per proseguire        ##
    ##          oltre durante l'esecuzione, bisogna chiudere la finestra che          ##
    ##          visualizza l'albero originale disegnato e attendere la generazione    ##
    ##          dell'albero manipolato).                                              ##
    ## ------------------------------------------------------------------------------ ##
    ## Durante la selezione:                                                          ##
    ##    q  => Torna al menù di scelta dell'esercitazione.                           ##
    ##                                                                                ##
    ####################################################################################
        """
            try:
                dummy = raw_input("Scelta della frase   (default 1): ")
                if len(dummy) == 0: dummy = 1
                if dummy == "q": return
                dummy = int(dummy)
                sentence_choice    = dummy
                if sentence_choice != 0 and sentence_choice != 1 and sentence_choice != 2: raise BaseException

                dummy = raw_input("Disegna graficamente (default 2): ")
                if len(dummy) == 0: dummy = 2
                if dummy == "q": return
                dummy = int(dummy)
                graphic_draw       = dummy
                if graphic_draw != 0 and graphic_draw != 1 and graphic_draw != 2: raise BaseException

                log.info("####################################################")
                log.info("################### ESECUZIONE #####################")
                log.info("####################################################")

                if sentence_choice == 0:
                    sentence = raw_input("Inserisci una frase: ")
                    t, nt = testing_svo_to_xsv(config, sentence, t = None)
                    if t != None:
                        config.LOGGER.info("Frase SVO:"+str(t.leaves()))
                    if graphic_draw > 0 and t != None:
                        t.draw()
                    if nt != None:
                        config.LOGGER.info("Frase XSV:"+str(nt.leaves()))
                    if graphic_draw > 1 and nt != None:
                        nt.draw()
                    go_on()
                elif sentence_choice == 1:
                    sentences = ["Io mangio la mela molto velocemente .", "Luca e Paola corsero per raggiungere il treno in partenza .",
                                 "Infine i miei genitori consegnarono appositamente i soldi raccolti per riaprire la palestra ."]
                    for s in sentences:
                        log.info("Frase comune: "+str(s))
                        t, nt = testing_svo_to_xsv(config, s, t = None)
                        if t != None:
                            config.LOGGER.info("Frase SVO:"+str(t.leaves()))
                        if graphic_draw > 0 and t != None:
                            t.draw()
                        if nt != None:
                            config.LOGGER.info("Frase XSV:"+str(nt.leaves()))
                        if graphic_draw > 1 and nt != None:
                            nt.draw()
                        raw_input("Premi invio per continuare...")
                elif sentence_choice == 2:
                    tr = Tree.fromstring(open(config.TREEBANK_FILE).readlines()[rand(0, len(open(config.TREEBANK_FILE).readlines()))].decode('utf-8'))
                    sentence = tr.leaves()
                    log.info("Frase dal treebank: "+str(sentence))
                    t, nt = testing_svo_to_xsv(config, sentence, tr)
                    if tr != None:
                        config.LOGGER.info("Frase SVO:"+str(tr.leaves()))
                    if graphic_draw > 0 and tr != None:
                        tr.draw()
                    if nt != None:
                        config.LOGGER.info("Frase XSV:"+str(nt.leaves()))
                    if graphic_draw > 1 and nt != None:
                        nt.draw()
                    go_on()
                else:
                    raise TypeError('Scelta non valida')

                log.info("=======>>>> FINE   TESTING  <<<<=======\n")
            except Exception, e:
                print "Opzione non valida", e
    else:
        os.system('clear')
        print "Opzione non valida"


def go_on():
    raw_input("Premi invio per continuare...")
    os.system('clear')

def executor(config):
    os.system('clear')
    while True:
        print """
    ####################################################################################
    #######     Esercitazioni - Matteo Madeddu - Sistemi Cognitivi - Modulo 1    #######
    ####################################################################################
    ##                                                                                ##
    ##    1 => POS Tagger                                                             ##
    ##    2 => CKY                                                                    ##
    ##    3 => SVO to XSV                                                             ##
    ## ------------------------------------------------------------------------------ ##
    ##    q => Esci                                                                   ##
    ##                                                                                ##
    ####################################################################################
        """
        try:
            es = raw_input("Apri esercitazione n.: ")
            if len(es) == 0: es = 1
            if es == "q":
                break
            else:
                es = int(es)
                main(es, config)
                os.system('clear')
        except:
            print "\nOpzione non valida"
            go_on()
    os.system('clear')
    exit(1)

if __name__ == '__main__':
    # creo la configurazione
    config = Config()
    # imposto un alias per il logging
    log = config.LOGGER
    executor(config)