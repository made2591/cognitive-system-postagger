#!/usr/bin/env python
# -*- coding: utf-8 -*-
__author__ = 'Matteo'
import logging

#########################################
############ CONFIGURAZIONI #############
#########################################

class Config(object):
    """Classe di configurazione con path dei file, versione del pos_tagger"""
    POS_TAG_GOOGLE_CONSTANT = "_POSTAG"

    def __init__(self, pt = 1, cp = 3, uw = 2):

        #####################################################################
        #~~~~~~~~~~~~~~~~~~~~~~~~~ NON MODIFICARE ~~~~~~~~~~~~~~~~~~~~~~~~~~#
        ################   PARAMETRI IMPOSTATI A RUNTIME   ##################
        #####################################################################

        # 1 => uso i pos tag della prima colonna   (Google)
        # 2 => uso i pos tag della seconda colonna (Penn Tree Bank)
        self.POS_TAG = pt
        # Gestione delle maiuscole e minuscole
        # 1 => Prendo tutte le parole e faccio lower su tutte
        # 2 => Prendo tutte le parole e faccio lower solo sulle prime parole (se sono a inizio, sono capital)
        # 3 (comportamento analogo a 2) => risolvo l'ambiguità contando quante volte una parola a inizio frase compare in mezzo alla frase capital e non
        # se compare più volte non capital, allora la metto non capital, se no la metto capital.
        self.CAPITALMODE = cp
        # Gestione delle unknown words
        # 1 => assumo distribuzione uniforme tra i vari tag
        # 2 => assumo distribuzione legata al training set
        # 3 (migliore) => assumo distribuzione uguale a quella fornita dalla distribuzione delle parole che compaiono una volta sola nel training set.
        self.UNKNOW_WORD = uw

        #####################################################################
        ################ PATH DEI FILE USATI DAL PROGRAMMA ##################
        #####################################################################

        # Cartella che contiene i corpora
        self.DIR_FILE                           = "./corpora/"

        # File CSV recuperati da https://uni-dep-tb.googlecode.com/svn/trunk/universal_treebanks_v2.0.tar.gz
        self.TEST_FILE                          = self.DIR_FILE+"it-universal-test.conll"
        self.DEV_FILE                           = self.DIR_FILE+"it-universal-dev.conll"
        self.TRAIN_FILE                         = self.DIR_FILE+"it-universal-train-and-dev.conll"
        self.TRAINDEV_FILE                      = self.DIR_FILE+"it-universal-train-dev.conll"

        self.TREEBANK_FILE                      = self.DIR_FILE+"tut-clean-simple.penn"
        self.TREEBANKGOOGLE_FILE                      = self.DIR_FILE+"tut-clean-simple-google.penn"

        self.POS_TAG_GOOGLE_CONSTANT            = self.POS_TAG_GOOGLE_CONSTANT

        # Cartella che contiene i DUMP delle computazioni
        self.DIR_OBJ                            = "./dump/"

        # File con i DUMP delle computazioni
        self.DUMP_TRAINING                      = self.DIR_OBJ+"v."+str(self.POS_TAG)+"."+str(self.CAPITALMODE)+"."+str(self.UNKNOW_WORD)+".viterbi_training.lst"

        self.PCFG                               = self.DIR_OBJ+"pcfg_cnf_grammar.pcfg"

        self.CKY_WITHOUT_VITERBI_GOLD_FILE      = self.DIR_OBJ+"cky_no_viterbi.gld"
        self.CKY_WITHOUT_VITERBI_TEST_FILE      = self.DIR_OBJ+"cky_no_viterbi.tst"
        self.CKY_WITH_VITERBI_GOLD_FILE         = self.DIR_OBJ+"cky_with_viterbi.gld"
        self.CKY_WITH_VITERBI_TEST_FILE         = self.DIR_OBJ+"cky_with_viterbi.tst"

        self.CACHE_SVO_EXEC                     = self.DIR_OBJ+"cache_svo_xsv.dict"

        # Cartella che contiene i DUMP delle computazioni
        self.EVALUATOR_DIR                      = "./evaluator/"
        self.EVALUATOR_PARAMETER_FILE           = self.EVALUATOR_DIR+"evaluator_parameter.prm"
        self.CKY_EVALUATOR_PATH                 = self.EVALUATOR_DIR+"evalb"

        #####################################################################
        ################ CONFIGURAZIONE DEL LOGGER DI POS ###################
        #####################################################################

        # Configurazione del logger (livello di verbosità generico, console, file)
        self.CONSOLE_LOG_LEVEL                  = logging.INFO
        self.FILE_LOG_LEVEL                     = logging.INFO
        # Abilitare log su file
        self.ENABLE_FILE_LOG                    = False
        # Cartella che contiene i DUMP delle computazioni
        self.DIR_LOGS                            = "./logs/"
        # Nome file di log
        self.LOGGER_FILE_NAME                   = self.DIR_LOGS+'verbosity_'+str(self.FILE_LOG_LEVEL).lower()+'.log'

        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        #~~~~~~~~~~~ NON MODIFICARE (configurazioni statiche) ~~~~~~~~~~~~~~~#
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

        console_log_level = self.CONSOLE_LOG_LEVEL
        file_log_level = self.FILE_LOG_LEVEL
        enable_log_to_file = self.ENABLE_FILE_LOG
        # set up logging to file
        logging.basicConfig(
             filename = self.LOGGER_FILE_NAME,
             level = file_log_level,
             format= '[%(asctime)s] {%(pathname)s:%(lineno)d} %(levelname)s - %(message)s',
             datefmt='%H:%M:%S'
         )
        # set up logging to console
        console = logging.StreamHandler()
        console.setLevel(console_log_level)
        # set a format which is simpler for console use
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        console.setFormatter(formatter)
        # add the handler to the root logger
        logging.getLogger('').addHandler(console)

        # Aggiungo il logger all'oggeto di configurazione
        self.LOGGER = logging.getLogger(__name__)