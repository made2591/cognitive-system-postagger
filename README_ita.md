##########################################################################################################
########################## README - ESERCITAZIONI MODULO 1 - MATTEO MADEDDU ##############################
##########################################################################################################

Contenuti:

    [1] Esecuzione
    [2] File principali
    [3] Consegne e relazioni
    [4] Note
    [5] Contatti

[1] - Esecuzione

    Per eseguire le esercitazioni: posizionarsi nella root
    ed eseguire:

    > python main.py

    Il programma nel main contiene delle stampe a video per
    scegliere quale esercitazione eseguire e con quale configurazione.

    Il programma di default stampa a video tramite un logger,
    che si limita a mostrare i passi principali dell'esecuzione dei vari
    metodi.

    Per cambiare le configurazioni inerenti le posizioni dei file
    o per abilitare il logging su file, o modificare la verbosità
    fare riferimento al file Config.

[2] - File principali

    I file che implementano i rispettivi core dei tre algoritmi implementati
    con le rispettive varianti sono:

        - viterbi.py
        - cky.py
        - xsv.py

[3] - Consegne e relazioni

    Nel cartelle consegne/ e relazioni/ c'è una copia del pdf delle consegne
    (senza le slide) e delle relazioni svolte riguardo le tre esercitazioni.
    Le relazioni contengono una descrizione dettagliata di cosa è stato implementato
    e dei risultati ottenuti più qualche riflessione su questi ultimi e alcuni
    tentativi di giustificazione di cosa è stato atteso e cosa no.

[4] - Note

    NOTA 1: alcuni dei parametri del Config non è necessario che siano
    esplicitati nella classe poiché vengono richiesti in fase di esecuzione.
    I parametri non richiesti e modificabili SOLO offline sono
    indicati all'interno del Config e riguardano principalmente
    path di dei file coinvolti nel dumping delle strutture dati o altro.

    NOTA 2: l'addestramento di Viterbi, con determinate configurazioni,
    può essere molto lungo. Nella cartella dump/ è presente un file

        v.1.3.2.viterbi_training.lst

    Questo file contiene il dump del training per Viterbi con configurazione
    1.3.2 (a runtime e nelle relazioni è spiegato cosa vogliono dire questi numeri).

    Di default, l'esecuzione di Viterbi viene fatta con questa configurazione
    e se non esplicitamente richiesto l'addestramento non viene rieseguito.
    Il testing viene invece effettuato ad ogni esecuzione sfruttando il file di
    training caricato / appena ricreato. Anche il testing non impiega poco (su 400 frasi)
    ma la sua esecuzione è scandita da messaggi che ne garantiscono l'avanzamento
    e comunque siamo nell'ordine di un paio di minuti al massimo.

    L'addestramento, invece, impiega molto ad essere eseguito, specialmente
    per a calcolare la distribuzione delle parole che compaiono una volta sola.
    In ogni caso, è sempre possibile sovrascrivere l'addestramento
    con un nuovo file: per ogni esecuzione con configurazione
    x.x.x se esplicitato, il programma sovrascrivere (se esiste) il dump della
    vecchia struttura d'addestramento per un futuro reload della stessa in un file
    x.x.x.viterbi_training.lst. E' quindi possibile mantenere più dump di configurazioni
    di Viterbi distinte nella stessa cartella.

    NOTA 3: l'algoritmo CKY che esegue facendo uso di Viterbi, parte dal presupposto
    che esista un dump del training di viterbi con configurazione 1.3.2. Vale a dire
    che per funzionare correttamente, l'algoritmo prevede la presenza del file

        v.1.3.2.viterbi_training.lst

    nella cartella dump/. Contrariamente all'addestramento di Viterbi, il metodo che si
    preoccupa di estrarre la PCFG in CNF dal training set della seconda esercitazione
    impiega pochi secondi a calcolare la grammatica. Quindi non è stato predisposto
    dump per questa struttura. Sono stati predisposti dei file di dump del testing
    di CKY: questo impiega DIVERSE ORE AD ESEGUIRE, e con determinate configurazioni
    può metterci anche tanti giorni ad ultimare i test. Per questo motivo, il metodo
    che sfrutta il programma di valutazione evalb, è in grado di recuperare dei dump
    di test preventivamente eseguiti presenti nella cartella dump.

    Precisamente, nella root di dump/ troviamo 4 file: questi sono il risultato dell'esecuzione
    di CKY con Viterbi e senza Viterbi su 110 sentence limitandosi alle sentence lunghe
    non più di 25 termini: ovvero circa 55 sentence. Rispettivamente, gld e tst sono i file
    gold e test (generati da CKY) con e senza viterbi (no e with)

    NOTA 4: nella cartella corpora/ troviamo i corpora usati e le modifiche attuate
    per esecuzioni specifiche (CKY + Viterbi).

[5] - Contatti

    Se qualcosa non va, matteo.madeddu@gmail.com