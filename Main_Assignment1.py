# Importa os per percorsi e manipolazione file.
import os
# Importa OpenCV per leggere immagini se in modalità prof test.
import cv2
# Importa numpy per arrays.
import numpy as np
# Importa le funzioni dai file che hai creato. Modula il codice e lo tiene in ordine.
from Carico_file import carica_dati_da_dataset
from Funzioni import stima_modello_geometrico, valuta_prestazioni_medie, salva_risultati_allineamento, stima_coarse_to_fine

# DEFINIZIONE DELLE VARIABILI GLOBALI (COSTANTI)
# Cartella principale dove risiedono i dataset.
CARTELLA_IMMAGINI = "DATASET" 
# File CSV che contiene il Ground Truth (verità di base).
FILE_CSV = "DATASET/GT.csv" 

# Imposta in quanti "pacchetti" (bin) dividere le intensità luminose per calcolare l'istogramma (64, 128 o 256).
BIN_CORRENTI = 64

# Imposta il nome dell'algoritmo che useremo per ottimizzare la stima.
METODO_CORRENTE = 'Powell'

# Permette di scegliere rapidamente se applicare un filtro o usare i dati grezzi.
FILTRO_CORRENTE = 'nessuno'

# Flag booleano: se True attiva l'algoritmo per scalare l'immagine al 50% prima di fare la stima.
USA_COARSE_TO_FINE = False

# Determina su quale set lavorare. True esegue il test finale, False lancia il set di validazione.
ESEGUI_SU_TEST_SET = True

# --- MODALITÀ DI TEST LIBERO (PROF) ---
# Se impostata a True, bypassa la lettura del file CSV.
MODALITA_PROF_TEST = False
# Specifica la cartella da cui leggere liberamente le immagini ignorando i vincoli dell'assignment.
CARTELLA_DA_TESTARE = "DATASET/Prova" 


# BIFORCAZIONE PRINCIPALE DEL PROGRAMMA
if MODALITA_PROF_TEST:
    """
    A COSA SERVE QUESTO BLOCCO: Esegue il codice su immagini casuali fornite dal professore senza CSV.
    PERCHÉ SI USA: Evita crash del programma (FileNotFoundError) se chi corregge vuole usare un suo dataset 
    segreto per vedere se il tuo codice è robusto e funzionante.
    """
    
    # Crea la cartella per ospitare i risultati del prof, se non esiste.
    if not os.path.exists("IMMAGINI_DI_PROVA"):
        os.makedirs("IMMAGINI_DI_PROVA")

    # Legge tutto il contenuto della cartella e filtra solo i file immagine (.png o .jpg).
    file_presenti = [f for f in os.listdir(CARTELLA_DA_TESTARE) if f.endswith('.png') or f.endswith('.jpg')]
    # Filtra ulteriormente estraendo solo i file che finiscono in "R", ovvero le immagini fisse (Reference).
    file_reference = [f for f in file_presenti if f.endswith('R.png') or f.endswith('R.jpg')]

    # Controllo di sicurezza, ferma il processo se non trova nulla.
    if not file_reference:
        print("Nessuna immagine Reference trovata")
    
    # Cicla su tutte le immagini Reference trovate.
    for nome_r in file_reference:
        # Crea dinamicamente il nome del file Moving sostituendo la lettera 'R' con la 'T'.
        # es: immagine_R.png -> immagine_T.png
        nome_m = nome_r[:-5] + 'T' + nome_r[-4:]
        
        # Crea i percorsi completi.
        percorso_r = os.path.join(CARTELLA_DA_TESTARE, nome_r)
        percorso_m = os.path.join(CARTELLA_DA_TESTARE, nome_m)

        # Controlla se la Moving generata esiste realmente.
        if os.path.exists(percorso_m):
            # Carica in grigio le immagini.
            img_r = cv2.imread(percorso_r, cv2.IMREAD_GRAYSCALE)
            img_m = cv2.imread(percorso_m, cv2.IMREAD_GRAYSCALE)

            # Lancia l'algoritmo usando la multirisoluzione oppure la stima base.
            if USA_COARSE_TO_FINE:
                parametri_trovati = stima_coarse_to_fine(img_r, img_m, METODO_CORRENTE, BIN_CORRENTI, FILTRO_CORRENTE, 0.5)
            else:
                parametri_trovati = stima_modello_geometrico(img_r, img_m, METODO_CORRENTE, BIN_CORRENTI, FILTRO_CORRENTE)

            # Rimuove il "_R.png" dal nome, rendendo il salvataggio dei file pulito.
            prefisso = nome_r[:-5] 
            # Applica i parametri, muove l'immagine e crea file su disco.
            salva_risultati_allineamento(img_r, img_m, parametri_trovati, prefisso, cartella="IMMAGINI_DI_PROVA")
        else:
            print(f" Errore immagini non trovate o non corrispondono i noi")
            
    print("\nEsecuzione terminata. Cartella creata con successo. Dentro troverà le immagini")


else:
    """
    A COSA SERVE QUESTO BLOCCO: È il flusso di lavoro standard dell'Assignment, esegue Validation o Test
    appoggiandosi ai dati forniti dal professore (GT.csv).
    """
    
    # Chiama la funzione su Carico_file per riempire due liste enormi con i dati del CSV e le matrici delle immagini.
    dati_validazione, dati_test = carica_dati_da_dataset(CARTELLA_IMMAGINI, FILE_CSV)
    
    # Usa un operatore ternario (un if/else in riga). 
    # Decide quale set di immagini passare nel ciclo in base alla variabile booleana all'inizio.
    dataset_da_usare = dati_test if ESEGUI_SU_TEST_SET else dati_validazione
    # Salva una stringa di testo solo per stampare nel terminale in che fase siamo.
    nome_fase = "TEST" if ESEGUI_SU_TEST_SET else "VALIDAZIONE"

    # Stampa a schermo con cosa sta calcolando, per controllo in console.
    print(f"Parametri: {BIN_CORRENTI} bin, Metodo: {METODO_CORRENTE}")
    print(f"Filtro: {FILTRO_CORRENTE}, PiramideMultirisoluzione: {USA_COARSE_TO_FINE}")

    # Inizializza lista che raccoglierà i 3 parametri finali trovati per ogni immagine, servirà per fare la media.
    lista_delle_stime = []

    # Ciclo principe: prende un pacchetto alla volta (una coppia di immagini) dal dataset scelto.
    for coppia in dataset_da_usare:
        # Estrae l'immagine Reference dal dizionario.
        immagine_r = coppia['immagine_riferimento']
        # Estrae l'immagine Moving dal dizionario.
        immagine_m = coppia['immagine_moving']
        
        # Decide se eseguire l'algoritmo veloce su immagini dimezzate o normale in base ai settings in alto.
        if USA_COARSE_TO_FINE:
            parametri_trovati = stima_coarse_to_fine(
                immagine_riferimento=immagine_r, immagine_moving=immagine_m,
                metodo_scelto=METODO_CORRENTE, numero_bin=BIN_CORRENTI,
                tipo_preproc=FILTRO_CORRENTE, fattore_scala=0.5
            )
        else:
            # Chiama la funzione principale per allineare geometricamente la coppia.
            # Questo processo ferma il ciclo per svariati secondi (la magia dell'ottimizzazione).
            parametri_trovati = stima_modello_geometrico(
                immagine_riferimento=immagine_r, immagine_moving=immagine_m,
                metodo_scelto=METODO_CORRENTE, numero_bin=BIN_CORRENTI,
                tipo_preproc=FILTRO_CORRENTE
            )
            
        # Aggiunge i 3 parametri all'elenco per poi calcolare la MAE in valuta_prestazioni_medie.
        lista_delle_stime.append(parametri_trovati)
        
        # Genera le immagini output su disco SOLO se stiamo operando nel Test Set,
        # (richiesto esplicitamente dall'Assignment: "Per ogni coppia di test...").
        if ESEGUI_SU_TEST_SET:
            # Combina due stringhe per creare un nome di file univoco e descrittivo.
            id_univoco = f"{coppia['valori_ground_truth']['Pair']}_{coppia['nome_coppia']}"
            # Salva i file visuali dell'allineamento chiamando la funzione.
            salva_risultati_allineamento(immagine_r, immagine_m, parametri_trovati, id_univoco, cartella="RISULTATI_TEST")

    # Fine del for. Una volta analizzate tutte le coppie del dataset, passa le stime
    # alla funzione per stampare la grande tabella matematica riassuntiva nel terminale.
    valuta_prestazioni_medie(dataset_da_usare, lista_delle_stime)