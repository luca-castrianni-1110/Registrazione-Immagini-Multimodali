# Importa il modulo 'os' che serve per interagire con il sistema operativo (creare percorsi, leggere cartelle).
import os
# Importa la libreria OpenCV, usata per caricare, manipolare e salvare le immagini.
import cv2
# Importa Pandas, una libreria essenziale per leggere e gestire file tabellari come il CSV del Ground Truth.
import pandas as pd

def carica_dati_da_dataset(percorso_base_dataset, percorso_file_csv):
    """
    A COSA SERVE: Questa funzione legge il file CSV contenente le soluzioni (Ground Truth) e carica le relative immagini.
    PERCHÉ SI USA: Serve per preparare un set di dati strutturato (dizionari) da passare all'algoritmo, 
    separando automaticamente le immagini usate per calibrare il sistema (Validazione) da quelle per il test finale (Test).
    """
    
    # Legge il file CSV usando Pandas. 'sep=';'' specifica che le colonne nel CSV sono separate da punto e virgola.
    tabella_ground_truth = pd.read_csv(percorso_file_csv, sep=';')
    
    # Inizializza una lista vuota che conterrà i dizionari con i dati di validazione.
    lista_dati_validazione = []
    # Inizializza una lista vuota che conterrà i dizionari con i dati di test.
    lista_dati_test = []
    
    # Avvia un ciclo iterando su ogni riga della tabella caricata dal CSV. '_' ignora l'indice della riga.
    for _, riga in tabella_ground_truth.iterrows():
        # Estrae il nome completo del file dalla colonna 'Filename' e rimuove eventuali spazi vuoti con .strip().
        nome_lungo = str(riga['Filename']).strip()
        # Estrae il nome identificativo della coppia dalla colonna 'Pair' rimuovendo spazi.
        nome_corto = str(riga['Pair']).strip()
        # Estrae a quale set appartiene ('val' o 'test'), rimuove spazi e lo converte in minuscolo per sicurezza.
        sottoinsieme = str(riga['Dataset']).strip().lower()
        
        # Costruisce il percorso della cartella che contiene la coppia di immagini (es. "DATASET/val/pair1").
        cartella_coppia = os.path.join(percorso_base_dataset, sottoinsieme, nome_corto)
        
        # Costruisce il percorso esatto per l'immagine di Riferimento (Reference) aggiungendo "_R.png".
        file_r = os.path.join(cartella_coppia, f"{nome_lungo}_R.png")
        # Costruisce il percorso esatto per l'immagine da Spostare (Moving) aggiungendo "_T.png".
        file_t = os.path.join(cartella_coppia, f"{nome_lungo}_T.png")
        
        # Carica l'immagine Reference. Lo '0' è FONDAMENTALE: forza OpenCV a caricarla in scala di grigi (1 canale).
        # Si usa perché la Mutua Informazione lavora sulle singole intensità dei pixel, non sui colori RGB.
        img_r = cv2.imread(file_r, 0)
        # Carica l'immagine Moving, sempre in scala di grigi.
        img_t = cv2.imread(file_t, 0)
            
        # Crea un dizionario (un "pacchetto") per raggruppare tutte le informazioni relative a questa specifica coppia.
        pacchetto = {
            'nome_coppia': nome_lungo,             # Salva il nome per usarlo nei log e nel salvataggio.
            'immagine_riferimento': img_r,         # Salva la matrice dell'immagine R.
            'immagine_moving': img_t,              # Salva la matrice dell'immagine T.
            'valori_ground_truth': riga.to_dict()  # Salva le traslazioni/rotazioni reali convertendo la riga CSV in dizionario.
        }
        
        # Controlla se la riga appartiene al set di validazione.
        if sottoinsieme == 'val':
            # Se sì, aggiunge il pacchetto alla lista di validazione.
            lista_dati_validazione.append(pacchetto)
        else:
            # Altrimenti, lo aggiunge alla lista di test.
            lista_dati_test.append(pacchetto)
    
    # Restituisce le due liste popolate, pronte per essere usate nel file Main.
    return lista_dati_validazione, lista_dati_test