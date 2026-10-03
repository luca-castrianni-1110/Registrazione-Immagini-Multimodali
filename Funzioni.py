# Importa numpy per operazioni matematiche su matrici (array) ad alta velocità.
import numpy as np
# Importa OpenCV per trasformazioni geometriche (rotazioni, traslazioni) e filtri immagine.
import cv2
# Importa la funzione 'minimize' da SciPy, che serve per trovare il punto di minimo di una funzione (l'ottimizzatore).
from scipy.optimize import minimize
# Importa matplotlib per generare i grafici di convergenza da mettere nelle slide.
import matplotlib.pyplot as plt
# Importa os per gestire la creazione di cartelle di output.
import os
# Importa time per generare nomi univoci per i file salvati basati sull'orario.
import time

def applica_preprocessing(immagine, tipo_filtro='nessuno'):
    """
    A COSA SERVE: Applica filtri all'immagine prima di darla in pasto all'algoritmo di allineamento.
    PERCHÉ SI USA: Richiesto per l'Ablation Study (slide). Serve per verificare se rimuovendo rumore 
    o evidenziando i bordi l'algoritmo calcola la Mutua Informazione in modo più preciso o veloce.
    """
    # Se il parametro è 'nessuno', restituisce l'immagine così com'è.
    if tipo_filtro == 'nessuno':
        return immagine

    # Se il parametro è 'blur', applica una sfocatura Gaussiana.
    elif tipo_filtro == 'blur':
        # La matrice (5, 5) indica la grandezza del filtro. Sfoca per rimuovere il rumore dei sensori termici.
        return cv2.GaussianBlur(immagine, (5, 5), 0)

    # Se il parametro è 'equalizzazione', migliora il contrasto globale dell'immagine.
    elif tipo_filtro == 'equalizzazione':
        # Spalma l'istogramma dell'immagine per rendere più visibili i dettagli nascosti in aree scure/chiare.
        return cv2.equalizeHist(immagine)

    # Se il parametro è 'bordi', calcola le derivate per estrarre i contorni.
    elif tipo_filtro == 'bordi':
        # Filtro di Sobel per trovare i bordi verticali (variazioni lungo l'asse X).
        grad_x = cv2.Sobel(immagine, cv2.CV_64F, 1, 0, ksize=3)
        # Filtro di Sobel per trovare i bordi orizzontali (variazioni lungo l'asse Y).
        grad_y = cv2.Sobel(immagine, cv2.CV_64F, 0, 1, ksize=3)
        # Combina i gradienti X e Y calcolando la magnitudo (forza totale del bordo).
        magnitudo = cv2.magnitude(grad_x, grad_y)
        
        # Converte la matrice di numeri in virgola mobile (64F) in formato immagine standard 8-bit (0-255).
        # Si usa perché la funzione histogram2d successiva richiede valori da 0 a 255.
        magnitudo_norm = cv2.normalize(magnitudo, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        return magnitudo_norm
    
    # Se il parametro è 'pipeline', applica tutti i filtri in sequenza.
    elif tipo_filtro == 'pipeline':
        # 1. Smussa l'immagine per togliere il rumore.
        sfocata = cv2.GaussianBlur(immagine, (5, 5), 0)
        # 2. Aumenta il contrasto sull'immagine smussata.
        equalizzata = cv2.equalizeHist(sfocata)
        
        # 3. Estrae i bordi dall'immagine pulita e contrastata.
        grad_x = cv2.Sobel(equalizzata, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(equalizzata, cv2.CV_64F, 0, 1, ksize=3)
        magnitudo = cv2.magnitude(grad_x, grad_y)
        
        # Normalizza e converte l'immagine finale a 8-bit.
        magnitudo_norm = cv2.normalize(magnitudo, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        return magnitudo_norm

    # Gestione errori: se passi un nome filtro sbagliato, non fa niente e restituisce l'immagine base.
    else:
        return immagine
    

def calcola_mutua_informazione(immagine_riferimento, immagine_moving, numero_bin):
    """
    A COSA SERVE: Calcola quanto due immagini sono matematicamente simili usando la Mutua Informazione.
    PERCHÉ SI USA: È la funzione fondamentale richiesta dall'assignment per valutare la bontà 
    dell'allineamento. Più alto è il numero restituito, migliore è l'allineamento.
    """
    
    # histogram2d confronta ogni pixel di un'immagine con il pixel corrispondente nell'altra.
    # .ravel() appiattisce le matrici 2D in array 1D per poterle confrontare elemento per elemento.
    # I 'bins' decidono in quanti contenitori raggruppare i livelli di grigio (es. 64, 128 o 256).
    istogramma_congiunto, _, _ = np.histogram2d(
        immagine_riferimento.ravel(),
        immagine_moving.ravel(),
        bins=numero_bin,
        range=[[0, 256], [0, 256]] # Il range dei pixel è sempre da nero (0) a bianco (255).
    )
    
    # Normalizza l'istogramma (divide ogni cella per il numero totale di pixel).
    # Si usa per trasformare un conteggio grezzo in una matrice di probabilità congiunta (che somma a 1).
    probabilita_congiunta = istogramma_congiunto / np.sum(istogramma_congiunto)
    
    # Somma le probabilità su tutte le colonne (axis=1) per ottenere la probabilità della sola immagine Reference.
    distribuzione_marginale_riferimento = np.sum(probabilita_congiunta, axis=1)
    # Somma le probabilità su tutte le righe (axis=0) per ottenere la probabilità della sola immagine Moving.
    distribuzione_marginale_moving = np.sum(probabilita_congiunta, axis=0)
    
    '''Entropia: calcolo della formula H(X) = -sum(p * log(p))'''
    
    # Estrae solo le probabilità maggiori di 0 per la Reference.
    # PERCHÉ SI USA: Il logaritmo di 0 non esiste. Se non lo togli, Python dà un errore critico (NaN).
    prob_riferimento_valida = distribuzione_marginale_riferimento[distribuzione_marginale_riferimento > 0]
    # Applica la formula dell'Entropia (dispersione dell'informazione) per l'immagine Reference.
    entropia_riferimento = -np.sum(prob_riferimento_valida * np.log(prob_riferimento_valida))
    
    # Filtra gli zeri per l'immagine Moving.
    prob_moving_valida = distribuzione_marginale_moving[distribuzione_marginale_moving > 0]
    # Calcola l'Entropia per l'immagine Moving.
    entropia_moving = -np.sum(prob_moving_valida * np.log(prob_moving_valida))
    
    # Filtra gli zeri per la matrice di probabilità congiunta.
    prob_congiunta_valida = probabilita_congiunta[probabilita_congiunta > 0]
    # Calcola l'Entropia Congiunta (quanto le due immagini sono disperse insieme).
    entropia_congiunta = -np.sum(prob_congiunta_valida * np.log(prob_congiunta_valida))
    
    # Calcola la formula finale: MI = H(R) + H(M) - H(R,M)
    mutua_informazione = entropia_riferimento + entropia_moving - entropia_congiunta
    
    # Restituisce il valore numerico dell'allineamento.
    return mutua_informazione



def funzione_obiettivo_da_minimizzare(parametri_geometrici, immagine_riferimento, immagine_moving, numero_bin, storia_mi=None):
    """
    A COSA SERVE: Prende una proposta di rotazione e traslazione, sposta l'immagine e valuta il risultato.
    PERCHÉ SI USA: SciPy ha bisogno di una singola funzione da chiamare migliaia di volte provando 
    valori diversi fino a quando non trova il risultato ottimale (quello che dà il valore minimo).
    """
    
    # Decomprime l'array proposto dall'ottimizzatore in 3 variabili: angolo in radianti, spostamento X, spostamento Y.
    angolo_theta, traslazione_x, traslazione_y = parametri_geometrici
    # Estrae l'altezza (h) e la larghezza (w) dell'immagine Moving.
    h, w = immagine_moving.shape[:2]
    
    # Definisce il punto attorno al quale ruotare l'immagine.
    # PERCHÉ (0,0): Il Ground Truth del prof è calcolato ruotando l'immagine rispetto all'angolo in alto a sinistra.
    centro = (0, 0)
    
    # Converte i radianti in gradi (OpenCV richiede gradi).
    # PERCHÉ IL SEGNO MENO: I sistemi di riferimento informatici invertono le rotazioni. Il meno allinea ai calcoli del prof.
    angolo_gradi = -np.degrees(angolo_theta)
    
    # Genera la matrice di base per ruotare l'immagine. 1.0 indica di non cambiare la scala (zoom).
    matrice_trasformazione = cv2.getRotationMatrix2D(centro, angolo_gradi, 1.0)
    
    # Modifica manualmente la matrice di rotazione aggiungendo lo spostamento sull'asse X.
    matrice_trasformazione[0, 2] += traslazione_x
    # Aggiunge lo spostamento sull'asse Y. Ora la matrice può fare sia rotazione che traslazione.
    matrice_trasformazione[1, 2] += traslazione_y
    
    # Applica fisicamente la matrice di trasformazione all'immagine Moving.
    # 'flags=cv2.INTER_LINEAR' assicura che usi l'interpolazione bilineare richiesta.
    immagine_moving_trasformata = cv2.warpAffine(
        immagine_moving,
        matrice_trasformazione,
        (w, h),
        flags=cv2.INTER_LINEAR
    )
    
    # Calcola il punteggio di Mutua Informazione sull'immagine appena spostata e ruotata.
    mutua_informazione = calcola_mutua_informazione(
        immagine_riferimento,
        immagine_moving_trasformata,
        numero_bin
    )
    
    # Se è stata passata una lista 'storia_mi', salva il punteggio corrente.
    # Serve per generare il grafico della convergenza per le slide.
    if storia_mi is not None:
        storia_mi.append(mutua_informazione)
    
    # Restituisce la MI cambiata di segno.
    # PERCHÉ SI USA: La funzione minimize di SciPy sa cercare solo il "punto più basso". 
    # Visto che noi vogliamo massimizzare la MI (cercare il punto più alto), le passiamo il negativo.
    return -mutua_informazione



def stima_modello_geometrico(immagine_riferimento, immagine_moving, metodo_scelto, numero_bin, tipo_preproc='nessuno', parametri_iniziali=None):
    """
    A COSA SERVE: Prepara e lancia il vero e proprio motore matematico di ottimizzazione.
    PERCHÉ SI USA: Serve per trovare i 3 parametri geometrici ottimali (rotazione, TX, TY) 
    senza doverli testare tutti a mano alla cieca.
    """
    
    # Applica i filtri scelti alle immagini prima di iniziare i calcoli.
    img_rif_filtrata = applica_preprocessing(immagine_riferimento, tipo_preproc)
    img_mov_filtrata = applica_preprocessing(immagine_moving, tipo_preproc)

    # Correzione formattazione: in alcuni sistemi il trattino di Nelder-Mead viene incollato male.
    metodo_scelto = metodo_scelto.replace('–', '-')
    
    # Configura le tolleranze in base all'algoritmo selezionato.
    if metodo_scelto == 'Powell':
        # Powell lavora meglio se gli diamo direzioni di ricerca iniziali ampie per evitare minimi locali.
        passi_iniziali = np.array([
            [0.05, 0.0, 0.0], # Prova a ruotare un po'
            [0.0, 10.0, 0.0], # Prova a spostare di 10 pixel su X
            [0.0, 0.0, 10.0]  # Prova a spostare di 10 pixel su Y
        ])
        
        # xtol/ftol decidono quando fermarsi. maxiter imposta un tetto massimo di tentativi.
        opzioni_ottimizzazione = {
            'xtol': 1e-4,
            'ftol': 1e-4,
            'maxiter': 2000,
            'direc': passi_iniziali
        }
        
    elif metodo_scelto == 'Nelder-Mead':
        # Configurazione specifica per il metodo Nelder-Mead.
        opzioni_ottimizzazione = {
            'xatol': 1e-3,
            'fatol': 1e-3,
            'adaptive': True,
            'maxiter': 2000,
        }
        
    elif metodo_scelto == 'BFGS':
        # Configurazione specifica per BFGS.
        opzioni_ottimizzazione = {
            'gtol': 1e-4,
            'eps': 1e-3,
            'maxiter': 2000
        }
        
    else:
        # Se non viene passato un metodo valido, usa opzioni di default.
        opzioni_ottimizzazione = None
        
    # Controlla se abbiamo fornito dei parametri di partenza (es. provenienti dalla funzione coarse_to_fine).
    if parametri_iniziali is None:
        # Se non li forniamo, l'algoritmo inizia presumendo che le immagini siano già allineate [Angolo 0, Tx 0, Ty 0].
        parametri_partenza = np.array([0.0, 0.0, 0.0])
    else:
        # Se forniti, inizia da quelli per fare prima.
        parametri_partenza = parametri_iniziali
        
    # Lista vuota che verrà riempita dentro la funzione_obiettivo con tutti i valori calcolati.
    storia_mi = []
    
    # Lancia l'ottimizzatore SciPy. È lui che chiama in loop la funzione_obiettivo.
    risultato_ottimizzazione = minimize(
        funzione_obiettivo_da_minimizzare, # La funzione da testare.
        parametri_partenza,                # Il punto iniziale [0,0,0].
        args=(img_rif_filtrata, img_mov_filtrata, numero_bin, storia_mi), # Argomenti extra da passare alla funzione.
        method=metodo_scelto,              # Il nome dell'algoritmo matematico.
        options=opzioni_ottimizzazione     # Le regole su quando fermarsi.
    )
    
    # Generazione grafico (se la lista è stata popolata).
    if len(storia_mi) > 0:
        # Crea una figura matplotlib.
        plt.figure(figsize=(8, 5))
        # Disegna la linea dell'andamento usando i valori registrati.
        plt.plot(storia_mi, linestyle='-', color='b') 
        # Aggiunge titolo e label.
        plt.title(f"Convergenza MI - Metodo: {metodo_scelto} {numero_bin} ({len(storia_mi)} valutazioni)")
        plt.xlabel("Valutazioni della Funzione Obiettivo")
        plt.ylabel("Valore della Mutua Informazione")
        plt.grid(True)
        
        # Crea la cartella per i grafici se non esiste.
        if not os.path.exists("GRAFICI_CONVERGENZA"):
            os.makedirs("GRAFICI_CONVERGENZA")
        
        # Crea un numero basato sul tempo per non sovrascrivere file precedenti.
        id_univoco = int(time.time() * 1000)
        # Salva l'immagine.
        plt.savefig(f"GRAFICI_CONVERGENZA/convergenza_{metodo_scelto}_{id_univoco}.png")
        # Chiude la figura per liberare la memoria RAM del computer.
        plt.close()
    
    # Restituisce l'array con i 3 valori ottimali trovati (risultato_ottimizzazione.x).
    return risultato_ottimizzazione.x


def stima_coarse_to_fine(immagine_riferimento, immagine_moving, metodo_scelto, numero_bin, tipo_preproc='nessuno', fattore_scala=0.5):
    """
    A COSA SERVE: Implementa una piramide multirisoluzione a 2 livelli. 
    PERCHÉ SI USA: Prima calcola un allineamento grezzo su immagini rimpicciolite (velocissimo e immune al rumore), 
    poi usa quei risultati per partire avvantaggiati sull'immagine grande. Supera il problema dei minimi locali.
    """
    
    # Estrae le dimensioni attuali.
    h, w = immagine_riferimento.shape[:2]
    # Calcola le nuove dimensioni dimezzate (fattore_scala = 0.5).
    nuove_dimensioni = (int(w * fattore_scala), int(h * fattore_scala))
    
    # Rimpicciolisce la Reference.
    img_rif_coarse = cv2.resize(immagine_riferimento, nuove_dimensioni, interpolation=cv2.INTER_AREA)
    # Rimpicciolisce la Moving.
    img_mov_coarse = cv2.resize(immagine_moving, nuove_dimensioni, interpolation=cv2.INTER_AREA)
    
    # Chiama l'ottimizzatore sulle immagini piccole (FASE COARSE).
    parametri_coarse = stima_modello_geometrico(
        img_rif_coarse, img_mov_coarse, metodo_scelto, numero_bin,
        tipo_preproc=tipo_preproc, parametri_iniziali=np.array([0.0, 0.0, 0.0])
    )
    
    # Ripristina i parametri per usarli sull'immagine a grandezza normale.
    # L'angolo non cambia al variare dello zoom.
    angolo_iniziale = parametri_coarse[0]
    # I pixel trovati a risoluzione dimezzata devono essere raddoppiati.
    tx_iniziale = parametri_coarse[1] * (1.0 / fattore_scala)
    ty_iniziale = parametri_coarse[2] * (1.0 / fattore_scala)
    
    # Prepara il punto di partenza per la fase finale.
    parametri_trampolino = np.array([angolo_iniziale, tx_iniziale, ty_iniziale])
    
    # Chiama l'ottimizzatore sulle immagini intere partendo dal punto appena calcolato (FASE FINE).
    parametri_fine = stima_modello_geometrico(
        immagine_riferimento, immagine_moving, metodo_scelto, numero_bin,
        tipo_preproc=tipo_preproc, parametri_iniziali=parametri_trampolino
    )
    
    # Restituisce i parametri precisi finali.
    return parametri_fine


def valuta_prestazioni_medie(dati_immagini, parametri_stimati_lista):
    """
    A COSA SERVE: Calcola l'errore medio (MAE) confrontando i tuoi risultati con il Ground Truth.
    PERCHÉ SI USA: Richiesto dalla consegna. Genera una tabella nel terminale per analizzare 
    quantitativamente (con numeri) se l'allineamento è buono.
    """
    
    # Inizializza a zero le variabili che sommeranno gli errori per fare poi la media.
    somma_err_ang = 0.0
    somma_err_tx = 0.0
    somma_err_ty = 0.0
    # Conta quante immagini stiamo valutando.
    n = len(dati_immagini)

    # Crea variabili per bloccare la larghezza delle colonne della tabella (w = width).
    # Assicura che i numeri nel terminale appaiano ordinati in riga.
    w_nome = 12
    w_ang  = 15
    w_deg  = 10
    w_tras = 12

    # Genera la stringa di intestazione della tabella con formattazione dinamica.
    header = (f"{'Coppia':<{w_nome}} | "
                f"{'Angolo_GT':<{w_ang}} {'Ang_Stim':<{w_ang}} {'Err_Ang(rad)':<{w_ang}} {'(deg)':<{w_deg}} | "
                f"{'TX_GT':<{w_tras}} {'TX_Stim':<{w_tras}} {'Err_TX':<{w_tras}} | "
                f"{'TY_GT':<{w_tras}} {'TY_Stim':<{w_tras}} {'Err_TY':<{w_tras}}")
    
    # Genera linee di separazione per estetica.
    sep = "=" * len(header)
    sub_sep = "-" * len(header)
    
    # Stampa la testa della tabella nel terminale.
    print("\n" + sep)
    print(header)
    print(sep)
    
    # Avvia un ciclo per valutare ogni singola stima calcolata.
    for i in range(n):
        # Estrae il nome dell'immagine.
        nome = dati_immagini[i]['nome_coppia']
        # Estrae i parametri reali dal Ground Truth.
        gt = dati_immagini[i]['valori_ground_truth']
        
        # Salva i valori reali di GT.
        a_gt = gt['AngleRad']
        tx_gt = gt['Tx']
        ty_gt = gt['Ty']
        
        # Estrae i parametri che ha trovato il tuo algoritmo.
        a_stim, tx_stim, ty_stim = parametri_stimati_lista[i]
        
        # Calcola l'errore assoluto (valore stimato meno valore reale senza considerare il segno negativo).
        e_a = abs(a_stim - a_gt)
        e_x = abs(tx_stim - tx_gt)
        e_y = abs(ty_stim - ty_gt)
        
        # Converte l'errore dell'angolo in gradi (più umano da leggere in debug rispetto ai radianti).
        e_a_deg = np.degrees(e_a)
        
        # Stampa i dati della singola riga formatattandoli in modo preciso.
        print(f"{nome:<{w_nome}} | "
                f"{a_gt:<{w_ang}.4f} {a_stim:<{w_ang}.4f} {e_a:<{w_ang}.4f} {e_a_deg:<{w_deg}.2f} | "
                f"{tx_gt:<{w_tras}.2f} {tx_stim:<{w_tras}.2f} {e_x:<{w_tras}.2f} | "
                f"{ty_gt:<{w_tras}.2f} {ty_stim:<{w_tras}.2f} {e_y:<{w_tras}.2f}")
        
        # Somma gli errori alla variabile totale.
        somma_err_ang += e_a
        somma_err_tx += e_x
        somma_err_ty += e_y

    # Finita la lista, divide la somma degli errori per il numero totale per trovare le MEDIE.
    m_a = somma_err_ang / n
    m_x = somma_err_tx / n
    m_y = somma_err_ty / n
    # Converte la media radianti in gradi.
    m_a_deg = np.degrees(m_a)
    
    # Stampa la riga finale con i risultati medi.
    print(sub_sep)
    print(f"{'MEDIA':<{w_nome}} | "
            f"{'-':<{w_ang}} {'-':<{w_ang}} {m_a:<{w_ang}.4f} {m_a_deg:<{w_deg}.2f} | "
            f"{'-':<{w_tras}} {'-':<{w_tras}} {m_x:<{w_tras}.2f} | "
            f"{'-':<{w_tras}} {'-':<{w_tras}} {m_y:<{w_tras}.2f}")
    print(sep + "\n")
    
    # Restituisce le medie nel caso servano ad altre funzioni del codice.
    return m_a, m_x, m_y


def salva_risultati_allineamento(immagine_r, immagine_m, parametri, nome_coppia, cartella="RISULTATI_TEST"):
    """
    A COSA SERVE: Applica i parametri finali all'immagine e salva su PC le immagini risultanti.
    PERCHÉ SI USA: Consegna finale del punto 3 (creare "immagine allineata" e "immagine differenza").
    Serve a dimostrare visivamente il lavoro svolto.
    """
    
    # Se la cartella di output (es. RISULTATI_TEST) non esiste, la crea nel sistema operativo.
    if not os.path.exists(cartella):
        os.makedirs(cartella)
    
    # Estrae i parametri calcolati dall'ottimizzatore, assicurandosi che siano valori singoli e non array.
    a_rad = parametri[0].item() if isinstance(parametri[0], np.ndarray) else parametri[0]
    tx = parametri[1].item() if isinstance(parametri[1], np.ndarray) else parametri[1]
    ty = parametri[2].item() if isinstance(parametri[2], np.ndarray) else parametri[2]
    
    # Prende le dimensioni dell'immagine per dire alla funzione warpAffine quanto deve essere grande l'output.
    h, w = immagine_r.shape[:2]
    
    # Imposta il centro di rotazione in alto a sinistra (0,0) coerentemente con l'algoritmo di stima.
    centro = (0, 0)
    # Converte l'angolo in gradi con il segno meno.
    a_deg = -np.degrees(a_rad)
    
    # Crea e popola la matrice di trasformazione (roto-traslazione).
    matrice = cv2.getRotationMatrix2D(centro, a_deg, 1.0)
    matrice[0, 2] += tx
    matrice[1, 2] += ty
    
    # Genera l'immagine Moving allineata sovrapponendola idealmente alla Reference.
    img_allineata = cv2.warpAffine(
        immagine_m,
        matrice,
        (w, h),
        flags=cv2.INTER_LINEAR
    )
    
    # Crea l'immagine differenza. absdiff sottrae il valore di ogni pixel dell'immagine allineata 
    # da quello della Reference (senza andare sotto zero).
    # Se le immagini sono identiche, sarà nera. Se ci sono differenze (non allineate), si vedono bordi bianchi.
    img_differenza = cv2.absdiff(immagine_r, img_allineata)
    
    # Salva fisicamente l'immagine allineata sul disco.
    cv2.imwrite(f"{cartella}/{nome_coppia}_allineata.png", img_allineata)
    # Salva fisicamente l'immagine differenza sul disco.
    cv2.imwrite(f"{cartella}/{nome_coppia}_differenza.png", img_differenza)