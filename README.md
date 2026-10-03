# Registrazione-Immagini-Multimodali
Sistema di registrazione di immagini multimodali tramite massimizzazione della Mutua Informazione e ottimizzazione (SciPy)


## Guida Completa all'Utilizzo e Architettura

### 1. Organizzazione della Struttura delle Cartelle
Per garantire il corretto funzionamento del sistema, la struttura delle directory deve seguire questa organizzazione:
1. Creare una cartella generale di progetto.
2. Inserire all'interno della cartella generale i file Python e la cartella del **DATASET** esattamente come è stata consegnata.

### 2. Aggiunta di Immagini di Prova Personalizzate
Per testare immagini esterne che non fanno parte dei set di validazione o test:
1. Creare una sottocartella all'interno della cartella `DATASET`.
2. Inserire le immagini di prova direttamente al suo interno (senza creare ulteriori sottocartelle nidificate).
3. **Convenzione di naming:** Le immagini devono seguire rigorosamente lo stesso stile di denominazione delle immagini presenti nelle cartelle `val` o `test`, poiché il programma effettua il matching automatico per identificare le coppie (`Reference` e `Moving`).

---

### 3. Scenari di Esecuzione (Gestiti dal file `Main`)
A seconda dei parametri impostati nel file `Main`, l'esecuzione produrrà tre diversi comportamenti:

* **Scenario 1 (Validation Set):** Non vengono creati file di output su disco. I risultati, i parametri utilizzati e le stime ottenute vengono stampati direttamente in una tabella nel terminale.
* **Scenario 2 (Test Set):** Viene creata automaticamente una cartella denominata `RISULTATI_TEST` nella directory principale, contenente i risultati dell'elaborazione.
* **Scenario 3 (Immagini di Prova):** Viene creata una cartella `IMMAGINI_DI_PROVA` in cui verranno salvate le immagini allineate e le relative immagini differenza.

---

### 4. Configurazione dei Parametri (`Main`)
Tutti i parametri di configurazione sono centralizzati e modificabili all'interno del file **`Main`**. È possibile regolare:
* Il numero di **Bin** dell'istogramma.
* Il metodo di ottimizzazione (es. Powell, Nelder-Mead, BFGS).
* L'attivazione dell'uso delle immagini di test, validazione o prova.
* L'applicazione di tecniche di **pre-processing**.
* L'utilizzo della **piramide a multirisoluzione**.

---

### 5. Grafici di Convergenza e Piramide Multirisoluzione
* Ad ogni avvio (sia in fase di valutazione che di test o prova), viene generata una cartella denominata **`Grafici_Convergenza`** (nel percorso in cui si trova il `Main`), contenente l'andamento della Mutua Informazione.
* **Piramide Multirisoluzione:** Quando questa opzione è attiva, per ciascuna immagine vengono generati **due grafici distinti**:
  1. *Fase Coarse (Grossolana):* Mostra la convergenza iniziale con un minor numero di iterazioni (grazie a una superficie di costo più regolare).
  2. *Fase Fine (Fine-tuning):* Mostra la convergenza della fase di perfezionamento, che parte da valori elevati di Mutua Informazione focalizzandosi unicamente sui micro-aggiustamenti finali.

---

> **NOTA BENE:** Se si sceglie di eseguire il programma utilizzando la cartella delle immagini di prova personalizzate, non è strettamente necessario disattivare l'uso delle immagini di test (`FALSE`), poiché il controllo delle immagini di prova viene eseguito per primo: se la condizione risulta vera, il blocco viene eseguito e il programma termina l'esecuzione in modo pulito.
