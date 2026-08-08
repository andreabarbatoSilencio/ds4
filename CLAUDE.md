# ds4

Le istruzioni per lavorare su questa codebase sono in AGENT.md, incluso qui sotto.

@AGENT.md

## Dove guardare

Prima di rispondere su come funziona qualcosa, leggi il codice — non dedurlo dai README.
La sezione "Layout" di AGENT.md è un puntatore parziale e lo dichiara. Punti d'ingresso
che non sono ovvi dal layout:

- **Come sono fatti i test**: `tests/ds4_test.c` include `../ds4_server.c` direttamente
  (prime 3 righe) per raggiungere le funzioni interne. La tabella dei check invocabili
  singolarmente è in fondo al file (`--server`, `--metal-kernels`, `--logprob-vectors`, …);
  `./ds4_test --list` la stampa.
- **L'oracolo**: i test `*_exact` confrontano path di riferimento contro path ottimizzato
  sullo stesso input (`ref_*` vs `simd_*`), commutando l'ottimizzazione via variabili
  d'ambiente `DS4_METAL_ENABLE_*` / `DS4_METAL_DISABLE_*`. È per questo che il backend
  CPU resta vivo: è il riferimento, non codice morto.
- **Contratto vs numeri**: `tests/test_engine_correctness.c` asserisce sui token ID
  (il contratto utente); il suo gemello `test_engine_mgpu_runtime.c` sui delta dei
  logit a 1e-4. Il commento in testa al primo spiega quale dei due è il contratto.
- **Vettori ufficiali**: `tests/test-vectors/README.md` — `official.vec` (verità esterna
  dall'API DeepSeek) e `local-golden.vec` (drift che mantiene lo stesso token greedy ma
  degrada la distribuzione).
- **Adattamento hardware**: i target del `Makefile` (`cuda-spark`, `cuda-generic`,
  `strix-halo`, `cpu`) sono precedenti da imitare; `STRIXHALO.md` è il verbale di un
  adattamento reale, workaround inclusi.
