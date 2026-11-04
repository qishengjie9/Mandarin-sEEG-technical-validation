# Model card

## Intended use

The models provide research benchmarks for decoding Mandarin lexical tone from
speech-production sEEG. They are not intended for clinical decisions, patient
monitoring, diagnosis, or real-time communication without additional validation.

## Within-participant model: STAG-Mandarin

Input shape: `[batch, channels, 2, 250]`.

The two streams are low-frequency activity (0.5--40 Hz) and high-gamma
amplitude (70--150 Hz). The model applies:

1. a shared per-electrode multi-scale temporal stem;
2. trial-adaptive sparse graph attention over electrodes;
3. learned channel-attention pooling;
4. a temporal Transformer;
5. attention pooling and a four-class tone head.

The default evaluation uses five-fold stratified group cross-validation with
stimulus identity as the grouping variable.

## Held-out-participant model

The cross-participant architecture is modular and uses:

1. a shared multi-scale temporal encoder;
2. permutation-equivariant set attention over variable electrode sets;
3. dynamic channel pooling and channel dropout;
4. a temporal context Transformer;
5. tone, initial, and final prediction heads;
6. a supervised contrastive objective;
7. gradient-reversal subject-domain regularization.

The default held-out participants are `sub-02` and `sub-08`; participant identity
and fixed electrode indices are not supplied as decoder inputs.

## Metrics

- Accuracy
- Balanced accuracy
- Macro-F1
- Macro one-vs-rest ROC-AUC
- Row-normalized confusion matrices

Balanced accuracy is the primary metric because the four tone classes are not
guaranteed to have equal support.

## Known limitations

- Electrode coverage differs substantially across participants.
- Tone labels can covary with utterance duration and syllable identity.
- The stored stimulus representation historically differed between acquisition cohorts.
- Cross-participant results should be described as zero-calibration transfer to unseen participants on a partially shared phonological inventory, not as a definitive unseen-word benchmark.
- The current benchmark does not establish a duration-independent neural representation of tone.
