"""
Gateway for Offline Training Runs
Accept:
    PolicyInformation
        Policy according to System scheme
    SynthDataset
        (Trace, Impetus, Intent) triplets
        System scheme
    MethodConfig
    MethodDataset
        e.g. message triplets, preference pairs, etc

    Always supports either SynthDataset or MethodDataset

SFT (Synth, Gemini, OpenAI)
DPO (Synth, OpenAI)
Progress Reward Model (Synth)


class TrainingRun
"""
