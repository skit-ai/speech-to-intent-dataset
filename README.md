# Speech to Intent Dataset
Dataset Release for the task of Intent Classification from Human Speech

## About



## Download and License

The dataset can be downloaded by clicking on this [link](https://speech-to-intent-dataset.s3.ap-south-1.amazonaws.com/speech-to-intent.zip). Incase you face any issues please reach out to kmanas@skit.ai.

This dataset is shared under [Creative Commons Attribution-NonCommercial 4.0 International](https://creativecommons.org/licenses/by-nc/4.0/) Licence. This places restrictions on commercial use of this dataset.

## Uses



## Structure

This release contains data of (Indian English) speech samples tagged with the relevant intent from the banking domain.

Frequency : 8 Khz

Structure

```
- wav_audios          [contains the wav audio files]
- dataset.csv         [contains the intent tag for audio files, where each row contains "<id> | <intent> | <template> | <audio-path> | <speaker-id>"]
- speaker_info.csv    [contains the speaker characteristics, where each row contains "<speaker-id> | <native-language> | <languages-spoken> | <places-lived> | <gender>"]

```

More information regarding the dataset can be found in the [datasheet](./datasheet.md).

## Citation

If you are using this dataset, please cite using the link in the About section on the right.
