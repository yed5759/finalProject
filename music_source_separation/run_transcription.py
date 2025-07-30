from transcribe_utils import transcribe_piano_audio

result = transcribe_piano_audio(
    r"C:\\Users\\Hillel\\Desktop\\delicate-classical-piano-sad-loop_50bpm_C_major.wav",
    model_path="models/checkpoints/model_epoch_10.pt"
)

print("Transcription done!")
print("Note tuples path:", result['note_tuples_path'])
print("Duration:", result['duration'], "seconds")
print("Notes detected:", result['notes'])