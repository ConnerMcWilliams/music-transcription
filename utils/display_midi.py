import torch
import pretty_midi
import matplotlib.pyplot as plt

def display_midi_from_dir(dir) :
    midi = pretty_midi.PrettyMIDI(dir)
    piano_roll = midi.instruments[0].get_piano_roll(fs=1)

    plt.imshow(piano_roll, aspect='auto', origin='lower', cmap='hot')
    plt.title("Piano Roll")
    plt.xlabel("Time (frames)")
    plt.ylabel("MIDI pitch")
    plt.show()
    
def display_midi_from_roll(roll) :
    plt.imshow(roll, aspect='auto', origin='lower', cmap='hot')
    plt.title("Piano Roll")
    plt.xlabel("Time (frames)")
    plt.ylabel("MIDI pitch")
    plt.show()
    
def compare_piano_rolls_stacked(roll1, roll2, labels=('Roll 1', 'Roll 2')):
    roll1 = roll1.detach().cpu().numpy() if isinstance(roll1, torch.Tensor) else roll1
    roll2 = roll2.detach().cpu().numpy() if isinstance(roll2, torch.Tensor) else roll2

    fig, axs = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

    axs[0].imshow(roll1.T, aspect='auto', origin='lower', cmap='hot')
    axs[0].set_title(labels[0])
    axs[0].set_ylabel("MIDI pitch")

    axs[1].imshow(roll2.T, aspect='auto', origin='lower', cmap='hot')
    axs[1].set_title(labels[1])
    axs[1].set_xlabel("Time (frames)")
    axs[1].set_ylabel("MIDI pitch")

    plt.tight_layout()
    plt.show()
    
def display_spectrogram(spec) :
    plt.figure(figsize=(10, 4))
    plt.imshow(spec[0,:,:].numpy(), aspect='auto', origin='lower')
    plt.title('Spectrogram (Torchaudio)')
    plt.ylabel('Frequency Bin')
    plt.xlabel('Time Frame')
    plt.colorbar(label='Power (dB)')
    plt.show()
    
def display_spectrogram_with_beat(spec, beats, subdivisions) :
    plt.figure(figsize=(10, 4))
    plt.imshow(spec[0,:,:].numpy(), aspect='auto', origin='lower')
    plt.title('Spectrogram (Torchaudio)')
    plt.ylabel('Frequency Bin')
    plt.xlabel('Beats')
    plt.colorbar(label='Power (dB)')
    custom_tick_locations = [i * subdivisions for i in range(0, beats)]
    custom_tick_labels = [i for i in range(0, beats)]
    plt.xticks(custom_tick_locations, custom_tick_labels)
    plt.show()