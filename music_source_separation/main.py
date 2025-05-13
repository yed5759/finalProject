import os
import argparse
import torch
from piano_transformer import PianoTranscriptionSystem

def main():
    parser = argparse.ArgumentParser(description='Piano WAV to MIDI Transformer')
    parser.add_argument('--input', '-i', type=str, required=True, help='Input WAV file or directory of WAV files')
    parser.add_argument('--output', '-o', type=str, default='output', help='Output directory or file')
    parser.add_argument('--model', '-m', type=str, help='Path to trained model')
    parser.add_argument('--batch', '-b', action='store_true', help='Process directory of WAV files')
    parser.add_argument('--cpu', action='store_true', help='Force CPU processing')
    parser.add_argument('--sample-rate', '-sr', type=int, default=16000, choices=[16000, 22050], 
                        help='Audio sample rate (16000 or 22050 Hz)')
    parser.add_argument('--fft-size', '-fs', type=int, default=2048, help='FFT window size')
    parser.add_argument('--cqt-bins', '-cb', type=int, default=84, help='Number of CQT bins')
    
    args = parser.parse_args()
    
    # Check device
    device = 'cpu' if args.cpu else ('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Initialize system
    system = PianoTranscriptionSystem(model_path=args.model, device=device)
    
    if args.batch:
        # Process all WAV files in directory
        if not os.path.isdir(args.input):
            print(f"Error: {args.input} is not a directory")
            return
            
        # Create output directory
        os.makedirs(args.output, exist_ok=True)
        
        # Process all WAV files
        processed = 0
        for filename in os.listdir(args.input):
            if filename.lower().endswith('.wav'):
                input_path = os.path.join(args.input, filename)
                output_path = os.path.join(args.output, filename.replace('.wav', '.mid'))
                
                print(f"Converting {input_path} to {output_path}")
                try:
                    system.transcribe(
                        input_path, 
                        output_path,
                        sample_rate=args.sample_rate,
                        n_fft=args.fft_size,
                        n_cqt_bins=args.cqt_bins
                    )
                    processed += 1
                except Exception as e:
                    print(f"Error processing {input_path}: {e}")
        
        print(f"Processed {processed} files")
    else:
        # Process single file
        if not os.path.isfile(args.input):
            print(f"Error: {args.input} does not exist or is not a file")
            return
            
        # Determine output file
        output_file = args.output
        if os.path.isdir(args.output) or not output_file.lower().endswith('.mid'):
            os.makedirs(args.output, exist_ok=True)
            output_file = os.path.join(args.output, os.path.basename(args.input).replace('.wav', '.mid'))
        
        print(f"Converting {args.input} to {output_file}")
        try:
            midi_data = system.transcribe(
                args.input, 
                output_file,
                sample_rate=args.sample_rate,
                n_fft=args.fft_size,
                n_cqt_bins=args.cqt_bins
            )
            
            # Print statistics
            if midi_data.instruments:
                piano = midi_data.instruments[0]
                print(f"Number of notes: {len(piano.notes)}")
                if piano.notes:
                    min_pitch = min(note.pitch for note in piano.notes)
                    max_pitch = max(note.pitch for note in piano.notes)
                    print(f"Pitch range: {min_pitch} - {max_pitch}")
                    
                    min_vel = min(note.velocity for note in piano.notes)
                    max_vel = max(note.velocity for note in piano.notes)
                    print(f"Velocity range: {min_vel} - {max_vel}")
                    
                    duration = max(note.end for note in piano.notes) - min(note.start for note in piano.notes)
                    print(f"Duration: {duration:.2f} seconds")
            
            print(f"Successfully converted to {output_file}")
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    main() 