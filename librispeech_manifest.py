import json
import os
from glob import glob
from tqdm import tqdm
from argparse import ArgumentParser

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument(
        "--libri_root",
        type=str,
        default=None,
        required=True,
        help="Path to LibriSpeech base directory",
    )
    parser.add_argument("--splits", type=str, nargs='+', default=["train-clean-100"], help="Split")
    parser.add_argument("--output_dir", type=str, default=None, required=True, help="Path to output directory")
    parser.add_argument("--lowercase", action='store_true', help="Whether to lowercase all transcription")
    args = parser.parse_args()

    id2trans = dict()
    os.makedirs(args.output_dir, exist_ok=True)
    for splt in args.splits:
        for txt in glob(f"{args.libri_root}/{splt}/**/*.trans.txt", recursive=True):
            with open(txt) as f:
                for line in f.readlines():
                    id, trans = line.strip().split(' ', 1)
                    id2trans[id] = trans.lower() if args.lowercase else trans
    
        with open(f'{args.output_dir}/{splt}.jsonl', 'w') as f:
            for audio in tqdm(glob(f"{args.libri_root}/{splt}/**/*.flac", recursive=True), desc=f"Processing {splt}"):
                id = audio.split('/')[-1].split('.')[0]
                f.write(json.dumps({"audio_path": audio, "text": id2trans[id].lower(), "task": "<|transcribe|>", "lang": "<|en|>", "prompt": "", "timestamp": "<|notimestamps|>"}))
                f.write('\n')