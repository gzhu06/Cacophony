#!/usr/bin/env python3
"""
Computes embeddings on a set of tasks
"""

import json
import os
import shutil
import time
from pathlib import Path
import click
from tqdm import tqdm
from .emb_utils import memmap_embeddings, get_dataloader_for_embedding,\
      save_scene_embedding_and_labels, get_labels_for_timestamps, \
        save_timestamp_embedding_and_labels
import random
import numpy as np

@click.command()
@click.option(
    "--model-path",
    default=None,
    help="Location of model weights file",
    type=click.Path(exists=True),
)
@click.option(
    "--tasks-dir",
    default="tasks",
    help="Location of tasks to compute embeddings on",
    type=str,
)
@click.option(
    "--task",
    default="all",
    help="Task to run. (Default: all)",
    type=str,
)
@click.option(
    "--embedding-name", default="caco", help="embedding name", type=str
)

@click.option(
    "--embeddings-dir", default="embeddings", help="Location to save task embeddings"
)

@click.option(
    "--batch-size", default=1, help="batch size for computing embeddings", type=int
)
@click.option(
    "--max-audio-len", default=160000, help="max audio length in samples", type=float
)
@click.option(
    "--sample-rate", default=16000, help="max audio length in samples", type=int
)

def runner(
    model_path,
    tasks_dir,
    task,
    embeddings_dir,
    embedding_name,
    batch_size: int = 1,
    max_audio_len: int = 160000,
    sample_rate: int = 16000
) -> None:

    # model loading
    if 'caco' in embedding_name or 'passt' in embedding_name:
        from heareval.embeddings.audio_embedding.caco_embeddings import Embedding
    elif 'audiomae' in embedding_name:
        from heareval.embeddings.audio_embedding.audiomae_embeddings import Embedding
        
    embedding = Embedding(model_path, batch_size=batch_size, 
                          audio_max_len=max_audio_len, 
                          sample_rate=sample_rate)

    # Check for directory containing the tasks
    tasks_dir_path = Path(tasks_dir)
    embeddings_dir_path = Path(embeddings_dir)

    if not tasks_dir_path.is_dir():
        raise ValueError(
            "Cannot locate directory containing tasks. "
            f"Ensure that directory named {tasks_dir_path} exists or specify a folder "
            f"containing HEAR tasks using the argument --tasks-dir"
        )

    if task == "all":
        tasks = list(tasks_dir_path.iterdir())
    else:
        tasks = [tasks_dir_path.joinpath(task)]
        assert os.path.exists(tasks[0]), f"{tasks[0]} does not exist"
    for task_path in tqdm(tasks):

        embed_dir = embeddings_dir_path.joinpath(embedding_name)

        task_name = task_path.name
        embed_task_dir = embed_dir.joinpath(task_name)

        done_embeddings = embed_task_dir.joinpath(".done.embeddings")
        if os.path.exists(done_embeddings):
            continue

        if os.path.exists(embed_task_dir):
            shutil.rmtree(embed_task_dir)

        start = time.time()
        task_embeddings(embedding, task_path, embed_task_dir)

        time_elapsed = time.time() - start
        print(
            f"...computed embeddings in {time_elapsed} sec "
        )
        open(embed_task_dir.joinpath("profile.embeddings.json"), "wt").write(
            json.dumps(
                {
                    "time_elapsed": time_elapsed,
                },
                indent=4,
            )
        )

        # Touch this file to indicate that processing completed successfully
        open(done_embeddings, "wt")
     
def task_embeddings(embedding, task_path,  embed_task_dir):
    prng = random.Random()
    prng.seed(0)

    metadata_path = task_path.joinpath("task_metadata.json")
    metadata = json.load(metadata_path.open())
    label_vocab_path = task_path.joinpath("labelvocabulary.csv")

    # Copy these two files to the embeddings directory,
    # so we have everything we need in embeddings for doing downstream
    # prediction and evaluation.
    if not os.path.exists(embed_task_dir):
        os.makedirs(embed_task_dir)
    shutil.copy(metadata_path, embed_task_dir)
    shutil.copy(label_vocab_path, embed_task_dir)

    for split in metadata["splits"]:
        print(f"Getting embeddings for split: {split}")

        split_path = task_path.joinpath(f"{split}.json")
        assert split_path.is_file()

        # Copy over the ground truth labels as they may be needed for evaluation
        shutil.copy(split_path, embed_task_dir)

        # Root directory for audio files for this split
        # embedding_sample_rate = str(task_path).split('/')[-3].split('-')[-1]
        embedding_sample_rate = str(embedding.sample_rate) if embedding.sample_rate !=32000 else '48000'
        audio_dir = task_path.joinpath(embedding_sample_rate, split)

        split_data = json.load(split_path.open())
        audio_filepath_list, label_dict = get_dataloader_for_embedding(split_data, audio_dir)

        outdir = embed_task_dir.joinpath(split)
        if not os.path.exists(outdir):
            os.makedirs(outdir)
        
        total_iter = int(np.ceil(len(audio_filepath_list) / embedding.batch_size))
        for i in tqdm(range(total_iter)):
            
            audio_filepath_sublist = audio_filepath_list[i*embedding.batch_size: (i+1)*embedding.batch_size]
            filenames = [filename.split('/')[-1] for filename in audio_filepath_sublist]
            labels = [split_data[filename.split('/')[-1]] for filename in audio_filepath_sublist]

            if metadata["embedding_type"] == "event":
                embeddings, timestamps = embedding.get_embedding_as_numpy(audio_filepath_sublist, metadata["embedding_type"])
                labels = get_labels_for_timestamps(labels, timestamps)
                save_timestamp_embedding_and_labels(
                    embeddings, timestamps, labels, filenames, outdir
                )

            else:

                embeddings = embedding.get_embedding_as_numpy(audio_filepath_sublist)
                save_scene_embedding_and_labels(embeddings, labels, filenames, outdir)

        memmap_embeddings(outdir, prng, metadata, split, embed_task_dir, split_data)

if __name__ == "__main__":
    runner()
