import time


def producer(queue, shared_texts, pipe_idx, gpu_idx, cfg_pretrained, start_idx):
    print(f'Starting producer {pipe_idx} at gpu {gpu_idx}')
    import torch
    from transformers import pipeline

    pipe = pipeline("feature-extraction", model=cfg_pretrained, tokenizer=cfg_pretrained, device=gpu_idx)
    i = 0
    for text in shared_texts:
        feats = pipe(text)
        tensor_feats = torch.tensor(feats[0], dtype=torch.double)

        queue.put([start_idx + i, tensor_feats])
        i += 1

    print(f"Process {pipe_idx} has processed {i} texts")

    del pipe
    while True:
        time.sleep(5)

    print(f"Exiting producer {pipe_idx} after processing {i} texts")


def consumer(queue, target_path, target_count, embedding_size, max_len):
    print(f"Starting consumer...")
    import h5py

    with h5py.File(target_path, 'a') as f:
        features_g = f["features"] if "features" in f else f.create_group("features")

        count = 0
        while True:
            if count == target_count:
                print(f"Final count: {count}")
                break
            elif count % 100000 == 0:
                print(f"Current count: {count}")

            data_arr = queue.get()

            data_idx = str(data_arr[0])

            features_g.create_dataset(data_idx, data=data_arr[1])

            if embedding_size.value < 0:
                embedding_size.value = data_arr[1].shape[1]

            max_len.value = max(data_arr[1].shape[0], max_len.value)

            count = count + 1

    print("Exiting consumer...")
