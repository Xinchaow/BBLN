import os
os.environ["PATH"] = "D:\Anaconda\envs\Python38"
def run_func(description, ppi_path, pseq_path, vec_path,
            split_new, split_mode, train_valid_index_path,
            use_lr_scheduler, save_path, graph_only_train, 
            batch_size, epochs):
    os.system("python -u train.py \
            --description={} \
            --ppi_path={} \
            --pseq_path={} \
            --vec_path={} \
            --split_new={} \
            --split_mode={} \
            --train_valid_index_path={} \
            --use_lr_scheduler={} \
            --save_path={} \
            --graph_only_train={} \
            --batch_size={} \
            --epochs={} \
            ".format(description, ppi_path, pseq_path, vec_path, 
                    split_new, split_mode, train_valid_index_path,
                    use_lr_scheduler, save_path, graph_only_train, 
                    batch_size, epochs))

if __name__ == "__main__":
    description = "test_27k_random"

    ppi_path = "./data/protein.actions.SHS27k.STRING.txt"
    pseq_path = "./data/protein.SHS27k.sequences.dictionary.tsv"
    # ppi_path = "./data/protein.actions.SHS148k.STRING.txt"
    # pseq_path = "./data/protein.SHS148k.sequences.dictionary.tsv"
    # ppi_path = "./data/9606.protein.actions.all_connected.txt"
    # pseq_path = "./data/protein.STRING_all_connected.sequences.dictionary.tsv"
    vec_path = "./data/vec5_CTC.txt"

    split_new = "False"
    split_mode = "random"
    train_valid_index_path = "./train_valid_index_json/train_valid_index_json.json"

    use_lr_scheduler = "True"
    save_path = "./save_model/"
    graph_only_train = "False"

    batch_size = 512
    epochs = 800

    run_func(description, ppi_path, pseq_path, vec_path, 
            split_new, split_mode, train_valid_index_path,
            use_lr_scheduler, save_path, graph_only_train, 
            batch_size, epochs)