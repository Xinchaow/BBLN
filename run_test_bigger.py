import os

os.environ["PATH"] = "D:\Anaconda\envs\Python38"
def run_func(description, ppi_path, pseq_path, vec_path,
            index_path, model, bigger_ppi_path, bigger_pseq_path):
    os.system("python test_bigger.py \
            --description={} \
            --ppi_path={} \
            --pseq_path={} \
            --vec_path={} \
            --index_path={} \
            --model={} \
            --bigger_ppi_path={} \
            --bigger_pseq_path={} \
            ".format(description, ppi_path, pseq_path, vec_path,
                    index_path, model, bigger_ppi_path, bigger_pseq_path))


if __name__ == "__main__":
    description = "test"

    ppi_path = "./data/protein.actions.SHS27k.STRING.txt"
    pseq_path = "./data/protein.SHS27k.sequences.dictionary.tsv"

    vec_path = "./data/vec5_CTC.txt"

    index_path = "./train_valid_index_json/train_valid_index_json.json"
    model = "./save_model/gnn_test_27k_random/gnn_model_valid_best.ckpt"

    bigger_ppi_path = "./data/protein.actions.SHS148k.STRING.txt"
    bigger_pseq_path = "./data/protein.SHS148k.sequences.dictionary.tsv"

    # bigger_ppi_path = "./data/9606.protein.actions.all_connected.txt"
    # bigger_pseq_path = "./data/protein.STRING_all_connected.sequences.dictionary.tsv"

    run_func(description, ppi_path, pseq_path, vec_path, index_path, model, bigger_ppi_path, bigger_pseq_path)
