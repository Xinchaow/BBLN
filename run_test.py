import os

os.environ["PATH"] = "D:\Anaconda\envs\Python38"
def run_func(description, ppi_path, pseq_path, vec_path,
            index_path, model, test_all):
    os.system("python test.py \
            --description={} \
            --ppi_path={} \
            --pseq_path={} \
            --vec_path={} \
            --index_path={} \
            --model={} \
            --test_all={} \
            ".format(description, ppi_path, pseq_path, vec_path,
                    index_path, model, test_all))

if __name__ == "__main__":
    description = "test"

    ppi_path = "./data/protein.actions.SHS27k.STRING.txt"
    pseq_path = "./data/protein.SHS27k.sequences.dictionary.tsv"
    vec_path = "./data/vec5_CTC.txt"

    index_path = "./train_valid_index_json/train_valid_index_json.json"
    model = "./save_model/gnn_test_27k_random/gnn_model_valid_best.ckpt"

    test_all = "False"

    # test test

    run_func(description, ppi_path, pseq_path, vec_path, index_path, model, test_all)
