from src.train import run_training

if __name__ == "__main__" :
    config = {"latent_dim" : 2,
            "beta" : 0.005,
            "epochs" : 500,
            "batch_size" : 256,
            "learning_rate" : 1e-4,
            "data_path" : "data/DataRe40.hdf5",
            "dataset_name" : "UV",
            "input_shape" : (2, 88, 300)
            }

    print("Start training ...")
    run_training(config)

