from GNNTE import run_GNNTE_experiment

if __name__ == "__main__":
    t1 = {  'project_name' : 'GNNTE',
            'dataset' :"/home/francesco.pugnaloni/wikipedia_tables/training_data/100000_samples",
            'lr' : 0.001,
            'batch_size' : 128,
            'num_epochs' : 50,
            'out_channels' : 300,
            'n_layers' : 3,
            'dropout' : 0.1,
            'n_sample':'100k',
            'weight_decay': 0,
            'step_size': 5,
            'gamma': 0.75,
            'gnn_type': 'GAT'
    }


    l = [t1]
    for i in range(len(l)):
        print(f'Test number: {i+1} / {len(l)}')
        run_GNNTE_experiment(**l[i])