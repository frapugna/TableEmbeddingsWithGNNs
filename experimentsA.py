from GNNTE import run_GNNTE_experiment

if __name__ == "__main__":
    t1 = {  'project_name' : 'GNNTE',
            'dataset' :"/home/francesco.pugnaloni/wikipedia_tables/training_data/100000_samples",
            'lr' : 0.001,
            'batch_size' : 128,
            'num_epochs' : 100,
            'out_channels' : 300,
            'n_layers' : 2,
            'dropout' : 0,
            'n_sample':'100k',
            'weight_decay': 0,
            'step_size': 10,
            'gamma': 0.75
    }
    t2 = {  'project_name' : 'GNNTE',
            'dataset' :"/home/francesco.pugnaloni/wikipedia_tables/training_data/100000_samples",
            'lr' : 0.001,
            'batch_size' : 128,
            'num_epochs' : 100,
            'out_channels' : 300,
            'n_layers' : 3,
            'dropout' : 0,
            'n_sample':'100k',
            'weight_decay': 0,
            'step_size': 10,
            'gamma': 0.75
    }
    t3 = {  'project_name' : 'GNNTE',
            'dataset' :"/home/francesco.pugnaloni/wikipedia_tables/training_data/100000_samples",
            'lr' : 0.001,
            'batch_size' : 128,
            'num_epochs' : 100,
            'out_channels' : 300,
            'n_layers' : 4,
            'dropout' : 0,
            'n_sample':'100k',
            'weight_decay': 0,
            'step_size': 10,
            'gamma': 0.75
    }

    l = [t1, t2, t3]
    for i in range(len(l)):
        print(f'Test number: {i+1} / {len(l)}')
        run_GNNTE_experiment(**l[i])