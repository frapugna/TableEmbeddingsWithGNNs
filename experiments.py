from GNNTE import run_GNNTE_experiment

if __name__ == "__main__":
    t1 = {  'project_name' : 'GNNTE',
            'dataset' :"/home/francesco.pugnaloni/wikipedia_tables/small_tables/10000_samples",
            'lr' : 0.001,
            'batch_size' : 64,
            'num_epochs' : 200,
            'out_channels' : 300,
            'n_layers' : 3,
            'dropout' : 0
    }

    l = [t1]
    for i in range(len(l)):
        print(f'Test number: {i+1} / {len(l)}')
        run_GNNTE_experiment(**l[i])