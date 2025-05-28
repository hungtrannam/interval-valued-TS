def suggest_model_specific(trial, model):
    d_model_choices = [8, 16, 32, 64, 128]
    d_ff_choices = [8, 16, 32, 64, 128]
    n_heads_choices = [1, 2, 4, 8]
    e_layers_range = (1, 8)
    d_layers_range = (1, 8)
    factor_range = (1, 3)
    patch_len_choices = [4, 8, 16, 32, 64]
    p_hidden_dims = [[256, 256]]
    p_hidden_layers_range = [2,4,8,16]
    top_k = [5,6,7,8,9,10]


    params = {}

    def suggest_d_model_and_heads():
        # Ensure d_model % n_heads == 0
        while True:
            d_model = trial.suggest_categorical('d_model', d_model_choices)
            n_heads = trial.suggest_categorical('n_heads', n_heads_choices)
            if d_model % n_heads == 0:
                return d_model, n_heads

    if model == 'PatchTST':
        d_model, n_heads = suggest_d_model_and_heads()
        params = {
            'd_model': d_model,
            'n_heads': n_heads,
            'e_layers': trial.suggest_int('e_layers', *e_layers_range),
            'd_layers': trial.suggest_int('d_layers', *d_layers_range),
            'd_ff': trial.suggest_categorical('d_ff', d_ff_choices),
            'factor': trial.suggest_int('factor', *factor_range),
            'pactch_len': trial.suggest_categorical('patch_len', patch_len_choices)
        }

    elif model in ['LSTM', 'GRU']:
        params = {
            'e_layers': trial.suggest_int('e_layers', *e_layers_range),
            'd_layers': trial.suggest_int('d_layers', *d_layers_range),
            'd_model': trial.suggest_categorical('d_model', d_ff_choices),
        }

    elif model == 'Nonstationary_Transformer':
        params = {
            'd_mmodel': d_model,
            'e_layers': trial.suggest_int('e_layers', *e_layers_range),
            'd_layers': trial.suggest_int('d_layers', *d_layers_range),
            'factor': trial.suggest_int('factor', factor_range),
            'p_hidden_dims': trial.suggest_int('factor', p_hidden_dims),
            'p_hidden_layers': trial.suggest_int('factor', p_hidden_layers_range),
        }

    elif model == 'DLinear':
        d_model, n_heads = suggest_d_model_and_heads()
        params = {
            'd_model': d_model,
            'n_heads': n_heads,
            'e_layers': trial.suggest_int('e_layers', *e_layers_range),
            'd_layers': trial.suggest_int('d_layers', *d_layers_range),
            'd_ff': trial.suggest_categorical('d_ff', d_ff_choices),
            'factor': trial.suggest_int('factor', *factor_range),
        }
    elif model == "FEDformer":
        params = {
            'd_model': d_model,
            'n_heads': n_heads,
            'e_layers': trial.suggest_int('e_layers', *e_layers_range),
            'd_layers': trial.suggest_int('d_layers', *e_layers_range),
            'd_ff': trial.suggest_categorical('d_ff', d_ff_choices),
            'factor': trial.suggest_int('factor', *factor_range),
            'top_k': trial.suggest_int('top_k', *top_k),
        }
    elif model == "TimeNet":
        params = {
            'd_model': d_model,
            'n_heads': n_heads,
            'e_layers': trial.suggest_int('e_layers', *e_layers_range),
            'd_layers': trial.suggest_int('d_layers', *d_layers_range),
            'd_ff': trial.suggest_categorical('d_ff', d_ff_choices),
            'factor': trial.suggest_int('factor', *factor_range),
        }
    elif model == "iTransformer":
        params = {
            'd_model': d_model,
            'n_heads': n_heads,
            'e_layers': trial.suggest_int('e_layers', *e_layers_range),
            'd_layers': trial.suggest_int('d_layers', *d_layers_range),
            'd_ff': trial.suggest_categorical('d_ff', d_ff_choices),
            'factor': trial.suggest_int('factor', *factor_range),
        }
    return params
