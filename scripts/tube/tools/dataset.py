from .cleaner import Cleaner


def prepare_dataset(name, df_X, df_Y, ext=[]):

    A_val = "A_{0}".format(name)
    B_val = "B_{0}".format(name)

    dataset = df_X[ ['A_rate', 'B_rate', A_val] ]
    dataset[ B_val ] = df_Y[B_val]

    if ext:
        for name_ext in ext:
            dataset[ name_ext ] = df_X[ name_ext ]
    
    return dataset


def prepare_eval_dataset(name, df_X):

    A_val = "A_{0}".format(name)

    dataset = df_X[ ['A_rate', 'B_rate', A_val] ]
    
    return dataset    


def clear_train_dataset(df_X, df_Y):

    cleaner = Cleaner()

    df_X = cleaner.clean(df_X)
    df_Y = cleaner.clean(df_Y)

    df_Y = df_Y['2020-01-01 04:30:00':]

    start_trash = df_X.index.get_loc('2020-01-24').start
    end_trash = df_X.index.get_loc('2020-02-13').start

    df_X = df_X.drop( df_X.index[ start_trash:end_trash ] )
    df_Y = df_Y.drop( df_Y.index[ start_trash:end_trash ] )


    start_trash = df_X.index.get_loc('2020-04-08').start
    end_trash = df_X.index.get_loc('2020-04-12').start

    df_X = df_X.drop( df_X.index[ start_trash:end_trash ] )
    df_Y = df_Y.drop( df_Y.index[ start_trash:end_trash ] )

    return df_X, df_Y