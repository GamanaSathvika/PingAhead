    print((df.groupby('scenario').is_attack.sum() * window_sec))            # attack seconds should be roughly constant
    print(df[TARGET].sum(), df['is_pre_episode_window'].sum())   
