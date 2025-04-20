import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from noise import injectGaussianNoise

def feature_testing(GLOBAL_RANDOM_STATE, maj_var, label_ratio, d_means, var_ratio):
    np.random.seed(GLOBAL_RANDOM_STATE)
    num_samples = 500
    # maj_var = 1.0        # variance for the majority (label 0)
    # label_ratio = 3      # number of majority labels per minority label
    # d_means = maj_var / 2
    # var_ratio = 2        # variance multiplier for minority labels

    total_majority = num_samples * label_ratio  # for label 0
    total_minority = num_samples                # for label 1
    total_samples = total_majority + total_minority

    labels = np.concatenate([np.zeros(total_majority), np.ones(total_minority)])

    feature_1_majority = np.random.normal(0, scale = maj_var, size = total_majority)
    feature_1_minority = np.random.normal(1, scale = maj_var, size = total_minority)
    feature_1 = np.concatenate([feature_1_majority, feature_1_minority])

    feature_2_majority = np.random.normal(loc=0, scale=maj_var, size=total_majority)
    feature_2_minority = np.random.normal(loc=d_means, scale=maj_var * var_ratio, size=total_minority)
    feature_2 = np.concatenate([feature_2_majority, feature_2_minority])

    feature_3 = np.ones(shape= total_samples)

    feature_4_majority = np.random.normal(loc=0, scale=maj_var * var_ratio, size=total_majority)
    feature_4_minority = np.random.normal(loc=1, scale=maj_var * var_ratio, size=total_minority)
    feature_4 = np.concatenate([feature_4_majority, feature_4_minority])

    feature_5 = np.random.normal(0, 1, size = total_samples)

    feature_6_majority = np.random.normal(loc=0, scale=maj_var*var_ratio, size=total_majority)
    feature_6_minority = np.random.normal(loc=d_means, scale=maj_var, size=total_minority)
    feature_6 = np.concatenate([feature_6_majority, feature_6_minority])

    # Build the DataFrame
    df = pd.DataFrame({
        "Clean Important Feature w/ Lower Variance": feature_1,
        "Dirty Important Feature": feature_2,
        "Constant": feature_3, 
        "Clean Important Feature w/ Higher Variance": feature_4,
        "Noise": feature_5, 
        "label": labels,
        "Dirty Important Feature Reversed": feature_6
    })


    y = df['label']
    X = df.drop(columns=['label'])

    # Define parameters
    batch_sizes = [1]  # Different batch sizes
    num_trials = 20  # Number of trials per batch size
    CHOICE_RANDOMS = np.random.randint(0, 2^31, num_trials)
    noise_levels = [0.125, 0.25, 0.5, 0.75, 1.0, 2.0, 4.0]  # Standard deviations

    # DataFrame to store results
    results = []

    # Loop over different batch sizes
    for batch_size in batch_sizes:
        for trial in range(num_trials):
            np.random.seed(CHOICE_RANDOMS[trial])
            sampled_features = np.random.choice(X.columns, batch_size, replace=False)
            X_subset = X[sampled_features]

            X_train, X_test, y_train, y_test = train_test_split(X_subset, y, random_state=CHOICE_RANDOMS[trial], test_size=0.25)

            model = RandomForestClassifier(n_estimators = 10, max_depth = 15, criterion="gini", random_state=CHOICE_RANDOMS[trial])
            model.fit(X_train, y_train.values.ravel())

            importance_features = {sampled_features[i]: float(model.feature_importances_[i]) for i in range(len(sampled_features))}

            y_pred = model.predict(X_test)
            baseline_accuracy = accuracy_score(y_test, y_pred)
            results.append([trial, batch_size, "Baseline", baseline_accuracy, 0, 0, 1, 0, 0, 1])

            for feature in sampled_features:
                for noise_std in noise_levels:
                    ft_noise_accuracy_sum = 0
                    ft_noise_importance_sum = 0
                    for seed in CHOICE_RANDOMS:
                        np.random.seed(seed)
                        X_train_noisy = X_train.copy()
                        X_test_noisy = X_test.copy()
                        model = RandomForestClassifier(n_estimators = 10, max_depth = 15, criterion="entropy", random_state=seed)

                        if feature == 'Constant':
                            X_train_noisy['Constant'] = np.random.normal(1, noise_std, int(0.75*num_samples*(label_ratio + 1)))
                        else:
                            injectGaussianNoise(X_train_noisy, [feature], 0, noise_std, GLOBAL_RANDOM_STATE)

                        np.random.seed(seed)
                        model.fit(X_train_noisy, y_train.values.ravel())
                        y_pred = model.predict(X_test_noisy)
                        trial_accuracy = accuracy_score(y_test, y_pred)

                        ft_noise_accuracy_sum += trial_accuracy

                        curr_importance_features = {sampled_features[i]: float(model.feature_importances_[i]) for i in range(len(sampled_features))}
                        ft_noise_importance_sum += curr_importance_features[feature]

                    prev_importance = importance_features[feature]
                    np.random.seed(GLOBAL_RANDOM_STATE)

                    new_importance = ft_noise_importance_sum/num_trials
                    trial_accuracy = ft_noise_accuracy_sum/num_trials
                    if prev_importance == 0:
                        results.append([trial, batch_size, feature, trial_accuracy, noise_std, prev_importance, "Infinity", prev_importance - new_importance, baseline_accuracy - trial_accuracy, trial_accuracy/baseline_accuracy])
                    else:
                        results.append([trial, batch_size, feature, trial_accuracy, noise_std, prev_importance, new_importance/prev_importance, prev_importance - new_importance, baseline_accuracy - trial_accuracy, trial_accuracy/baseline_accuracy])

    results_df = pd.DataFrame(results, columns=["Trial Number", "Batch Size", "Feature Noised", "Accuracy", "Noise Std Dev", "Feature Importance", "Importance Ratio", "Importance Difference", "Accuracy Difference", "Accuracy Ratio"])
    results_df.to_csv('manualDataTrials_{}_{}_{}_{}.csv'.format(maj_var, label_ratio, d_means, var_ratio), index=False)

feature_testing(GLOBAL_RANDOM_STATE=13434323, maj_var=0.1, label_ratio=9, d_means= 0.2, var_ratio = 10)

