%% ========================================================================
%  SCRIPT 3: CLASSIFICATION & EVALUATION
%  STAGE 6: Baseline Multi-Class MLP Training
%  PUSL3123 Coursework - User Authentication via Gait Analysis
%% ========================================================================
%
%  PURPOSE:
%  Train ONE multi-class feed-forward neural network (MLP) to classify
%  users 1-10 based on gait features. This is NOT 10 binary classifiers.
%
%  STAGES IN THIS SCRIPT:
%  - Stage 6: Train baseline MLP on combined variant
%  - Stage 7: Evaluate FAR/FRR/EER (to be added next)
%  - Stage 8: Optimization experiments (to be added later)
%
%% ========================================================================

clear; clc; close all;

fprintf('========================================\n');
fprintf('SCRIPT 3: CLASSIFICATION & EVALUATION\n');
fprintf('========================================\n\n');

%% ========================================================================
%  STAGE 6: BASELINE MULTI-CLASS MLP TRAINING
%% ========================================================================

fprintf('========================================\n');
fprintf('STAGE 6: BASELINE MLP TRAINING\n');
fprintf('========================================\n\n');

% ----------------------------------------------------------------
% CONFIGURATION
% ----------------------------------------------------------------

% Set random seed for reproducibility
rng(42, 'twister');

% Set overlap tag used to create templates (string '0%' or '50%')
overlap_tag = '0%';   % update when using 50% overlap templates

% Input template file (combined variant; feature count auto-detected at load)
template_file = 'templates/combined/AllTemplates_combined.mat';

% Output model folder
model_folder = 'models/combined/';

% Network architecture (baseline)
hidden_layers = [128, 64];  % Two hidden layers

fprintf('Configuration:\n');
fprintf('  Input file: %s\n', template_file);
fprintf('  Hidden layers: [%s]\n', num2str(hidden_layers));
fprintf('  Output neurons: 10 (one per user)\n');
fprintf('  Random seed: 42\n\n');

% ----------------------------------------------------------------
% LOAD DATA
% ----------------------------------------------------------------

fprintf('Loading training data...\n');

load(template_file, 'X', 'y_user', 'meta');

fprintf('  ✓ Data loaded successfully\n');
fprintf('  ✓ Samples: %d\n', size(X, 1));
fprintf('  ✓ Features: %d\n', size(X, 2));
fprintf('  ✓ Classes: %d (users 1-10)\n\n', length(unique(y_user)));

% ----------------------------------------------------------------
% PREPARE DATA FOR PATTERNNET
% ----------------------------------------------------------------

% patternnet expects:
% - Input: [features × samples]
% - Target: [classes × samples] ONE-HOT ENCODED

X_train = X';  % Transpose: [features × samples]

% Convert labels (1-10) to one-hot encoding [10 × samples]
y_train_onehot = full(ind2vec(y_user'));  % [classes × samples]

% Verify data dimensions
fprintf('Prepared data dimensions:\n');
fprintf('  Input (X_train): [%d × %d] = [features × samples]\n', size(X_train, 1), size(X_train, 2));
fprintf('  Target (y_train_onehot): [%d × %d] = [classes × samples]\n', size(y_train_onehot, 1), size(y_train_onehot, 2));
fprintf('  Label range (original): [%d, %d]\n\n', min(y_user), max(y_user));

% ----------------------------------------------------------------
% CREATE MULTI-CLASS MLP (PATTERNNET)
% ----------------------------------------------------------------

fprintf('Creating multi-class MLP...\n');

% Create pattern recognition network
net = patternnet(hidden_layers, 'trainscg');  % Scaled Conjugate Gradient

% Configure training parameters
net.trainParam.epochs = 1000;           % Maximum epochs
net.trainParam.max_fail = 20;           % Early stopping patience
net.trainParam.showWindow = false;      % Disable training GUI
net.trainParam.showCommandLine = false; % Disable verbose output

% Configure data division (default: 70% train, 15% val, 15% test)
% Use ALL provided data for training (no validation/testing)
net.divideFcn = 'dividetrain';

fprintf('  ✓ Network created\n');
fprintf('  ✓ Architecture: %d inputs → [%s] hidden → %d outputs\n', ...
        size(X_train, 1), num2str(hidden_layers), length(unique(y_user)));
fprintf('  ✓ Training algorithm: Scaled Conjugate Gradient\n\n');

% ----------------------------------------------------------------
% TRAIN THE NETWORK
% ----------------------------------------------------------------

fprintf('Training baseline MLP...\n');

tic;
[net, tr] = train(net, X_train, y_train_onehot);
training_time = toc;

fprintf('  ✓ Training complete in %.2f seconds\n', training_time);
fprintf('  ✓ Best epoch: %d (validation performance: %.4f)\n', tr.best_epoch, tr.best_perf);
fprintf('  ✓ Final epochs: %d\n', tr.num_epochs);
fprintf('  ✓ Stop reason: %s\n\n', tr.stop);

% ----------------------------------------------------------------
% SAVE TRAINED MODEL
% ----------------------------------------------------------------

fprintf('Saving trained model...\n');

% Prepare metadata
training_info = struct();
training_info.architecture = hidden_layers;
training_info.num_features = size(X, 2);
training_info.num_classes = length(unique(y_user));
training_info.num_samples = size(X, 1);
training_info.training_time = training_time;
training_info.best_epoch = tr.best_epoch;
training_info.best_perf = tr.best_perf;
training_info.random_seed = 42;
training_info.variant = 'combined';
training_info.date_trained = datestr(now);

% Save model and metadata
model_file = 'baseline_mlp.mat';
model_path = fullfile(model_folder, model_file);
save(model_path, 'net', 'tr', 'training_info');

fprintf('  ✓ Model saved: %s\n\n', model_path);

% ----------------------------------------------------------------
% BASELINE TRAINING SUMMARY
% ----------------------------------------------------------------

fprintf('========================================\n');
fprintf('STAGE 6 SUMMARY\n');
fprintf('========================================\n');
fprintf('✓ Baseline MLP trained successfully\n');
fprintf('✓ Architecture: %d → [%s] → %d\n', ...
        size(X_train, 1), num2str(hidden_layers), length(unique(y_user)));
fprintf('✓ Training samples: %d\n', size(X, 1));
fprintf('✓ Features: %d (combined: accel + gyro, time + freq)\n', size(X, 2));
fprintf('✓ Classes: %d users\n', length(unique(y_user)));
fprintf('✓ Training time: %.2f seconds\n', training_time);
fprintf('✓ Model saved to: %s\n', model_path);
fprintf('========================================\n\n');

%% ========================================================================
%  STAGE 7: EVALUATION - FAR, FRR, EER (3 SCENARIOS)
%% ========================================================================

fprintf('\n========================================\n');
fprintf('STAGE 7: EVALUATION (3 SCENARIOS)\n');
fprintf('========================================\n\n');

% ----------------------------------------------------------------
% CONFIGURATION
% ----------------------------------------------------------------

% Results folder
results_folder = 'results/combined/';

% Reload full dataset
load(template_file, 'X', 'y_user', 'meta');

% Scenario names
scenarios = {'Same-Day', 'Cross-Day', 'Combined'};

% Storage for results
all_results = [];

% Fixed RNG for reproducibility
rng(42, 'twister');

%% ========================================================================
%  SCENARIO 1: SAME-DAY (Day1 train/test split)
%% ========================================================================

fprintf('--------------------------------------------------\n');
fprintf('SCENARIO 1: SAME-DAY (Day1 70/30 split)\n');
fprintf('--------------------------------------------------\n');

% Filter Day1 samples only
day1_idx = find(meta.day_num == 1);
X_day1 = X(day1_idx, :);
y_day1 = y_user(day1_idx);

% Stratified 70/30 split (maintain class balance)
cv = cvpartition(y_day1, 'HoldOut', 0.3);
train_idx = training(cv);
test_idx = test(cv);

% --- Prepare raw train/test matrices (no transpose yet) ---
X_train_s1_mat = X_day1(train_idx, :);   % [N_train × features]
y_train_s1 = y_day1(train_idx);
X_test_s1_mat  = X_day1(test_idx, :);    % [N_test × features]
y_test_s1 = y_day1(test_idx);

% --- GLOBAL Z-SCORE NORMALIZATION (train-only stats) ---
mu = mean(X_train_s1_mat, 1);                  % [1 × features]
sigma = std(X_train_s1_mat, 0, 1);             % [1 × features]
sigma(sigma == 0) = 1;                         % avoid divide-by-zero

X_train_s1 = ((X_train_s1_mat - mu) ./ sigma)';  % transpose to [features × N_train]
X_test_s1  = ((X_test_s1_mat  - mu) ./ sigma)';  % transpose to [features × N_test]

fprintf('  Train samples: %d, Test samples: %d\n', size(X_train_s1, 2), size(X_test_s1, 2));

% Train MLP
net_s1 = patternnet(hidden_layers, 'trainscg');
net_s1.trainParam.showWindow = false;
net_s1.trainParam.showCommandLine = false;

% Disable internal division — we already provided X_train_s1
net_s1.divideFcn = 'divideind';
net_s1.divideParam.trainInd = 1:size(X_train_s1,2);
net_s1.divideParam.valInd   = [];
net_s1.divideParam.testInd  = [];

y_train_s1_onehot = full(ind2vec(y_train_s1'));
[net_s1, ~] = train(net_s1, X_train_s1, y_train_s1_onehot);

% Predictions
predictions_s1 = net_s1(X_test_s1);  % [10 × N_test]

% Compute metrics
[far_s1, frr_s1, eer_s1, acc_s1] = compute_authentication_metrics(predictions_s1, y_test_s1);

fprintf('  ✓ FAR: %.4f, FRR: %.4f, EER: %.4f, Accuracy: %.4f\n\n', far_s1, frr_s1, eer_s1, acc_s1);

% Save results
results_s1 = table({'Same-Day'}, far_s1, frr_s1, eer_s1, acc_s1, ...
    'VariableNames', {'Scenario', 'FAR', 'FRR', 'EER', 'Accuracy'});
writetable(results_s1, fullfile(results_folder, 'scenario1_sameday_metrics.csv'));
all_results = [all_results; results_s1];

%% ========================================================================
%  SCENARIO 2: CROSS-DAY (Day1 → Day2, most realistic)
%% ========================================================================

fprintf('--------------------------------------------------\n');
fprintf('SCENARIO 2: CROSS-DAY (Train Day1 → Test Day2)\n');
fprintf('--------------------------------------------------\n');

% Filter Day1 for training, Day2 for testing
day1_idx = find(meta.day_num == 1);
day2_idx = find(meta.day_num == 2);



X_train_s2_mat = X(day1_idx, :);
y_train_s2 = y_user(day1_idx);
X_test_s2_mat  = X(day2_idx, :);
y_test_s2 = y_user(day2_idx);

% Global z-score (train-only)
mu = mean(X_train_s2_mat, 1);
sigma = std(X_train_s2_mat, 0, 1);
sigma(sigma == 0) = 1;

X_train_s2 = ((X_train_s2_mat - mu) ./ sigma)';   % [features × N]
X_test_s2  = ((X_test_s2_mat  - mu) ./ sigma)';   % [features × N]




fprintf('  Train samples (Day1): %d, Test samples (Day2): %d\n', size(X_train_s2, 2), size(X_test_s2, 2));

% Train MLP
net_s2 = patternnet(hidden_layers, 'trainscg');
net_s2.trainParam.showWindow = false;
net_s2.trainParam.showCommandLine = false;

net_s2.divideFcn = 'divideind';
net_s2.divideParam.trainInd = 1:size(X_train_s2,2);
net_s2.divideParam.valInd   = [];
net_s2.divideParam.testInd  = [];

y_train_s2_onehot = full(ind2vec(y_train_s2'));
[net_s2, ~] = train(net_s2, X_train_s2, y_train_s2_onehot);

% Predictions
predictions_s2 = net_s2(X_test_s2);  % [10 × N_test]

% Compute metrics
[far_s2, frr_s2, eer_s2, acc_s2] = compute_authentication_metrics(predictions_s2, y_test_s2);


%% ------------------------------------------------------------
%% PER-USER METRICS (Cross-Day) — Minimal Insert Block
%% ------------------------------------------------------------
num_users = 10;
user_metrics = zeros(num_users, 3);  % FAR, FRR, EER

scores = predictions_s2';   % [N_test × 10]

thresholds = linspace(0,1,5001);

for u = 1:num_users
    s = scores(:, u);
    genuine = s(y_test_s2 == u);
    impostor = s(y_test_s2 ~= u);

    far_curve = zeros(length(thresholds),1);
    frr_curve = zeros(length(thresholds),1);

    for i = 1:length(thresholds)
        thr = thresholds(i);
        if ~isempty(genuine)
            frr_curve(i) = mean(genuine < thr);
        end
        if ~isempty(impostor)
            far_curve(i) = mean(impostor >= thr);
        end
    end

    diff_curve = abs(far_curve - frr_curve);
    [~, idx_min] = min(diff_curve);
    eer_u = (far_curve(idx_min) + frr_curve(idx_min)) / 2;

    user_metrics(u,1) = mean(far_curve);
    user_metrics(u,2) = mean(frr_curve);
    user_metrics(u,3) = eer_u;
end

user_ids = (1:num_users)';

per_user_table = table(user_ids, ...
                       user_metrics(:,1), ...
                       user_metrics(:,2), ...
                       user_metrics(:,3), ...
        'VariableNames', {'User','FAR','FRR','EER'});

writetable(per_user_table, fullfile(results_folder, 'per_user_crossday_metrics.csv'));

fprintf('\n=============================================\n');
fprintf(' PER-USER CROSS-DAY METRICS (FAR / FRR / EER)\n');
fprintf('=============================================\n');
disp(per_user_table);
fprintf('✓ Saved: results/combined/per_user_crossday_metrics.csv\n\n');

fprintf('  ✓ FAR: %.4f, FRR: %.4f, EER: %.4f, Accuracy: %.4f\n\n', far_s2, frr_s2, eer_s2, acc_s2);

%% ======================================================
%% THRESHOLD SWEEP FOR CROSS-DAY (For EER Plotting)
%% ======================================================

thresholds = 0:0.001:1;
num_users = 10;

far_per_user = zeros(num_users, length(thresholds));
frr_per_user = zeros(num_users, length(thresholds));

scores = predictions_s2';        % [N_test x 10]
labels = y_test_s2;              % [N_test x 1]

% Compute FAR/FRR across thresholds
for u = 1:num_users
    user_scores = scores(:, u);
    genuine = user_scores(labels == u);
    impostor = user_scores(labels ~= u);

    for t = 1:length(thresholds)
        thr = thresholds(t);
        far_per_user(u, t) = mean(impostor >= thr);
        frr_per_user(u, t) = mean(genuine < thr);
    end
end

% Average curves
far_curve = mean(far_per_user, 1);
frr_curve = mean(frr_per_user, 1);

% Compute EER
[~, eer_idx] = min(abs(far_curve - frr_curve));
eer_threshold = thresholds(eer_idx);
eer_value = far_curve(eer_idx);

%% Save plot data for external plotting
plot_data = struct();
plot_data.thresholds = thresholds;
plot_data.far_curve = far_curve;
plot_data.frr_curve = frr_curve;
plot_data.far_per_user = far_per_user;
plot_data.frr_per_user = frr_per_user;
plot_data.eer_value = eer_value;
plot_data.eer_threshold = eer_threshold;
plot_data.eer_idx = eer_idx;
plot_data.scenario = 'Cross-Day';
plot_data.overlap = overlap_tag;         %change this to 0-50 when swtiching between overlap versions
plot_data.num_users = 10;

save('results/combined/scenario2_crossday_plot_data.mat', 'plot_data');
fprintf('  ✓ Threshold sweep plot data saved.\n');

% Save results
results_s2 = table({'Cross-Day'}, far_s2, frr_s2, eer_s2, acc_s2, ...
    'VariableNames', {'Scenario', 'FAR', 'FRR', 'EER', 'Accuracy'});
writetable(results_s2, fullfile(results_folder, 'scenario2_crossday_metrics.csv'));
all_results = [all_results; results_s2];

%% ========================================================================
%  SCENARIO 2B: CROSS-DAY (Gait-Only)
%% ========================================================================

fprintf('--------------------------------------------------\n');
fprintf('SCENARIO 2B: CROSS-DAY (Gait-Only)\n');
fprintf('--------------------------------------------------\n');

load('templates/combined/AllTemplates_combined_gait.mat', 'X_gait', 'y_gait', 'meta_gait');

day1_idx_g = find(meta_gait.day_num == 1);
day2_idx_g = find(meta_gait.day_num == 2);



X_train_g_mat = X_gait(day1_idx_g, :);
y_train_g = y_gait(day1_idx_g);
X_test_g_mat  = X_gait(day2_idx_g, :);
y_test_g = y_gait(day2_idx_g);

% Global z-score (train-only)
mu = mean(X_train_g_mat, 1);
sigma = std(X_train_g_mat, 0, 1);
sigma(sigma == 0) = 1;

X_train_g = ((X_train_g_mat - mu) ./ sigma)';   % [features × N_train]
X_test_g  = ((X_test_g_mat  - mu) ./ sigma)';   % [features × N_test]




fprintf('  Train samples (Day1 gait): %d, Test samples (Day2 gait): %d\n', ...
        size(X_train_g,2), size(X_test_g,2));

net_g = patternnet(hidden_layers, 'trainscg');
net_g.trainParam.showWindow = false;
net_g.trainParam.showCommandLine = false;

net_g.divideFcn = 'divideind';
net_g.divideParam.trainInd = 1:size(X_train_g,2);
net_g.divideParam.valInd = [];
net_g.divideParam.testInd = [];

y_train_g_onehot = full(ind2vec(y_train_g'));
[net_g, ~] = train(net_g, X_train_g, y_train_g_onehot);

predictions_g = net_g(X_test_g);

[far_g, frr_g, eer_g, acc_g] = compute_authentication_metrics(predictions_g, y_test_g);

fprintf('  ✓ GAIT-ONLY EER: %.4f, Accuracy: %.4f\n\n', eer_g, acc_g);

results_g = table({'Cross-Day-Gait'}, far_g, frr_g, eer_g, acc_g, ...
    'VariableNames', {'Scenario','FAR','FRR','EER','Accuracy'});

writetable(results_g, fullfile(results_folder,'scenario2b_crossday_gait_metrics.csv'));
all_results = [all_results; results_g];

%% ========================================================================
%  SCENARIO 3: COMBINED RANDOM (Day1+Day2, 70/30 split)
%% ========================================================================

fprintf('--------------------------------------------------\n');
fprintf('SCENARIO 3: COMBINED (Day1+Day2, 70/30 split)\n');
fprintf('--------------------------------------------------\n');

% Use all data (Day1 + Day2)
X_all = X;
y_all = y_user;

% Stratified 70/30 split
cv = cvpartition(y_all, 'HoldOut', 0.3);
train_idx = training(cv);
test_idx = test(cv);


X_train_s3_mat = X_all(train_idx, :);
y_train_s3 = y_all(train_idx);
X_test_s3_mat  = X_all(test_idx, :);
y_test_s3 = y_all(test_idx);

% Global z-score (train-only)
mu = mean(X_train_s3_mat, 1);
sigma = std(X_train_s3_mat, 0, 1);
sigma(sigma == 0) = 1;

X_train_s3 = ((X_train_s3_mat - mu) ./ sigma)';   % [features × N_train]
X_test_s3  = ((X_test_s3_mat  - mu) ./ sigma)';   % [features × N_test]



fprintf('  Train samples: %d, Test samples: %d\n', size(X_train_s3, 2), size(X_test_s3, 2));

% Train MLP
net_s3 = patternnet(hidden_layers, 'trainscg');
net_s3.trainParam.showWindow = false;
net_s3.trainParam.showCommandLine = false;

net_s3.divideFcn = 'divideind';
net_s3.divideParam.trainInd = 1:size(X_train_s3,2);
net_s3.divideParam.valInd   = [];
net_s3.divideParam.testInd  = [];

y_train_s3_onehot = full(ind2vec(y_train_s3'));
[net_s3, ~] = train(net_s3, X_train_s3, y_train_s3_onehot);

% Predictions
predictions_s3 = net_s3(X_test_s3);  % [10 × N_test]

% Compute metrics
[far_s3, frr_s3, eer_s3, acc_s3] = compute_authentication_metrics(predictions_s3, y_test_s3);

fprintf('  ✓ FAR: %.4f, FRR: %.4f, EER: %.4f, Accuracy: %.4f\n\n', far_s3, frr_s3, eer_s3, acc_s3);

% Save results
results_s3 = table({'Combined'}, far_s3, frr_s3, eer_s3, acc_s3, ...
    'VariableNames', {'Scenario', 'FAR', 'FRR', 'EER', 'Accuracy'});
writetable(results_s3, fullfile(results_folder, 'scenario3_combined_metrics.csv'));
all_results = [all_results; results_s3];

%% ========================================================================
%  STAGE 7 SUMMARY
%% ========================================================================

fprintf('========================================\n');
fprintf('STAGE 7 EVALUATION SUMMARY\n');
fprintf('========================================\n\n');

% Display consolidated results table
disp(all_results);

fprintf('\n✓ Results saved to: %s\n', results_folder);
fprintf('========================================\n\n');

fprintf('Stage 7 complete. Ready for Stage 8 (Optimization).\n');

%% ========================================================================
%  STAGE 8: OPTIMIZATION (ALLOWED LEVERS ONLY)
%% ========================================================================

fprintf('\n========================================\n');
fprintf('STAGE 8: OPTIMIZATION EXPERIMENTS\n');
fprintf('========================================\n\n');

% ----------------------------------------------------------------
% CONFIGURATION
% ----------------------------------------------------------------

% Optimization results folder
opt_results_folder = 'results/optimization/';

% Reload full dataset
load(template_file, 'X', 'y_user', 'meta');

% Fixed RNG for reproducibility
rng(42, 'twister');

% Storage for all optimization results
opt_results = [];

% Baseline configuration (from Stage 7)
baseline_config = struct();
baseline_config.features = size(X, 2);        % auto-detect from loaded template X
baseline_config.architecture = [128, 64];     % leave or update as preferred
baseline_config.split_ratio = 0.7;
baseline_config.overlap = 0;                  % change manually when using 50% overlap

fprintf('Baseline configuration:\n');
fprintf('  Features: %d (all)\n', baseline_config.features);
fprintf('  Architecture: [%s]\n', num2str(baseline_config.architecture));
fprintf('  Train/Test split: %.0f/%.0f\n', baseline_config.split_ratio*100, (1-baseline_config.split_ratio)*100);
fprintf('  Window overlap: %d%%\n\n', baseline_config.overlap);

%% ========================================================================
%  LEVER 1: PER-USER FEATURE SELECTION (Rank-Average Top-N)
%% ========================================================================

fprintf('LEVER 1: Per-User Feature Selection (Rank-Averaged)\n');
fprintf('Testing: Top-30, Top-45, Top-60 features\n\n');

day1_idx = find(meta.day_num == 1);
day2_idx = find(meta.day_num == 2);

X_train_base = X(day1_idx, :);
y_train_base = y_user(day1_idx);

X_test_base = X(day2_idx, :);
y_test_base = y_user(day2_idx);

num_users = 10;
num_features = size(X_train_base,2);

R_all = zeros(num_users, num_features);

for u = 1:num_users
    idx_u = (y_train_base == u);
    y_binary = zeros(size(y_train_base));
    y_binary(idx_u) = 1;

    fisher_u = compute_fisher_scores(X_train_base, y_binary);
    [~, ranks_u] = sort(fisher_u, 'descend');

    R = zeros(1,num_features);
    R(ranks_u) = 1:num_features;
    R_all(u,:) = R;
end

R_avg = mean(R_all,1);

[~, ranked_features] = sort(R_avg, 'ascend');

feature_counts = [30 35 40 51 52 53 54 58 59 60 62 65];

for f = 1:length(feature_counts)

    n_features = feature_counts(f);
    fprintf('  Testing PerUser-Top-%d... ', n_features);

    selected_features = ranked_features(1:n_features);

    X_train_fs = X_train_base(:, selected_features)';
    X_test_fs  = X_test_base(:, selected_features)';

    net_fs = patternnet(baseline_config.architecture, 'trainscg');
    net_fs.trainParam.showWindow = false;
    net_fs.trainParam.showCommandLine = false;

    net_fs.divideFcn = 'divideind';
    net_fs.divideParam.trainInd = 1:size(X_train_fs,2);
    net_fs.divideParam.valInd   = [];
    net_fs.divideParam.testInd  = [];

    y_train_fs_onehot = full(ind2vec(y_train_base'));
    [net_fs, ~] = train(net_fs, X_train_fs, y_train_fs_onehot);

    predictions_fs = net_fs(X_test_fs);

    [far_fs, frr_fs, eer_fs, acc_fs] = compute_authentication_metrics(predictions_fs, y_test_base);

    fprintf('EER: %.4f, Accuracy: %.4f\n', eer_fs, acc_fs);

    opt_row = table({'PerUser-RankAvg'}, {sprintf('Top-%d', n_features)}, {'Cross-Day'}, ...
                    far_fs, frr_fs, eer_fs, acc_fs, n_features, {baseline_config.architecture}, ...
                    baseline_config.split_ratio, baseline_config.overlap, ...
                    'VariableNames', {'Lever','Setting','Scenario','FAR','FRR','EER','Accuracy', ...
                                      'Features_Used','Hidden_Layers','Split_Ratio','Overlap'});
    opt_results = [opt_results; opt_row];
end

fprintf('\n');


%% ========================================================================
%  LEVER 2: CLASSIFIER ARCHITECTURE
%% ========================================================================

fprintf('LEVER 2: Classifier Architecture\n');
fprintf('Testing: [64], [128], [128,64], [128,128]\n\n');

architectures = {[64], [128], [128, 64], [128, 128], [160, 80]};
arch_names = {'[64]', '[128]', '[128,64]', '[128,128]', '[160,80]'};

for a = 1:length(architectures)
    arch = architectures{a};
    arch_name = arch_names{a};
    
    fprintf('  Testing architecture %s... ', arch_name);
    
    % Use all features (baseline)
    X_train_arch = X_train_base';
    X_test_arch = X_test_base';
    
    % Train MLP
    net_arch = patternnet(arch, 'trainscg');
    net_arch.trainParam.showWindow = false;
    net_arch.trainParam.showCommandLine = false;
    
    net_arch.divideFcn = 'divideind';
    net_arch.divideParam.trainInd = 1:size(X_train_arch,2);
    net_arch.divideParam.valInd   = [];
    net_arch.divideParam.testInd  = [];
    
    y_train_arch_onehot = full(ind2vec(y_train_base'));
    [net_arch, ~] = train(net_arch, X_train_arch, y_train_arch_onehot);
    
    % Predictions
    predictions_arch = net_arch(X_test_arch);
    
    % Compute metrics
    [far_arch, frr_arch, eer_arch, acc_arch] = compute_authentication_metrics(predictions_arch, y_test_base);
    
    fprintf('EER: %.4f, Accuracy: %.4f\n', eer_arch, acc_arch);
    
    % Save results
    opt_row = table({'Architecture'}, {arch_name}, {'Cross-Day'}, ...
                    far_arch, frr_arch, eer_arch, acc_arch, baseline_config.features, {arch}, ...
                    baseline_config.split_ratio, baseline_config.overlap, ...
                    'VariableNames', {'Lever', 'Setting', 'Scenario', 'FAR', 'FRR', 'EER', ...
                                      'Accuracy', 'Features_Used', 'Hidden_Layers', 'Split_Ratio', 'Overlap'});
    opt_results = [opt_results; opt_row];
end

fprintf('\n');

%% ========================================================================
%  LEVER 3: TRAIN/TEST RATIO (Combined Scenario)
%% ========================================================================

fprintf('LEVER 3: Train/Test Split Ratio\n');
fprintf('Testing: 70/30 (baseline) vs 80/20 on Combined scenario\n\n');

split_ratios = [0.7, 0.8];
split_names = {'70/30', '80/20'};

for s = 1:length(split_ratios)
    split_ratio = split_ratios(s);
    split_name = split_names{s};
    
    fprintf('  Testing %s split... ', split_name);
    
    % Use all data (Combined scenario)
    X_all = X;
    y_all = y_user;
    
    % Stratified split
    cv = cvpartition(y_all, 'HoldOut', 1 - split_ratio);
    train_idx = training(cv);
    test_idx = test(cv);
    
    X_train_split = X_all(train_idx, :)';
    y_train_split = y_all(train_idx);
    X_test_split = X_all(test_idx, :)';
    y_test_split = y_all(test_idx);
    
    % Train MLP
    net_split = patternnet(baseline_config.architecture, 'trainscg');
    net_split.trainParam.showWindow = false;
    net_split.trainParam.showCommandLine = false;
    
    net_split.divideFcn = 'divideind';
    net_split.divideParam.trainInd = 1:size(X_train_split,2);
    net_split.divideParam.valInd   = [];
    net_split.divideParam.testInd  = [];
    
    y_train_split_onehot = full(ind2vec(y_train_split'));
    [net_split, ~] = train(net_split, X_train_split, y_train_split_onehot);
    
    % Predictions
    predictions_split = net_split(X_test_split);
    
    % Compute metrics
    [far_split, frr_split, eer_split, acc_split] = compute_authentication_metrics(predictions_split, y_test_split);
    
    fprintf('EER: %.4f, Accuracy: %.4f\n', eer_split, acc_split);
    
    % Save results
    opt_row = table({'Train/Test Ratio'}, {split_name}, {'Combined'}, ...
                    far_split, frr_split, eer_split, acc_split, baseline_config.features, ...
                    {baseline_config.architecture}, split_ratio, baseline_config.overlap, ...
                    'VariableNames', {'Lever', 'Setting', 'Scenario', 'FAR', 'FRR', 'EER', ...
                                      'Accuracy', 'Features_Used', 'Hidden_Layers', 'Split_Ratio', 'Overlap'});
    opt_results = [opt_results; opt_row];
end

fprintf('\n');

%% ========================================================================
%  LEVER 4: WINDOW OVERLAP (NOTE)
%% ========================================================================

fprintf('LEVER 4: Window Overlap\n');
fprintf('NOTE: Testing 50%% overlap requires re-running Script 1 (Stages 1-4)\n');
fprintf('      to regenerate segmented data with overlap=50%%.\n');

%% ========================================================================
%  SAVE OPTIMIZATION RESULTS
%% ========================================================================

% Save full optimization results
summary_file = fullfile(opt_results_folder, 'summary.csv');
writetable(opt_results, summary_file);

fprintf('========================================\n');
fprintf('STAGE 8 OPTIMIZATION SUMMARY\n');
fprintf('========================================\n\n');

fprintf('✓ Total optimization runs: %d\n', height(opt_results));
fprintf('✓ Results saved to: %s\n\n', summary_file);

% Find top-3 best configurations by EER
[~, sorted_idx] = sort(opt_results.EER);
top3 = opt_results(sorted_idx(1:min(3, height(opt_results))), :);

fprintf('Top 3 Best Configurations (by EER):\n');
fprintf('Rank | Lever                | Setting      | Scenario   | EER     | Accuracy\n');
fprintf('-----|----------------------|--------------|------------|---------|----------\n');
for i = 1:height(top3)
    fprintf('  %d  | %-20s | %-12s | %-10s | %.4f  | %.4f\n', ...
            i, top3.Lever{i}, top3.Setting{i}, top3.Scenario{i}, top3.EER(i), top3.Accuracy(i));
end

fprintf('\n========================================\n\n');

fprintf('Stage 8 complete. Ready for Stage 9 (Cross-Variant Evaluation).\n');


%% ========================================================================
%  OPTIONAL OPTIMIZATION: SVM CLASSIFIER (CROSS-DAY COMPARISON)
%  Allowed as a single alternative classifier for optimization.
%% ========================================================================

fprintf('\n========================================\n');
fprintf('OPTIONAL OPTIMIZATION: SVM CLASSIFIER\n');
fprintf('========================================\n\n');

% Reload full combined dataset (same as MLP)
load(template_file, 'X', 'y_user', 'meta');

% Cross-Day: Day1 → Train, Day2 → Test
day1_idx = find(meta.day_num == 1);
day2_idx = find(meta.day_num == 2);

X_train_svm_mat = X(day1_idx, :);
y_train_svm = y_user(day1_idx);

X_test_svm_mat  = X(day2_idx, :);
y_test_svm = y_user(day2_idx);

% Global z-score (train-only)
mu = mean(X_train_svm_mat, 1);
sigma = std(X_train_svm_mat, 0, 1);
sigma(sigma == 0) = 1;

X_train_svm = (X_train_svm_mat - mu) ./ sigma;
X_test_svm  = (X_test_svm_mat  - mu) ./ sigma;

fprintf('Training linear SVM (ECOC multi-class)...\n');

% ECOC multi-class SVM (Linear kernel)
template = templateSVM('KernelFunction','linear','Standardize',true);
svm_model = fitcecoc(X_train_svm, y_train_svm, 'Learners', template, 'Coding','onevsall');

fprintf('  ✓ SVM training complete\n');

% --------------------------------------------------------------------
%  SVM PREDICTIONS
% --------------------------------------------------------------------

[~, score_svm] = predict(svm_model, X_test_svm); 
% score_svm = [N_test × 10] → convert to [10 × N_test] for metric function
predictions_svm = score_svm';

% --------------------------------------------------------------------
%  METRICS (Reuse Existing MLP Metric Function)
% --------------------------------------------------------------------

[far_svm, frr_svm, eer_svm, acc_svm] = ...
     compute_authentication_metrics(predictions_svm, y_test_svm);

fprintf('  ✓ SVM FAR: %.4f, FRR: %.4f, EER: %.4f, Accuracy: %.4f\n\n', ...
        far_svm, frr_svm, eer_svm, acc_svm);

% Save SVM results
svm_results = table({'SVM'}, {'Cross-Day'}, far_svm, frr_svm, eer_svm, acc_svm, ...
    'VariableNames', {'Model','Scenario','FAR','FRR','EER','Accuracy'});

writetable(svm_results, fullfile('results/optimization/', 'svm_crossday_metrics.csv'));

fprintf('  ✓ SVM results saved to results/optimization/svm_crossday_metrics.csv\n');


%% ========================================================================
%  STAGE 9: CROSS-VARIANT EVALUATION (ACCEL / GYRO / COMBINED)
%% ========================================================================

fprintf('\n========================================\n');
fprintf('STAGE 9: CROSS-VARIANT EVALUATION\n');
fprintf('========================================\n\n');

% ----------------------------------------------------------------
% CONFIGURATION
% ----------------------------------------------------------------

% Variant comparison results folder
variant_results_folder = 'results/';
if ~exist(variant_results_folder, 'dir')
    mkdir(variant_results_folder);
end

% Fixed RNG for reproducibility
rng(42, 'twister');

% Best configuration from Stage 8
best_architecture = [128];  % final recommended default (can be changed)
scenario_name = 'Cross-Day';    % Most realistic scenario

fprintf('Configuration:\n');
fprintf('  Scenario: %s (most realistic)\n', scenario_name);
fprintf('  Architecture: [%s]\n', num2str(best_architecture));
fprintf('  Comparison: Accel-only vs Gyro-only vs Combined\n\n');

% Variant template files
variants = {'accel', 'gyro', 'combined'};
variant_files = {
    'templates/accel/AllTemplates_accel.mat'
    'templates/gyro/AllTemplates_gyro.mat'
    'templates/combined/AllTemplates_combined.mat'
};

% Storage for results
variant_results = [];

%% ========================================================================
%  EVALUATE EACH VARIANT
%% ========================================================================

fprintf('--------------------------------------------------\n');
fprintf('Evaluating Variants on %s Scenario\n', scenario_name);
fprintf('--------------------------------------------------\n\n');

for v = 1:length(variants)
    variant_name = variants{v};
    template_file_v = variant_files{v};
    
    fprintf('VARIANT %d: %s\n', v, upper(variant_name));
    
    % Load variant template
    if ~isfile(template_file_v)
        warning('Template file not found: %s. Skipping variant.', template_file_v);
        continue;
    end
    
    load(template_file_v, 'X', 'y_user', 'meta');
    
    fprintf('  Loaded: %d samples, %d features\n', size(X, 1), size(X, 2));
    
    % Cross-Day scenario: Train on Day1, Test on Day2
    day1_idx = find(meta.day_num == 1);
    day2_idx = find(meta.day_num == 2);
    
    X_train_v = X(day1_idx, :)';  % [features × N_day1]
    y_train_v = y_user(day1_idx);
    X_test_v = X(day2_idx, :)';   % [features × N_day2]
    y_test_v = y_user(day2_idx);
    
    fprintf('  Train samples (Day1): %d, Test samples (Day2): %d\n', ...
            size(X_train_v, 2), size(X_test_v, 2));
    
    % Train MLP with best architecture
    net_v = patternnet(best_architecture, 'trainscg');
    net_v.trainParam.epochs = 1000;
    net_v.trainParam.max_fail = 20;
    net_v.trainParam.showWindow = false;
    net_v.trainParam.showCommandLine = false;
    
    net_v.divideFcn = 'divideind';
    net_v.divideParam.trainInd = 1:size(X_train_v,2);
    net_v.divideParam.valInd   = [];
    net_v.divideParam.testInd  = [];
    
    % Convert labels to one-hot encoding
    y_train_v_onehot = full(ind2vec(y_train_v'));
    
    % Train network
    fprintf('  Training... ');
    tic;
    [net_v, ~] = train(net_v, X_train_v, y_train_v_onehot);
    train_time_v = toc;
    fprintf('done in %.2f seconds\n', train_time_v);
    
    % Predictions
    predictions_v = net_v(X_test_v);  % [classes × N_test]
    
    % Compute authentication metrics
    [far_v, frr_v, eer_v, acc_v] = compute_authentication_metrics(predictions_v, y_test_v);
    
    fprintf('  ✓ FAR: %.4f, FRR: %.4f, EER: %.4f, Accuracy: %.4f\n\n', ...
            far_v, frr_v, eer_v, acc_v);
    
    % Save results
    result_row = table({variant_name}, {scenario_name}, far_v, frr_v, eer_v, acc_v, size(X, 2), ...
                       'VariableNames', {'Variant', 'Scenario', 'FAR', 'FRR', 'EER', 'Accuracy', 'Features'});
    variant_results = [variant_results; result_row];
end

%% ========================================================================
%  SAVE VARIANT COMPARISON RESULTS
%% ========================================================================

% Save comparison CSV
comparison_file = fullfile(variant_results_folder, 'variants_comparison.csv');
writetable(variant_results, comparison_file);

fprintf('========================================\n');
fprintf('STAGE 9: VARIANT COMPARISON SUMMARY\n');
fprintf('========================================\n\n');

% Display results table
disp(variant_results);

fprintf('\n');

% Find best variant by EER
[min_eer, best_idx] = min(variant_results.EER);
best_variant = variant_results.Variant{best_idx};
best_acc = variant_results.Accuracy(best_idx);

fprintf('✓ Best Variant: %s (EER: %.4f, Accuracy: %.4f)\n', ...
        upper(best_variant), min_eer, best_acc);
fprintf('✓ Results saved to: %s\n', comparison_file);
fprintf('========================================\n\n');

fprintf('Stage 9 complete. Ready for Stage 10 (Final Packaging & Inventory).\n');

%% ========================================================================
%  HELPER FUNCTION: COMPUTE AUTHENTICATION METRICS (Interpolated EER)
%% ========================================================================
function [far_avg, frr_avg, eer_avg, accuracy] = compute_authentication_metrics(predictions, y_test)

    num_users = 10;

    % Fine threshold resolution
    thresholds = linspace(0,1,5001);

    far_per_user = zeros(num_users,1);
    frr_per_user = zeros(num_users,1);
    eer_per_user = zeros(num_users,1);

    for user = 1:num_users

        scores = predictions(user,:)';
        genuine_scores  = scores(y_test ==  user);
        impostor_scores = scores(y_test ~= user);

        far_curve = zeros(length(thresholds),1);
        frr_curve = zeros(length(thresholds),1);

        for i = 1:length(thresholds)
            thr = thresholds(i);
            if ~isempty(genuine_scores)
                frr_curve(i) = mean(genuine_scores < thr);
            end
            if ~isempty(impostor_scores)
                far_curve(i) = mean(impostor_scores >= thr);
            end
        end

        diff_curve = far_curve - frr_curve;

        idx = find(diff_curve .* circshift(diff_curve, -1) < 0, 1);
    
        % ============================================================
        % CASE 1 — No crossing found OR crossing at the final element
        % ============================================================
        if isempty(idx) || idx == length(thresholds)
            % fallback: choose point of minimum absolute difference
            [~, idx_min] = min(abs(diff_curve));
            eer_user = (far_curve(idx_min) + frr_curve(idx_min)) / 2;
        
        else
        % ============================================================
        % CASE 2 — Valid crossing → interpolate EER
        % ============================================================
            x1 = thresholds(idx);
            x2 = thresholds(idx+1);
    
            y1 = diff_curve(idx);
            y2 = diff_curve(idx+1);
    
            % threshold where FAR-FRR = 0
            eer_thr = x1 + (0 - y1) * (x2 - x1) / (y2 - y1);
    
            % interpolated FAR/FRR at EER
            eer_far = interp1(thresholds, far_curve, eer_thr);
            eer_frr = interp1(thresholds, frr_curve, eer_thr);
    
            eer_user = (eer_far + eer_frr) / 2;
        end

        far_per_user(user) = mean(far_curve);
        frr_per_user(user) = mean(frr_curve);
        eer_per_user(user) = eer_user;
    end

    far_avg = mean(far_per_user);
    frr_avg = mean(frr_per_user);
    eer_avg = mean(eer_per_user);

    [~, predicted_class] = max(predictions, [], 1);
    accuracy = mean(predicted_class' == y_test);

end


%% ========================================================================
%  HELPER FUNCTION: COMPUTE FISHER SCORES
%% ========================================================================
function fisher_scores = compute_fisher_scores(X, y)

    [N, D] = size(X);
    classes = unique(y);
    C = length(classes);

    fisher_scores = zeros(D,1);

    mu_total = mean(X,1);

    for d = 1:D
        mu_c = zeros(C,1);
        var_c = zeros(C,1);
        n_c = zeros(C,1);

        for c = 1:C
            idx = (y == classes(c));
            mu_c(c) = mean(X(idx,d));
            var_c(c) = var(X(idx,d));
            n_c(c) = sum(idx);
        end

        var_between = sum(n_c .* (mu_c - mu_total(d)).^2) / N;
        var_within  = sum(n_c .* var_c) / N;

        if var_within > 1e-10
            fisher_scores(d) = var_between / var_within;
        else
            fisher_scores(d) = 0;
        end
    end
end
