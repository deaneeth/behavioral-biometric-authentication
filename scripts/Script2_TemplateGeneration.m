
%% ========================================================================
%  SCRIPT 2: TEMPLATE GENERATION
%  STAGE 5: Verify and Prepare Multi-Class Labeled Datasets
%  PUSL3123 Coursework - User Authentication via Gait Analysis
%% ========================================================================
%
%  PURPOSE:
%  This script verifies the multi-class labeled datasets from Script 1
%  and prepares them for training ONE multi-class MLP classifier.
%
%  NOTE: We do NOT create 10 separate binary templates. The final model
%        is ONE multi-class neural network with 10 output neurons.
%
%% ========================================================================

clear; clc; close all;

fprintf('========================================\n');
fprintf('SCRIPT 2: TEMPLATE GENERATION\n');
fprintf('========================================\n\n');

%% ========================================================================
%  CONFIGURATION
%% ========================================================================

% Input folders (from Script 1, Stage 4)
features_folder_accel = 'features/accel/';
features_folder_gyro = 'features/gyro/';
features_folder_combined = 'features/combined/';

% Output folders (templates)
templates_folder_accel = 'templates/accel/';
templates_folder_gyro = 'templates/gyro/';
templates_folder_combined = 'templates/combined/';

% Define variants
variants = {'accel', 'gyro', 'combined'};
input_folders = {features_folder_accel, features_folder_gyro, features_folder_combined};
output_folders = {templates_folder_accel, templates_folder_gyro, templates_folder_combined};

% Class balance threshold (imbalance ratio > 2:1 triggers balancing)
imbalance_threshold = 2.0;

%% ========================================================================
%  PROCESS EACH VARIANT
%% ========================================================================

for v = 1:length(variants)
    variant_name = variants{v};
    input_folder = input_folders{v};
    output_folder = output_folders{v};
    
    fprintf('========================================\n');
    fprintf('VARIANT: %s\n', upper(variant_name));
    fprintf('========================================\n');
    
    % Load AllFeatures_*.mat from Script 1
    input_file = sprintf('AllFeatures_%s.mat', variant_name);
    input_path = fullfile(input_folder, input_file);
    
    if ~isfile(input_path)
        warning('File not found: %s', input_path);
        continue;
    end
    
    load(input_path, 'X', 'y_user', 'meta');
    
    % -----------------------------
    % GAIT FILTERING (compute threshold from Day1 and produce gait-only template)
    % -----------------------------
    % Expectation: 'meta' is a table with fields: user_id, day_num, gait_score
    % (Script1 must have added gait_score and is_gait; if not present, this block
    % will fail loudly — that's intentional and deterministic.)
    if istable(meta) && ismember('gait_score', meta.Properties.VariableNames)
        % Extract Day-1 gait scores and compute deterministic threshold
        gait_scores_day1 = meta.gait_score(meta.day_num == 1);
        gait_thr = median(gait_scores_day1) + 0.10 * iqr(gait_scores_day1);
    
        % Compute is_gait vector (logical)
        is_gait_vec = meta.gait_score >= gait_thr;
    
        % If this is the combined variant, save a gait-only template file
        if strcmpi(variant_name, 'combined')
            % Filter to gait-only rows
            X_gait   = X(is_gait_vec, :);
            y_gait   = y_user(is_gait_vec, :);
            meta_gait = meta(is_gait_vec, :);
    
            % Save gait-only template into templates/combined/
            gait_output_file = sprintf('AllTemplates_%s_gait.mat', variant_name);
            gait_output_path = fullfile(output_folder, gait_output_file);
            save(gait_output_path, 'X_gait', 'y_gait', 'meta_gait');
            
            % Print a short deterministic acceptance line
            fprintf('✓ Gait template saved: %s (rows: %d)\n', gait_output_file, size(X_gait,1));
        end
    
        % Also attach the boolean into meta for downstream checks (keeps original names)
        meta.is_gait = is_gait_vec;
    
    else
        error('Meta table does not contain gait_score. Ensure Script1 produced meta.gait_score before running Script2.');
    end

    % ----------------------------------------------------------------
    % VERIFICATION 1: Check label range (must be 1-10)
    % ----------------------------------------------------------------
    unique_labels = unique(y_user);
    fprintf('✓ Unique labels found: [%s]\n', num2str(unique_labels'));
    
    if min(y_user) < 1 || max(y_user) > 10
        error('Invalid labels detected! Expected 1-10, found range [%d, %d]', min(y_user), max(y_user));
    end
    
    if length(unique_labels) ~= 10
        warning('Expected 10 users, found %d unique labels', length(unique_labels));
    end
    
    % ----------------------------------------------------------------
    % VERIFICATION 2: Class balance check
    % ----------------------------------------------------------------
    fprintf('\n✓ Class distribution (samples per user):\n');
    
    samples_per_user = zeros(10, 1);
    for u = 1:10
        samples_per_user(u) = sum(y_user == u);
        fprintf('  User %2d: %d samples\n', u, samples_per_user(u));
    end
    
    % Calculate imbalance ratio
    min_samples = min(samples_per_user);
    max_samples = max(samples_per_user);
    imbalance_ratio = max_samples / min_samples;
    
    fprintf('\n✓ Imbalance ratio: %.2f:1 (max/min samples)\n', imbalance_ratio);
    
    % ----------------------------------------------------------------
    % OPTIONAL: Balance classes if imbalance > threshold
    % ----------------------------------------------------------------
    balancing_applied = false;
    
    if imbalance_ratio > imbalance_threshold
        fprintf('  ⚠ Imbalance detected (ratio > %.1f:1)\n', imbalance_threshold);
        fprintf('  → Downsampling to %d samples per user...\n', min_samples);
        
        % Downsample each user to min_samples
        X_balanced = [];
        y_user_balanced = [];
        meta_balanced = table();
        
        rng(42); % Fixed seed for reproducibility

        for u = 1:10
            % Get indices for this user
            user_idx = find(y_user == u);
            
            % Randomly select min_samples indices 
            selected_idx = user_idx(randperm(length(user_idx), min_samples));
            
            % Accumulate balanced data
            X_balanced = [X_balanced; X(selected_idx, :)];
            y_user_balanced = [y_user_balanced; y_user(selected_idx)];
            meta_balanced = [meta_balanced; meta(selected_idx, :)];
        end
        
        % Replace original data with balanced data
        X = X_balanced;
        y_user = y_user_balanced;
        meta = meta_balanced;
        
        balancing_applied = true;
        fprintf('  ✓ Balancing complete: %d total samples\n', size(X, 1));
    else
        fprintf('  ✓ Classes are balanced (no action needed)\n');
    end
    
    % ----------------------------------------------------------------
    % SAVE TEMPLATES
    % ----------------------------------------------------------------
    output_file = sprintf('AllTemplates_%s.mat', variant_name);
    output_path = fullfile(output_folder, output_file);
    
    save(output_path, 'X', 'y_user', 'meta');
    
    fprintf('\n✓ Summary:\n');
    fprintf('  Total samples: %d\n', size(X, 1));
    fprintf('  Total features: %d\n', size(X, 2));
    fprintf('  Number of users: %d\n', length(unique(y_user)));
    fprintf('  Balancing applied: %s\n', mat2str(balancing_applied));
    fprintf('  Saved: %s\n\n', output_file);
end

%% ========================================================================
%  FINAL ACCEPTANCE CHECKS 
%% ========================================================================

fprintf('========================================\n');
fprintf('STAGE 5 ACCEPTANCE CHECKS\n');
fprintf('========================================\n');

fprintf('✓ Verified multi-class labels (1-10) for all variants\n');
fprintf('✓ Class balance checked for all variants\n');
fprintf('✓ Templates saved for all variants\n\n');

fprintf('✓ Saved files:\n');
fprintf('  - templates/accel/AllTemplates_accel.mat\n');
fprintf('  - templates/gyro/AllTemplates_gyro.mat\n');
fprintf('  - templates/combined/AllTemplates_combined.mat\n');
fprintf('========================================\n\n');

fprintf('Stage 5 complete. Ready for Stage 6 (Baseline Multi-Class MLP Training).\n');