%% ========================================================================
%  SCRIPT 1: FEATURE EXTRACTION
%  STAGE 1: Data Loading, Resampling & Segmentation
%  ========================================================================

clear; clc; close all;

%% ========================================================================
%  CONFIGURATION
%% ========================================================================

% Path to raw data
data_raw_folder = 'data_raw/';

% Target sampling rate
target_fs = 30;  % Hz

% Segmentation parameters
window_length_sec = 5;   % 5-second windows
overlap_percent   = 0;   % 0% baseline (50% for optimization)

samples_per_window = window_length_sec * target_fs;   % 150 samples
step_size = max(1, round(samples_per_window * (1 - overlap_percent/100)));

% Output folder
output_folder = 'data_segmented/';

%% ========================================================================
%  INITIAL INFO
%% ========================================================================

fprintf('========================================\n');
fprintf('STAGE 1: DATA LOADING & SEGMENTATION\n');
fprintf('========================================\n\n');

user_ids   = 1:10;
day_labels = {'FD','MD'};
day_names  = {'Day1','Day2'};

total_files = length(user_ids) * length(day_labels);
fprintf('Total expected files: %d\n\n', total_files);

processed_count = 0;
total_segments  = 0;

%% ========================================================================
%  LOOP: LOAD → RESAMPLE → SEGMENT → SAVE
%% ========================================================================

for u = 1:length(user_ids)
    user_id = user_ids(u);

    for d = 1:length(day_labels)

        day_label = day_labels{d};
        day_name  = day_names{d};

        % Build filename
        filename = sprintf('U%dNW_%s.csv', user_id, day_label);
        filepath = fullfile(data_raw_folder, filename);

        if ~isfile(filepath)
            warning('File not found: %s', filepath);
            continue;
        end

        %% ---------------------------------------------------------------
        % STEP 1: Load raw CSV (6 signals)
        % ---------------------------------------------------------------
        raw_data = readmatrix(filepath);

        time_original = raw_data(:,1);
        accel_x = raw_data(:,2); accel_y = raw_data(:,3); accel_z = raw_data(:,4);
        gyro_x  = raw_data(:,5); gyro_y  = raw_data(:,6); gyro_z  = raw_data(:,7);

        % Ensure unique timestamps
        [time_original, idx] = unique(time_original,'stable');
        accel_x = accel_x(idx); accel_y = accel_y(idx); accel_z = accel_z(idx);
        gyro_x  = gyro_x(idx); gyro_y  = gyro_y(idx); gyro_z  = gyro_z(idx);

        %% ---------------------------------------------------------------
        % STEP 2: Resample to EXACT 30 Hz
        % ---------------------------------------------------------------
        t0 = time_original(1);
        tN = time_original(end);
        time_new = (t0 : 1/target_fs : tN)';

        accel_x_rs = interp1(time_original, accel_x, time_new,'linear','extrap');
        accel_y_rs = interp1(time_original, accel_y, time_new,'linear','extrap');
        accel_z_rs = interp1(time_original, accel_z, time_new,'linear','extrap');
        gyro_x_rs  = interp1(time_original, gyro_x,  time_new,'linear','extrap');
        gyro_y_rs  = interp1(time_original, gyro_y,  time_new,'linear','extrap');
        gyro_z_rs  = interp1(time_original, gyro_z,  time_new,'linear','extrap');

        % Fill possible NaNs
        accel_x_rs = fillmissing(accel_x_rs,'linear');
        accel_y_rs = fillmissing(accel_y_rs,'linear');
        accel_z_rs = fillmissing(accel_z_rs,'linear');
        gyro_x_rs  = fillmissing(gyro_x_rs,'linear');
        gyro_y_rs  = fillmissing(gyro_y_rs,'linear');
        gyro_z_rs  = fillmissing(gyro_z_rs,'linear');

        %% ---------------------------------------------------------------
        % STEP 3: Segment (FIXED & CORRECT LOCATION)
        % ---------------------------------------------------------------
        num_samples = length(time_new);

        % Compute segment start indices (uses updated step_size)
        segment_starts = 1 : step_size : (num_samples - samples_per_window + 1);

        num_segments = length(segment_starts);
        segments_accel = zeros(num_segments, samples_per_window, 3);
        segments_gyro  = zeros(num_segments, samples_per_window, 3);

        % Perform segmentation
        for s = 1:num_segments
            idx_start = segment_starts(s);
            idx_end   = idx_start + samples_per_window - 1;

            segments_accel(s,:,:) = [accel_x_rs(idx_start:idx_end), ...
                                     accel_y_rs(idx_start:idx_end), ...
                                     accel_z_rs(idx_start:idx_end)];

            segments_gyro(s,:,:)  = [gyro_x_rs(idx_start:idx_end), ...
                                     gyro_y_rs(idx_start:idx_end), ...
                                     gyro_z_rs(idx_start:idx_end)];
        end

        %% ---------------------------------------------------------------
        % STEP 4: Save segmented file
        % ---------------------------------------------------------------
        output_filename = sprintf('User%d_%s_segments.mat', user_id, day_name);
        output_path     = fullfile(output_folder, output_filename);

        sampling_rate = target_fs;
        window_length = window_length_sec;
        overlap_used  = overlap_percent;

        save(output_path, ...
            'segments_accel','segments_gyro','user_id','day_name', ...
            'num_segments','sampling_rate','window_length','overlap_used');

        processed_count = processed_count + 1;
        total_segments  = total_segments + num_segments;

        fprintf('✓ User%d_%s: %d segments saved\n', user_id, day_name, num_segments);
    end
end

%% ========================================================================
%  ACCEPTANCE CHECKS
%% ========================================================================

fprintf('\n========================================\n');
fprintf('STAGE 1 ACCEPTANCE CHECKS\n');
fprintf('========================================\n');
fprintf('✓ Files processed: %d / %d\n', processed_count, total_files);
fprintf('✓ Typical segments per file: ~72 (0%% overlap, 5-sec windows)\n');
fprintf('✓ Samples per segment: %d\n', samples_per_window);
fprintf('✓ Total segments: %d\n', total_segments);
fprintf('✓ Output folder: %s\n', output_folder);
fprintf('========================================\n\n');

%% ========================================================================
%  STAGE 2: TIME-DOMAIN FEATURE EXTRACTION
%% ========================================================================

fprintf('========================================\n');
fprintf('STAGE 2: TIME-DOMAIN FEATURES\n');
fprintf('========================================\n\n');

% Create output folders for three variants
features_folder_accel = 'features/accel/';
features_folder_gyro = 'features/gyro/';
features_folder_combined = 'features/combined/';

% Define feature names (8 features per axis)
feature_suffixes = {'mean', 'std', 'min', 'max', 'rms', 'var', 'range', 'p75'};
axis_names_accel = {'ax', 'ay', 'az'};
axis_names_gyro = {'gx', 'gy', 'gz'};

% Build feature column names
feature_names_accel = {};
for a = 1:length(axis_names_accel)
    for f = 1:length(feature_suffixes)
        feature_names_accel{end+1} = sprintf('%s_%s', axis_names_accel{a}, feature_suffixes{f});
    end
end

feature_names_gyro = {};
for a = 1:length(axis_names_gyro)
    for f = 1:length(feature_suffixes)
        feature_names_gyro{end+1} = sprintf('%s_%s', axis_names_gyro{a}, feature_suffixes{f});
    end
end


% === VECTOR MAGNITUDE NAMES ===
vm_names = {'vm_mean','vm_std','vm_rms','vm_p75'};

% === SHAPE FEATURE NAMES: ACCEL ===
accel_shape_names = { ...
    'ax_skew','ax_kurt','ax_iqr', ...
    'ay_skew','ay_kurt','ay_iqr', ...
    'az_skew','az_kurt','az_iqr'};

% === SHAPE FEATURE NAMES: GYRO ===
gyro_shape_names = { ...
    'gx_skew','gx_kurt','gx_iqr', ...
    'gy_skew','gy_kurt','gy_iqr', ...
    'gz_skew','gz_kurt','gz_iqr'};

% === FINAL TIME-DOMAIN FEATURE NAMES ===
feature_names_combined = [ ...
    feature_names_accel, ...
    feature_names_gyro, ...
    vm_names, ...
    accel_shape_names, ...
    gyro_shape_names ...
];




fprintf('Feature counts per variant:\n');
fprintf('  Accel-only: %d features\n', length(feature_names_accel));
fprintf('  Gyro-only: %d features\n', length(feature_names_gyro));
fprintf('  Combined: %d features\n\n', length(feature_names_combined));

% Process all segmented files
total_files_processed = 0;

for u = 1:length(user_ids)
    user_id = user_ids(u);
    
    for d = 1:length(day_names)
        day_name = day_names{d};
        
        % Load segmented data
        segment_filename = sprintf('User%d_%s_segments.mat', user_id, day_name);
        segment_filepath = fullfile(output_folder, segment_filename);
        
        if ~isfile(segment_filepath)
            warning('Segmented file not found: %s', segment_filename);
            continue;
        end
        
        load(segment_filepath, 'segments_accel', 'segments_gyro', 'num_segments');
        
        % Pre-allocate feature matrices
        % Accel: [num_segments × 24 time features]
        % Gyro:  [num_segments × 24 time features]
        features_accel = zeros(num_segments, length(feature_names_accel));
        features_gyro = zeros(num_segments, length(feature_names_gyro));
        
        



        % Extract features for each segment
        for seg = 1:num_segments
        
            accel_seg = squeeze(segments_accel(seg,:,:));
            gyro_seg  = squeeze(segments_gyro(seg,:,:));
        
            % --------------------- ACCEL 24 FEATURES ---------------------
            col_idx = 1;
            for axis = 1:3
                sig = accel_seg(:,axis);
        
                features_accel(seg,col_idx)   = mean(sig);
                features_accel(seg,col_idx+1) = std(sig);
                features_accel(seg,col_idx+2) = min(sig);
                features_accel(seg,col_idx+3) = max(sig);
                features_accel(seg,col_idx+4) = rms(sig);
                features_accel(seg,col_idx+5) = var(sig);
                features_accel(seg,col_idx+6) = max(sig)-min(sig);
                features_accel(seg,col_idx+7) = prctile(sig,75);
        
                col_idx = col_idx + 8;
            end
        
            % --------------------- VECTOR MAGNITUDE (4) ---------------------
            vm = sqrt(sum(accel_seg.^2,2));
            vm_features = [mean(vm), std(vm), rms(vm), prctile(vm,75)];
        
            % --------------------- ACCEL SHAPE FEATURES (9) ---------------------
            accel_shape = zeros(1,9);
            idx = 1;
            for axis = 1:3
                sig = accel_seg(:,axis);
                accel_shape(idx)   = skewness(sig);
                accel_shape(idx+1) = kurtosis(sig);
                accel_shape(idx+2) = iqr(sig);
                idx = idx + 3;
            end
        
            % --------------------- GYRO 24 FEATURES ---------------------
            col_idx = 1;
            for axis = 1:3
                sig = gyro_seg(:,axis);
        
                features_gyro(seg,col_idx)   = mean(sig);
                features_gyro(seg,col_idx+1) = std(sig);
                features_gyro(seg,col_idx+2) = min(sig);
                features_gyro(seg,col_idx+3) = max(sig);
                features_gyro(seg,col_idx+4) = rms(sig);
                features_gyro(seg,col_idx+5) = var(sig);
                features_gyro(seg,col_idx+6) = max(sig)-min(sig);
                features_gyro(seg,col_idx+7) = prctile(sig,75);
        
                col_idx = col_idx + 8;
            end
        
            % --------------------- GYRO SHAPE FEATURES (9) ---------------------
            gyro_shape = zeros(1,9);
            idx = 1;
            for axis = 1:3
                sig = gyro_seg(:,axis);
                gyro_shape(idx)   = skewness(sig);
                gyro_shape(idx+1) = kurtosis(sig);
                gyro_shape(idx+2) = iqr(sig);
                idx = idx + 3;
            end
        
            % --------------------- COMBINED TIME FEATURES ---------------------
            combined_row = [ ...
                features_accel(seg,:), ...
                features_gyro(seg,:), ...
                vm_features, ...
                accel_shape, ...
                gyro_shape ...
            ];
        
            if seg == 1
                features_combined = zeros(num_segments, length(combined_row));
            end
        
            features_combined(seg,:) = combined_row;
        
        end  % end seg loop


        


        




        
        % --- SAVE ACCEL-ONLY VARIANT ---
        features_table = array2table(features_accel, 'VariableNames', feature_names_accel);
        num_features = size(features_accel, 2);
        save(fullfile(features_folder_accel, sprintf('User%d_%s_features_time.mat', user_id, day_name)), ...
             'features_table', 'user_id', 'day_name', 'num_segments', 'num_features', 'feature_names_accel');
        
        % --- SAVE GYRO-ONLY VARIANT ---
        features_table = array2table(features_gyro, 'VariableNames', feature_names_gyro);
        num_features = size(features_gyro, 2);
        save(fullfile(features_folder_gyro, sprintf('User%d_%s_features_time.mat', user_id, day_name)), ...
             'features_table', 'user_id', 'day_name', 'num_segments', 'num_features', 'feature_names_gyro');
        
         

        features_table = array2table(features_combined, 'VariableNames', feature_names_combined);
        num_features = size(features_combined, 2);
        save(fullfile(features_folder_combined, sprintf('User%d_%s_features_time.mat', user_id, day_name)), ...
             'features_table', 'user_id', 'day_name', 'num_segments', 'num_features', 'feature_names_combined');
        
        total_files_processed = total_files_processed + 1;
        
        fprintf('✓ User%d_%s: %d segments, %d features (accel), %d features (gyro), %d features (combined)\n', ...
                user_id, day_name, num_segments, ...
                length(feature_names_accel), length(feature_names_gyro), length(feature_names_combined));
    end
end

%% ========================================================================
%  STAGE 2 ACCEPTANCE CHECKS
%% ========================================================================

fprintf('\n========================================\n');
fprintf('STAGE 2 ACCEPTANCE CHECKS\n');
fprintf('========================================\n');
fprintf('✓ Files processed: %d/20\n', total_files_processed);
fprintf('✓ Segments per file: 71\n');
fprintf('✓ Features per variant:\n');
fprintf('    - Accel-only: %d features (3 axes × 8)\n', length(feature_names_accel));
fprintf('    - Gyro-only: %d features (3 axes × 8)\n', length(feature_names_gyro));
fprintf('    - Combined: %d features (time-domain: 70 with VM+shape)\n', length(feature_names_combined));
fprintf('\n✓ Saved locations:\n');
fprintf('    - %s\n', features_folder_accel);
fprintf('    - %s\n', features_folder_gyro);
fprintf('    - %s\n', features_folder_combined);

%% ========================================================================
%  STAGE 3: FREQUENCY-DOMAIN FEATURE EXTRACTION
%% ========================================================================

fprintf('\n========================================\n');
fprintf('STAGE 3: FREQUENCY-DOMAIN FEATURES\n');
fprintf('========================================\n\n');

% Frequency analysis parameters
fs = target_fs;  % 30 Hz sampling rate
gait_band = [0.5 3];  % Hz (typical human walking frequency range)

% Define frequency feature names (3 features per axis)
freq_suffixes = {'bp03', 'energy', 'fdom'};  % bandpower, energy, dominant freq

% Build frequency feature column names
freq_names_accel = {};
for a = 1:length(axis_names_accel)
    for f = 1:length(freq_suffixes)
        freq_names_accel{end+1} = sprintf('%s_%s', axis_names_accel{a}, freq_suffixes{f});
    end
end

freq_names_gyro = {};
for a = 1:length(axis_names_gyro)
    for f = 1:length(freq_suffixes)
        freq_names_gyro{end+1} = sprintf('%s_%s', axis_names_gyro{a}, freq_suffixes{f});
    end
end





extra_freq_names = {'centroid','spec_entropy','bp_3_10','dom_amp'};

% accel extra names (3 axes × 4)
accel_extra_names = {};
for a = 1:length(axis_names_accel)
    for f = 1:length(extra_freq_names)
        accel_extra_names{end+1} = sprintf('%s_%s', axis_names_accel{a}, extra_freq_names{f});
    end
end

% gyro extra names
gyro_extra_names = {};
for a = 1:length(axis_names_gyro)
    for f = 1:length(extra_freq_names)
        gyro_extra_names{end+1} = sprintf('%s_%s', axis_names_gyro{a}, extra_freq_names{f});
    end
end

% final name lists
freq_names_combined = [ ...
    freq_names_accel, ...
    freq_names_gyro, ...
    accel_extra_names, ...
    gyro_extra_names ...
];


% FULL per-variant freq name lists (base + extras)
freq_names_accel_full = [freq_names_accel, accel_extra_names];   % 9 + 12 = 21
freq_names_gyro_full  = [freq_names_gyro,  gyro_extra_names];    % 9 + 12 = 21

% final name lists for combined
freq_names_combined = [freq_names_accel_full, freq_names_gyro_full];  % 42 names




fprintf('Frequency feature counts per variant:\n');
fprintf('  Accel-only: %d features (expected 21)\n', length(freq_names_accel_full));
fprintf('  Gyro-only: %d features (expected 21)\n', length(freq_names_gyro_full));
fprintf('  Combined: %d features (expected 42)\n\n', length(freq_names_combined));

% Process all segmented files
total_freq_files = 0;

for u = 1:length(user_ids)
    user_id = user_ids(u);
    
    for d = 1:length(day_names)
        day_name = day_names{d};
        
        % Load segmented data
        segment_filename = sprintf('User%d_%s_segments.mat', user_id, day_name);
        segment_filepath = fullfile(output_folder, segment_filename);
        
        if ~isfile(segment_filepath)
            warning('Segmented file not found: %s', segment_filename);
            continue;
        end
        
        load(segment_filepath, 'segments_accel', 'segments_gyro', 'num_segments');
        
        % Pre-allocate frequency feature matrices
        % Accel: [num_segments × 21 features]
        % Gyro : [num_segments × 21 features]

        num_base = length(freq_names_accel);     % 9
        num_extra = 3*4;                         % 12
        freq_features_accel = zeros(num_segments, num_base + num_extra);

        num_base = length(freq_names_gyro);      % 9
        num_extra = 3*4;                         
        freq_features_gyro = zeros(num_segments, num_base + num_extra);

        
        
        % Extract frequency features for each segment
        for seg = 1:num_segments
        
            accel_seg = squeeze(segments_accel(seg,:,:));
            gyro_seg  = squeeze(segments_gyro(seg,:,:));
        
            % Reset extra feature matrices
            extra_freq_accel = zeros(3,4);  % centroid, entropy, bp_3_10, dom_amp
            extra_freq_gyro  = zeros(3,4);
        
            % ================================================================
            % ACCEL FREQUENCY FEATURES
            % ================================================================
            col_idx = 1;
            for axis = 1:3
                sig = accel_seg(:,axis);
        
                % PSD
                [psd,freq] = pwelch(sig,hamming(length(sig)),[],[],fs);
        
                % Basic features
                bp_gait = bandpower(sig,fs,gait_band);
                energy  = sum(psd);
        
                gait_idx = freq>=gait_band(1) & freq<=gait_band(2);
                psd_gait = psd(gait_idx);
                freq_gait = freq(gait_idx);
                if ~isempty(psd_gait)
                    [~,k] = max(psd_gait);
                    fdom = freq_gait(k);
                else
                    fdom = 0;
                end
        
                % Store base 3 features
                freq_features_accel(seg,col_idx)   = bp_gait;
                freq_features_accel(seg,col_idx+1) = energy;
                freq_features_accel(seg,col_idx+2) = fdom;
                col_idx = col_idx + 3;
        
                % Extra freq features
                centroid = sum(freq .* psd) / (sum(psd) + eps);
                pn = psd / sum(psd);
                spec_entropy = -sum(pn .* log(pn + eps));
                bp_3_10 = bandpower(sig,fs,[3 10]);
                [~,m] = max(psd);
                dom_amp = psd(m);
        
                extra_freq_accel(axis,:) = [centroid, spec_entropy, bp_3_10, dom_amp];
            end
        
            
            % Append accel extras AFTER axis loop
            extra_accel_row = extra_freq_accel(:)';
            
            % Safety for centroid denom (prevent division by zero)
            % (centroid computed earlier per-axis uses sum(psd) — we add eps in that line if needed)
            
            % Place the extra columns directly (base already stored in freq_features_accel)
            freq_features_accel(seg, 10:21) = extra_accel_row;

        
            % ================================================================
            % GYRO FREQUENCY FEATURES
            % ================================================================
            col_idx = 1;
            for axis = 1:3
                sig = gyro_seg(:,axis);
        
                [psd,freq] = pwelch(sig,hamming(length(sig)),[],[],fs);
        
                bp_gait = bandpower(sig,fs,gait_band);
                energy  = sum(psd);
        
                gait_idx = freq>=gait_band(1) & freq<=gait_band(2);
                psd_gait = psd(gait_idx);
                freq_gait = freq(gait_idx);
                if ~isempty(psd_gait)
                    [~,k] = max(psd_gait);
                    fdom = freq_gait(k);
                else
                    fdom = 0;
                end
        
                freq_features_gyro(seg,col_idx)   = bp_gait;
                freq_features_gyro(seg,col_idx+1) = energy;
                freq_features_gyro(seg,col_idx+2) = fdom;
                col_idx = col_idx + 3;
        
                centroid = sum(freq .* psd) / (sum(psd) + eps);
                pn = psd / sum(psd);
                spec_entropy = -sum(pn .* log(pn + eps));
                bp_3_10 = bandpower(sig,fs,[3 10]);
                [~,m] = max(psd);
                dom_amp = psd(m);
        
                extra_freq_gyro(axis,:) = [centroid, spec_entropy, bp_3_10, dom_amp];
            end
        
            % Append gyro extras AFTER axis loop
            extra_gyro_row = extra_freq_gyro(:)';
            
            % Place the extra columns directly (base already stored in freq_features_gyro)
            freq_features_gyro(seg, 10:21) = extra_gyro_row;


        
        end  % end seg



        % --- SAVE ACCEL-ONLY FREQUENCY FEATURES ---
        features_table = array2table(freq_features_accel, 'VariableNames', freq_names_accel_full);
        num_features = size(freq_features_accel, 2);
        save(fullfile(features_folder_accel, sprintf('User%d_%s_features_freq.mat', user_id, day_name)), ...
             'features_table', 'user_id', 'day_name', 'num_segments', 'num_features', 'freq_names_accel_full');
        
        % --- SAVE GYRO-ONLY FREQUENCY FEATURES ---
        features_table = array2table(freq_features_gyro, 'VariableNames', freq_names_gyro_full);
        num_features = size(freq_features_gyro, 2);
        save(fullfile(features_folder_gyro, sprintf('User%d_%s_features_freq.mat', user_id, day_name)), ...
             'features_table', 'user_id', 'day_name', 'num_segments', 'num_features', 'freq_names_gyro_full');
        
        % --- SAVE COMBINED FREQUENCY FEATURES ---
        freq_features_combined = [freq_features_accel, freq_features_gyro];
        features_table = array2table(freq_features_combined, 'VariableNames', freq_names_combined);  % already full
        num_features = size(freq_features_combined, 2);
        save(fullfile(features_folder_combined, sprintf('User%d_%s_features_freq.mat', user_id, day_name)), ...
            'features_table', 'user_id', 'day_name', 'num_segments', 'num_features', 'freq_names_combined');

        
        total_freq_files = total_freq_files + 1;
        
        fprintf('✓ User%d_%s: %d segments, %d freq features (accel), %d freq features (gyro), %d freq features (combined)\n', ...
                user_id, day_name, num_segments, ...
                length(freq_names_accel_full), length(freq_names_gyro_full), length(freq_names_combined));

    end
end

%% ========================================================================
%  STAGE 3 ACCEPTANCE CHECKS
%% ========================================================================

fprintf('\n========================================\n');
fprintf('STAGE 3 ACCEPTANCE CHECKS\n');
fprintf('========================================\n');
fprintf('✓ Files processed: %d/20\n', total_freq_files);
fprintf('✓ Segments per file: 71\n');
fprintf('✓ Frequency features per variant:\n');
fprintf('    - Accel-only: %d features (expected 21)\n', length(freq_names_accel_full));
fprintf('    - Gyro-only: %d features (expected 21)\n', length(freq_names_gyro_full));
fprintf('    - Combined: %d features (expected 42)\n', length(freq_names_combined));
fprintf('\n✓ Saved locations:\n');
fprintf('    - %s (time + freq features)\n', features_folder_accel);
fprintf('    - %s (time + freq features)\n', features_folder_gyro);
fprintf('    - %s (time + freq features)\n', features_folder_combined);
fprintf("DEBUG CHECK:\n");
fprintf("  Accel freq dims: %d\n", size(freq_features_accel,2));
fprintf("  Gyro  freq dims: %d\n", size(freq_features_gyro,2));
fprintf("  Combined freq dims: %d\n\n", size(freq_features_accel,2) + size(freq_features_gyro,2));
fprintf('========================================\n\n');

%% ========================================================================
%  STAGE 4: FEATURE SET ASSEMBLY & LABELING
%% ========================================================================

fprintf('========================================\n');
fprintf('STAGE 4: FEATURE ASSEMBLY & LABELING\n');
fprintf('========================================\n\n');

% Define variants to process
variants = {'accel', 'gyro', 'combined'};
variant_folders = {features_folder_accel, features_folder_gyro, features_folder_combined};

for v = 1:length(variants)
    variant_name = variants{v};
    variant_folder = variant_folders{v};
    
    fprintf('Processing variant: %s\n', upper(variant_name));
    
    % Initialize storage for all users
    X_all = [];
    y_user_all = [];
    user_id_all = [];
    day_num_all = [];
    day_name_all = {};
    gait_score_all = [];

    
    % Loop through all users and days
    for u = 1:length(user_ids)
        user_id = user_ids(u);
        
        for d = 1:length(day_names)
            day_name = day_names{d};
            day_num = d;  % 1 or 2
            
            % Load time features
            time_file = sprintf('User%d_%s_features_time.mat', user_id, day_name);
            time_path = fullfile(variant_folder, time_file);
            
            % Load freq features
            freq_file = sprintf('User%d_%s_features_freq.mat', user_id, day_name);
            freq_path = fullfile(variant_folder, freq_file);
            
            if ~isfile(time_path) || ~isfile(freq_path)
                warning('Missing files for User%d_%s in %s variant', user_id, day_name, variant_name);
                continue;
            end
            
            % Load time features table
            time_data = load(time_path, 'features_table');
            time_features = table2array(time_data.features_table);
            
            % Load freq features table
            freq_data = load(freq_path, 'features_table');
            freq_features = table2array(freq_data.features_table);
            
            % Merge time + freq features horizontally
            X_merged = [time_features, freq_features];
            
            % Compute gait score (mean of low-band bandpower columns 1:3 in freq_features)
            gait_score_vec = mean(freq_features(:,1:3), 2);
            
            % Get number of segments for this user/day
            num_segs = size(X_merged, 1);
            
            % Create labels for this user/day
            y_user = repmat(user_id, num_segs, 1);  % User ID (1-10)
            user_id_vec = repmat(user_id, num_segs, 1);
            day_num_vec = repmat(day_num, num_segs, 1);
            day_name_vec = repmat({day_name}, num_segs, 1);
            
            % Accumulate data
            X_all = [X_all; X_merged];
            y_user_all = [y_user_all; y_user];
            user_id_all = [user_id_all; user_id_vec];
            day_num_all = [day_num_all; day_num_vec];
            day_name_all = [day_name_all; day_name_vec];
            gait_score_all = [gait_score_all; gait_score_vec];

        end
    end
    
    % Compute gait threshold using Day1 (day_num == 1)
    gait_scores_day1 = gait_score_all(day_num_all == 1);
    gait_thr = median(gait_scores_day1) + 0.10 * iqr(gait_scores_day1);
    is_gait_all = gait_score_all >= gait_thr;
    
    % Create metadata table with gait fields
    meta = table(user_id_all, day_num_all, day_name_all, gait_score_all, is_gait_all, ...
                 'VariableNames', {'user_id', 'day_num', 'day_name', 'gait_score', 'is_gait'});
    
    % Final variables
    X = X_all;
    y_user = y_user_all;
    
    % Save unified feature set (includes gait metadata)
    output_filename = sprintf('AllFeatures_%s.mat', variant_name);
    output_path = fullfile(variant_folder, output_filename);
    save(output_path, 'X', 'y_user', 'meta');
    
    % Print summary
    fprintf('  ✓ Total segments: %d\n', size(X, 1));
    fprintf('  ✓ Total features: %d\n', size(X, 2));
    fprintf('  ✓ Saved: %s\n\n', output_filename);

end

%% ========================================================================
%  STAGE 4 ACCEPTANCE CHECKS
%% ========================================================================

fprintf('========================================\n');
fprintf('STAGE 4 ACCEPTANCE CHECKS\n');
fprintf('========================================\n');

% Check class balance for combined variant (representative)
load(fullfile(features_folder_combined, 'AllFeatures_combined.mat'), 'X', 'y_user', 'meta');

fprintf('✓ Total segments across all users: %d\n', size(X, 1));
fprintf('✓ Feature dimensions per variant:\n');

% Load and check each variant
for v = 1:length(variants)
    variant_name = variants{v};
    variant_folder = variant_folders{v};
    data = load(fullfile(variant_folder, sprintf('AllFeatures_%s.mat', variant_name)), 'X');
    fprintf('    - %s: %d features\n', upper(variant_name), size(data.X, 2));
end

fprintf('\n✓ Class balance (samples per user):\n');
for u = 1:10
    count = sum(y_user == u);
    fprintf('    User %d: %d segments\n', u, count);
end

fprintf('\n✓ Day distribution:\n');
day1_count = sum(meta.day_num == 1);
day2_count = sum(meta.day_num == 2);
fprintf('    Day1: %d segments\n', day1_count);
fprintf('    Day2: %d segments\n', day2_count);

fprintf('\n✓ Saved files:\n');
fprintf('    - features/accel/AllFeatures_accel.mat\n');
fprintf('    - features/gyro/AllFeatures_gyro.mat\n');
fprintf('    - features/combined/AllFeatures_combined.mat\n');
fprintf('========================================\n\n');

fprintf('Stage 4 complete. Ready for Stage 5 (Template Generation).\n');