%% ========================================================================
%  PLOTTING SCRIPT (FINAL)
%  Generates all required figures using saved results only.
%  No recomputation. Minimal & deterministic.
%% ========================================================================

clear; clc; close all;

%% ========================================================================
%  1. Cross-Day FAR/FRR/EER Curve (0% overlap)
%% ========================================================================

load('results/combined/scenario2_crossday_plot_data.mat');

figure('Position', [100 100 800 600]);
plot(plot_data.thresholds, plot_data.far_curve, 'r','LineWidth',2); hold on;
plot(plot_data.thresholds, plot_data.frr_curve, 'b','LineWidth',2);
plot(plot_data.thresholds(plot_data.eer_idx), plot_data.eer_value, ...
     'o','MarkerSize',10,'MarkerFaceColor',[1 0.8 0],'Color',[1 0.8 0]);
xlabel('Threshold'); ylabel('Error Rate');
title('FAR/FRR Curve (Cross-Day)');
legend('FAR','FRR','EER','Location','best');
grid on;

saveas(gcf, 'results/plots/EER_Curve_CrossDay.png');

%% ========================================================================
%  2. Cross-Day EER: 0% vs 50% Overlap
%% ========================================================================

t0 = readtable('results/combined/scenario2_crossday_metrics.csv');
t50 = readtable('results_50overlap/combined/scenario2_crossday_metrics.csv');

eer_vals = [t0.EER; t50.EER];

figure('Position', [100 100 800 600]);
bar(eer_vals,'LineWidth',1.5);
set(gca,'XTickLabel',{'0%','50%'});
ylabel('EER');
title('Cross-Day EER: 0% vs 50% Overlap');
grid on;

text(1,eer_vals(1)*1.05,sprintf('%.2f%%',eer_vals(1)*100),'HorizontalAlignment','center');
text(2,eer_vals(2)*1.05,sprintf('%.2f%%',eer_vals(2)*100),'HorizontalAlignment','center');

saveas(gcf,'results/plots/CrossDay_EER_0vs50.png');

%% ========================================================================
%  3. Variant Comparison (Accel / Gyro / Combined)
%% ========================================================================

T = readtable('results/variants_comparison.csv');

figure('Position',[100 100 800 600]);
bar(T.EER,'LineWidth',1.5);
set(gca,'XTickLabel',T.Variant);
ylabel('EER');
title('Variant Comparison (Cross-Day)');
grid on;

for i=1:height(T)
    text(i,T.EER(i)*1.05,sprintf('%.2f%%',T.EER(i)*100), ...
        'HorizontalAlignment','center');
end

saveas(gcf,'results/plots/Variant_Comparison.png');

%% ========================================================================
%  4. Feature Count vs EER (Top-N Feature Selection)
%% ========================================================================

S = readtable('results/optimization/summary.csv');
Sf = S(strcmp(S.Lever,'PerUser-RankAvg'),:);
Sf = sortrows(Sf,'Features_Used');

figure('Position',[100 100 800 600]);
plot(Sf.Features_Used, Sf.EER,'-o','LineWidth',2,'MarkerSize',7);
xlabel('Number of Features'); ylabel('EER');
title('Feature Count vs EER');
grid on;

[min_eer,idx] = min(Sf.EER);
hold on;
plot(Sf.Features_Used(idx),min_eer,'ro','MarkerSize',10,'MarkerFaceColor','r');
text(Sf.Features_Used(idx),min_eer*0.97, ...
     sprintf('Best: %d\n%.2f%%',Sf.Features_Used(idx),min_eer*100), ...
     'HorizontalAlignment','center');

saveas(gcf,'results/plots/Feature_Count_vs_EER.png');

%% ========================================================================
%  5. Architecture Comparison (Cross-Day)
%% ========================================================================

Sa = S(strcmp(S.Lever,'Architecture'),:);

figure('Position',[100 100 900 600]);
b = bar(Sa.EER,'LineWidth',1.5);
set(gca,'XTickLabel',Sa.Setting);
ylabel('EER');
title('MLP Architecture Comparison');
grid on;

for i=1:height(Sa)
    text(i,Sa.EER(i)*1.05,sprintf('%.2f%%',Sa.EER(i)*100), ...
        'HorizontalAlignment','center');
end

% highlight best
[~,best_idx] = min(Sa.EER);
b.FaceColor = 'flat';
b.CData(best_idx,:) = [0.85 0.2 0.2];

saveas(gcf,'results/plots/MLP_Architecture_Comparison.png');

%% ========================================================================
%  6. Scenario Comparison (Same-Day / Cross-Day / Combined)
%% ========================================================================

s1 = readtable('results/combined/scenario1_sameday_metrics.csv');
s2 = readtable('results/combined/scenario2_crossday_metrics.csv');
s3 = readtable('results/combined/scenario3_combined_metrics.csv');

eer_vals = [s1.EER; s2.EER; s3.EER];

figure('Position',[100 100 800 600]);
bar(eer_vals,'LineWidth',1.5);
set(gca,'XTickLabel',{'Same-Day','Cross-Day','Combined'});
ylabel('EER');
title('Scenario Comparison');
grid on;

for i=1:3
    text(i,eer_vals(i)*1.05,sprintf('%.2f%%',eer_vals(i)*100), ...
        'HorizontalAlignment','center');
end

saveas(gcf,'results/plots/Scenario_Comparison.png');

%% ========================================================================
%  7. Sample Window Visualization (Accel XYZ)
%% ========================================================================

load('data_segmented/User1_Day1_segments.mat');
sample = squeeze(segments_accel(1,:,:));

figure('Position',[100 100 900 600]);
plot(sample,'LineWidth',1.5);
xlabel('Samples (150 @ 30Hz)'); ylabel('Acceleration');
title('Sample Acceleration Window (User1 Day1)');
legend('AX','AY','AZ');
grid on;

saveas(gcf,'results/plots/SampleWindow_Accel.png');

%% ========================================================================
%  8. Gait vs Non-Gait EER Comparison
%% ========================================================================

ng = readtable('results/combined/scenario2_crossday_metrics.csv');
g  = readtable('results/combined/scenario2b_crossday_gait_metrics.csv');

eer_vals = [ng.EER; g.EER];

figure('Position',[100 100 800 600]);
bar(eer_vals,'LineWidth',1.5);
set(gca,'XTickLabel',{'Non-Gait','Gait'});
ylabel('EER');
title('Cross-Day: Gait vs Non-Gait EER');
grid on;

text(1,eer_vals(1)*1.05,sprintf('%.2f%%',eer_vals(1)*100),'HorizontalAlignment','center');
text(2,eer_vals(2)*1.05,sprintf('%.2f%%',eer_vals(2)*100),'HorizontalAlignment','center');

saveas(gcf,'results/plots/Gait_vs_NonGait_EER.png');

