classdef HawkerFoodCalorieApp < matlab.apps.AppBase

    % Premium Malaysian Hawker Food Recognition GUI
    % Modern UI/UX with beautiful design and smooth interactions
    
    properties (Access = public)
        UIFigure                   matlab.ui.Figure
        
        % Main Layout
        MainGrid                   matlab.ui.container.GridLayout
        
        % Header
        HeaderPanel                matlab.ui.container.Panel
        TitleLabel                 matlab.ui.control.Label
        SubtitleLabel              matlab.ui.control.Label
        
        % Left Side - Images
        ImagePanel                 matlab.ui.container.Panel
        ImageGrid                  matlab.ui.container.GridLayout
        OriginalAxes               matlab.ui.control.UIAxes
        ProcessedAxes              matlab.ui.control.UIAxes
        SegmentedAxes              matlab.ui.control.UIAxes
        OriginalLabel              matlab.ui.control.Label
        ProcessedLabel             matlab.ui.control.Label
        SegmentedLabel             matlab.ui.control.Label
        
        % Right Side - Controls & Results
        ControlPanel               matlab.ui.container.Panel
        ControlGrid                matlab.ui.container.GridLayout
        
        % Buttons & Controls
        LoadButton                 matlab.ui.control.Button
        AnalyzeButton              matlab.ui.control.Button
        ResetButton                matlab.ui.control.Button
        
        % Classifier Selection
        ClassifierDropdown         matlab.ui.control.DropDown
        ClassifierLabel            matlab.ui.control.Label
        
        % Results Display
        ResultsPanel               matlab.ui.container.Panel
        FoodNameLabel              matlab.ui.control.Label
        ConfidenceMeter            matlab.ui.control.Label
        ConfidenceBar              matlab.ui.control.Label
        PortionLabel               matlab.ui.control.Label
        
        % Nutrition Card
        NutritionPanel             matlab.ui.container.Panel
        CaloriesValueLabel         matlab.ui.control.Label
        CaloriesUnitLabel          matlab.ui.control.Label
        ProteinLabel               matlab.ui.control.Label
        CarbsLabel                 matlab.ui.control.Label
        FatLabel                   matlab.ui.control.Label
        DailyValueLabel            matlab.ui.control.Label
        
        % Status Bar
        StatusPanel                matlab.ui.container.Panel
        StatusLabel                matlab.ui.control.Label
        ProcessingTimeLabel        matlab.ui.control.Label
    end
    
    properties (Access = private)
        CurrentImage
        CurrentResults
        ProjectPath
        UseDeepLearning = false    % Toggle for DL classification
        
        % Colors - Premium Palette
        PrimaryColor = [0.0, 0.48, 1.0]        % Apple Blue (Modern/Clean)
        SecondaryColor = [0.15, 0.68, 0.38]    % Emerald Green (Success)
        AccentColor = [1.0, 0.58, 0.0]         % Amber (Warnings/Highlights)
        DangerColor = [1.0, 0.23, 0.19]        % System Red (Errors)
        DLColor = [0.35, 0.34, 0.84]           % Indigo (Deep Learning)
        BackgroundColor = [0.96, 0.97, 0.99]   % Off-White Cool Gray
        CardColor = [1, 1, 1]                  % Pure White
        TextPrimary = [0.10, 0.10, 0.10]       % Near Black
        TextSecondary = [0.50, 0.50, 0.55]     % Slate Gray
    end
    
    methods (Access = private)
        
        function updateStatus(app, message, color)
            if nargin < 3
                color = app.TextSecondary;
            end
            app.StatusLabel.Text = message;
            app.StatusLabel.FontColor = color;
            drawnow;
        end
        
        function clearResults(app)
            % Clear axes
            cla(app.OriginalAxes);
            cla(app.ProcessedAxes);
            cla(app.SegmentedAxes);
            
            % Reset axes backgrounds
            app.OriginalAxes.Color = [0.97, 0.97, 0.97];
            app.ProcessedAxes.Color = [0.97, 0.97, 0.97];
            app.SegmentedAxes.Color = [0.97, 0.97, 0.97];
            
            % Reset results
            app.FoodNameLabel.Text = 'No food detected';
            app.FoodNameLabel.FontColor = app.TextSecondary;
            app.ConfidenceMeter.Text = '-- %';
            app.ConfidenceBar.BackgroundColor = [0.9, 0.9, 0.9];
            app.PortionLabel.Text = 'Portion: --';
            
            app.CaloriesValueLabel.Text = '---';
            app.CaloriesValueLabel.FontColor = app.TextSecondary;
            app.ProteinLabel.Text = {'Protein: --'; 'Carbohydrates: --'; 'Total Fat: --'; ''; '-- Daily Value'};
            
            app.ProcessingTimeLabel.Text = '';
            
            app.CurrentImage = [];
            app.CurrentResults = [];
            
            app.AnalyzeButton.Enable = 'off';
            app.updateStatus('Ready. Load an image to begin.', app.TextSecondary);
        end
        
        function showPlaceholder(app, ax, msg)
            cla(ax);
            ax.Color = [0.97, 0.97, 0.97];
            text(ax, 0.5, 0.5, msg, ...
                 'HorizontalAlignment', 'center', ...
                 'VerticalAlignment', 'middle', ...
                 'FontSize', 11, 'Color', [0.6, 0.6, 0.6]);
            ax.XLim = [0 1];
            ax.YLim = [0 1];
        end
    end
    
    methods (Access = private)
        
        function LoadButtonPushed(app, ~, ~)
            [filename, pathname] = uigetfile(...
                {'*.jpg;*.jpeg;*.png;*.bmp', 'Image Files'; '*.*', 'All Files'}, ...
                'Select a Food Image');
            
            if isequal(filename, 0)
                return;
            end
            
            imagePath = fullfile(pathname, filename);
            
            try
                app.CurrentImage = imread(imagePath);
                
                % Display original image
                cla(app.OriginalAxes);
                imshow(app.CurrentImage, 'Parent', app.OriginalAxes);
                axis(app.OriginalAxes, 'image');
                app.OriginalAxes.XLim = [0.5, size(app.CurrentImage, 2) + 0.5];
                app.OriginalAxes.YLim = [0.5, size(app.CurrentImage, 1) + 0.5];
                
                % Clear other axes
                app.showPlaceholder(app.ProcessedAxes, 'Click Analyze');
                app.showPlaceholder(app.SegmentedAxes, 'Click Analyze');
                
                % Update UI
                app.AnalyzeButton.Enable = 'on';
                app.updateStatus(['✓ Loaded: ' filename], app.SecondaryColor);
                
            catch ME
                uialert(app.UIFigure, ['Error: ' ME.message], 'Load Failed');
            end
        end
        
        function AnalyzeButtonPushed(app, ~, ~)
            if isempty(app.CurrentImage)
                return;
            end
            
            % Disable buttons during processing
            app.AnalyzeButton.Enable = 'off';
            app.LoadButton.Enable = 'off';
            
            if app.UseDeepLearning
                app.updateStatus('⏳ Analyzing with CNN (Deep Learning)...', app.DLColor);
            else
                app.updateStatus('⏳ Analyzing with SVM (Classical)...', app.AccentColor);
            end
            drawnow;
            
            try
                % Run analysis
                tic;
                if app.UseDeepLearning
                    results = analyzeHawkerFoodDL(app.CurrentImage);
                else
                    results = analyzeHawkerFood(app.CurrentImage);
                end
                processingTime = toc;
                app.CurrentResults = results;
                
                % Display processed image
                cla(app.ProcessedAxes);
                imshow(results.processedImage, 'Parent', app.ProcessedAxes);
                axis(app.ProcessedAxes, 'image');
                app.ProcessedAxes.XLim = [0.5, size(results.processedImage, 2) + 0.5];
                app.ProcessedAxes.YLim = [0.5, size(results.processedImage, 1) + 0.5];
                
                % Display segmented image
                cla(app.SegmentedAxes);
                imshow(results.segmentedImage, 'Parent', app.SegmentedAxes);
                axis(app.SegmentedAxes, 'image');
                app.SegmentedAxes.XLim = [0.5, size(results.segmentedImage, 2) + 0.5];
                app.SegmentedAxes.YLim = [0.5, size(results.segmentedImage, 1) + 0.5];
                
                % Update food name with nice formatting
                displayName = strrep(results.foodClass, '_', ' ');
                displayName = regexprep(displayName, '(\<\w)', '${upper($1)}');
                app.FoodNameLabel.Text = displayName;
                app.FoodNameLabel.FontColor = app.PrimaryColor;
                
                % Update confidence
                confidence = results.confidence * 100;
                app.ConfidenceMeter.Text = sprintf('%.0f%%', confidence);
                
                % Color confidence bar based on level
                if confidence >= 80
                    app.ConfidenceBar.BackgroundColor = app.SecondaryColor;
                    app.ConfidenceMeter.FontColor = app.SecondaryColor;
                elseif confidence >= 50
                    app.ConfidenceBar.BackgroundColor = app.AccentColor;
                    app.ConfidenceMeter.FontColor = app.AccentColor;
                else
                    app.ConfidenceBar.BackgroundColor = app.DangerColor;
                    app.ConfidenceMeter.FontColor = app.DangerColor;
                end
                
                % Update portion
                app.PortionLabel.Text = sprintf('Portion: %s (%.1fx)', ...
                    results.portionLabel, results.portionRatio);
                
                % Update nutrition
                app.CaloriesValueLabel.Text = sprintf('%d', results.calories);
                app.CaloriesValueLabel.FontColor = [1.0, 0.58, 0.0]; % Amber
                
                nutri = results.nutrition;
                % UPDATED: Individual labels for perfect alignment
                app.ProteinLabel.Text = sprintf('Protein: %.1f grams', nutri.protein);
                app.CarbsLabel.Text = sprintf('Carbohydrates: %.1f grams', nutri.carbs);
                app.FatLabel.Text = sprintf('Total Fat: %.1f grams', nutri.fat);
                app.DailyValueLabel.Text = sprintf('(%d%% Daily Intake)', nutri.caloriesDV);
                
                % Ensure they are visible and styled
                app.ProteinLabel.Visible = 'on';
                app.CarbsLabel.Visible = 'on';
                app.FatLabel.Visible = 'on';
                app.DailyValueLabel.Visible = 'on';
                
                % Cleared erroneous text reset
                
                % Update status with classifier info
                app.ProcessingTimeLabel.Text = sprintf('%.2fs', processingTime);
                if app.UseDeepLearning
                    classifierInfo = ' [CNN]';
                else
                    classifierInfo = ' [SVM]';
                end
                app.updateStatus(['✓ Analysis complete!' classifierInfo], app.SecondaryColor);
                
            catch ME
                uialert(app.UIFigure, ['Error: ' ME.message], 'Analysis Failed');
                app.updateStatus('✗ Analysis failed', app.DangerColor);
            end
            
            app.AnalyzeButton.Enable = 'on';
            app.LoadButton.Enable = 'on';
        end
        
        function ResetButtonPushed(app, ~, ~)
            app.clearResults();
            app.showPlaceholder(app.OriginalAxes, 'Drop image here');
            app.showPlaceholder(app.ProcessedAxes, '--');
            app.showPlaceholder(app.SegmentedAxes, '--');
        end
    end
    
    methods (Access = public)
        
        function app = HawkerFoodCalorieApp()
            createComponents(app);
            app.ProjectPath = fileparts(fileparts(mfilename('fullpath')));
            addpath(genpath(app.ProjectPath));
            
            % Initialize
            app.clearResults();
            app.showPlaceholder(app.OriginalAxes, 'Click "Load Image"');
            app.showPlaceholder(app.ProcessedAxes, '--');
            app.showPlaceholder(app.SegmentedAxes, '--');
            
            app.UIFigure.Visible = 'on';
        end
        
        function delete(app)
            delete(app.UIFigure);
        end
    end
    
    methods (Access = private)
        
        function createComponents(app)
            
            %% Main Figure
            app.UIFigure = uifigure('Name', 'Malaysian Hawker Food Calorie Estimator');
            app.UIFigure.Position = [100, 50, 1200, 850]; % Increased height for better visibility
            app.UIFigure.Color = app.BackgroundColor;
            app.UIFigure.Resize = 'on';
            
            %% Main Grid Layout
            app.MainGrid = uigridlayout(app.UIFigure, [3, 2]);
            app.MainGrid.RowHeight = {100, '1x', 40};
            app.MainGrid.ColumnWidth = {'2x', '1x'};
            app.MainGrid.Padding = [20, 15, 20, 15];
            app.MainGrid.RowSpacing = 10;
            app.MainGrid.ColumnSpacing = 20;
            app.MainGrid.BackgroundColor = app.BackgroundColor;
            
            %% ================ HEADER ================
            app.HeaderPanel = uipanel(app.MainGrid);
            app.HeaderPanel.Layout.Row = 1;
            app.HeaderPanel.Layout.Column = [1, 2];
            app.HeaderPanel.BackgroundColor = app.PrimaryColor;
            app.HeaderPanel.BorderType = 'none';
            
            headerGrid = uigridlayout(app.HeaderPanel, [2, 2]);
            headerGrid.RowHeight = {'1x', '1x'};
            headerGrid.ColumnWidth = {'1x', 150};
            headerGrid.Padding = [20, 10, 20, 10];
            headerGrid.BackgroundColor = app.PrimaryColor;
            
            app.TitleLabel = uilabel(headerGrid);
            app.TitleLabel.Text = '🍲 Malaysian Hawker Food Recognition';
            app.TitleLabel.FontSize = 22;
            app.TitleLabel.FontWeight = 'bold';
            app.TitleLabel.FontColor = 'white';
            app.TitleLabel.Layout.Row = 1;
            app.TitleLabel.Layout.Column = 1;
            
            app.SubtitleLabel = uilabel(headerGrid);
            app.SubtitleLabel.Text = 'Portion-Based Calorie Estimation System';
            app.SubtitleLabel.FontSize = 13;
            app.SubtitleLabel.FontColor = [0.9, 0.95, 1.0];
            app.SubtitleLabel.Layout.Row = 2;
            app.SubtitleLabel.Layout.Column = 1;
            
            %% ================ IMAGE PANEL (Left) ================
            app.ImagePanel = uipanel(app.MainGrid);
            app.ImagePanel.Layout.Row = 2;
            app.ImagePanel.Layout.Column = 1;
            app.ImagePanel.Title = '';
            app.ImagePanel.BackgroundColor = app.CardColor;
            app.ImagePanel.BorderType = 'none';
            
            app.ImageGrid = uigridlayout(app.ImagePanel, [2, 3]);
            app.ImageGrid.RowHeight = {30, '1x'};
            app.ImageGrid.ColumnWidth = {'1x', '1x', '1x'};
            app.ImageGrid.Padding = [15, 10, 15, 10];
            app.ImageGrid.RowSpacing = 5;
            app.ImageGrid.ColumnSpacing = 15;
            app.ImageGrid.BackgroundColor = app.CardColor;
            
            % Labels
            app.OriginalLabel = uilabel(app.ImageGrid);
            app.OriginalLabel.Text = '📷 Original';
            app.OriginalLabel.FontWeight = 'bold';
            app.OriginalLabel.FontSize = 13;
            app.OriginalLabel.FontColor = app.TextPrimary;
            app.OriginalLabel.HorizontalAlignment = 'center';
            app.OriginalLabel.Layout.Row = 1;
            app.OriginalLabel.Layout.Column = 1;
            
            app.ProcessedLabel = uilabel(app.ImageGrid);
            app.ProcessedLabel.Text = '🔧 Processed';
            app.ProcessedLabel.FontWeight = 'bold';
            app.ProcessedLabel.FontSize = 13;
            app.ProcessedLabel.FontColor = app.TextPrimary;
            app.ProcessedLabel.HorizontalAlignment = 'center';
            app.ProcessedLabel.Layout.Row = 1;
            app.ProcessedLabel.Layout.Column = 2;
            
            app.SegmentedLabel = uilabel(app.ImageGrid);
            app.SegmentedLabel.Text = '✂️ Segmented';
            app.SegmentedLabel.FontWeight = 'bold';
            app.SegmentedLabel.FontSize = 13;
            app.SegmentedLabel.FontColor = app.TextPrimary;
            app.SegmentedLabel.HorizontalAlignment = 'center';
            app.SegmentedLabel.Layout.Row = 1;
            app.SegmentedLabel.Layout.Column = 3;
            
            % Axes
            app.OriginalAxes = uiaxes(app.ImageGrid);
            app.OriginalAxes.Layout.Row = 2;
            app.OriginalAxes.Layout.Column = 1;
            app.OriginalAxes.XTick = [];
            app.OriginalAxes.YTick = [];
            app.OriginalAxes.Box = 'on';
            app.OriginalAxes.Color = [0.97, 0.97, 0.97];
            
            app.ProcessedAxes = uiaxes(app.ImageGrid);
            app.ProcessedAxes.Layout.Row = 2;
            app.ProcessedAxes.Layout.Column = 2;
            app.ProcessedAxes.XTick = [];
            app.ProcessedAxes.YTick = [];
            app.ProcessedAxes.Box = 'on';
            app.ProcessedAxes.Color = [0.97, 0.97, 0.97];
            
            app.SegmentedAxes = uiaxes(app.ImageGrid);
            app.SegmentedAxes.Layout.Row = 2;
            app.SegmentedAxes.Layout.Column = 3;
            app.SegmentedAxes.XTick = [];
            app.SegmentedAxes.YTick = [];
            app.SegmentedAxes.Box = 'on';
            app.SegmentedAxes.Color = [0.97, 0.97, 0.97];
            
            %% ================ CONTROL PANEL (Right) ================
            app.ControlPanel = uipanel(app.MainGrid);
            app.ControlPanel.Layout.Row = 2;
            app.ControlPanel.Layout.Column = 2;
            app.ControlPanel.Title = '';
            app.ControlPanel.BackgroundColor = app.CardColor;
            app.ControlPanel.BorderType = 'none';
            
            app.ControlGrid = uigridlayout(app.ControlPanel, [10, 1]);
            app.ControlGrid.RowHeight = {25, 35, 45, 45, 45, 20, 120, 20, 280, 20};
            app.ControlGrid.Padding = [20, 10, 20, 10];
            app.ControlGrid.RowSpacing = 8;
            app.ControlGrid.BackgroundColor = app.CardColor;
            
            % Classifier Selection
            app.ClassifierLabel = uilabel(app.ControlGrid);
            app.ClassifierLabel.Text = '🧠 Classification Method';
            app.ClassifierLabel.FontSize = 11;
            app.ClassifierLabel.FontWeight = 'bold';
            app.ClassifierLabel.FontColor = app.TextSecondary;
            app.ClassifierLabel.Layout.Row = 1;
            
            app.ClassifierDropdown = uidropdown(app.ControlGrid);
            app.ClassifierDropdown.Items = {'Classic ML (SVM)', 'Deep Learning (CNN)'};
            app.ClassifierDropdown.Value = 'Classic ML (SVM)';
            app.ClassifierDropdown.FontSize = 13;
            app.ClassifierDropdown.FontWeight = 'bold';
            app.ClassifierDropdown.BackgroundColor = [1, 1, 1];
            app.ClassifierDropdown.Layout.Row = 2;
            app.ClassifierDropdown.ValueChangedFcn = @app.ClassifierChanged;
            
            % Buttons
            app.LoadButton = uibutton(app.ControlGrid, 'push');
            app.LoadButton.Text = '📂  Load Image';
            app.LoadButton.FontSize = 14;
            app.LoadButton.FontWeight = 'bold';
            app.LoadButton.BackgroundColor = app.PrimaryColor;
            app.LoadButton.FontColor = 'white';
            app.LoadButton.Layout.Row = 3;
            app.LoadButton.ButtonPushedFcn = @app.LoadButtonPushed;
            
            app.AnalyzeButton = uibutton(app.ControlGrid, 'push');
            app.AnalyzeButton.Text = '🔍  Analyze Food';
            app.AnalyzeButton.FontSize = 14;
            app.AnalyzeButton.FontWeight = 'bold';
            app.AnalyzeButton.BackgroundColor = app.SecondaryColor;
            app.AnalyzeButton.FontColor = 'white';
            app.AnalyzeButton.Layout.Row = 4;
            app.AnalyzeButton.Enable = 'off';
            app.AnalyzeButton.ButtonPushedFcn = @app.AnalyzeButtonPushed;
            
            app.ResetButton = uibutton(app.ControlGrid, 'push');
            app.ResetButton.Text = '🔄  Reset';
            app.ResetButton.FontSize = 12;
            app.ResetButton.BackgroundColor = [0.94, 0.94, 0.95];
            app.ResetButton.FontColor = app.TextSecondary;
            app.ResetButton.Layout.Row = 5;
            app.ResetButton.ButtonPushedFcn = @app.ResetButtonPushed;
            
            divider1 = uilabel(app.ControlGrid);
            divider1.Text = '─────── Results ───────';
            divider1.FontColor = [0.8, 0.8, 0.8];
            divider1.HorizontalAlignment = 'center';
            divider1.Layout.Row = 6;
            
            % Results Panel
            app.ResultsPanel = uipanel(app.ControlGrid);
            app.ResultsPanel.Layout.Row = 7;
            app.ResultsPanel.BackgroundColor = [0.98, 0.99, 1.0];
            app.ResultsPanel.BorderType = 'none';
            
            resultsGrid = uigridlayout(app.ResultsPanel, [3, 2]);
            resultsGrid.RowHeight = {'1x', '1x', '1x'};
            resultsGrid.ColumnWidth = {'1x', 90};
            resultsGrid.Padding = [15, 10, 15, 10];
            resultsGrid.BackgroundColor = [0.98, 0.99, 1.0];
            
            app.FoodNameLabel = uilabel(resultsGrid);
            app.FoodNameLabel.Text = 'No food detected';
            app.FoodNameLabel.FontSize = 16;
            app.FoodNameLabel.FontWeight = 'bold';
            app.FoodNameLabel.FontColor = app.TextSecondary;
            app.FoodNameLabel.Layout.Row = 1;
            app.FoodNameLabel.Layout.Column = 1;
            
            app.ConfidenceMeter = uilabel(resultsGrid);
            app.ConfidenceMeter.Text = '-- %';
            app.ConfidenceMeter.FontSize = 32;
            app.ConfidenceMeter.FontWeight = 'bold';
            app.ConfidenceMeter.FontColor = app.TextSecondary;
            app.ConfidenceMeter.HorizontalAlignment = 'right';
            app.ConfidenceMeter.Layout.Row = [1, 2];
            app.ConfidenceMeter.Layout.Column = 2;
            
            app.ConfidenceBar = uilabel(resultsGrid);
            app.ConfidenceBar.Text = '';
            app.ConfidenceBar.BackgroundColor = [0.9, 0.9, 0.9];
            app.ConfidenceBar.Layout.Row = 2;
            app.ConfidenceBar.Layout.Column = 1;
            
            app.PortionLabel = uilabel(resultsGrid);
            app.PortionLabel.Text = 'Portion: --';
            app.PortionLabel.FontSize = 13;
            app.PortionLabel.FontColor = app.TextSecondary;
            app.PortionLabel.Layout.Row = 3;
            app.PortionLabel.Layout.Column = [1, 2];
            
            divider2 = uilabel(app.ControlGrid);
            divider2.Text = '────── Nutrition ──────';
            divider2.FontColor = [0.8, 0.8, 0.8];
            divider2.HorizontalAlignment = 'center';
            divider2.Layout.Row = 8;
            
            % Nutrition Panel
            app.NutritionPanel = uipanel(app.ControlGrid);
            app.NutritionPanel.Layout.Row = 9;
            app.NutritionPanel.BackgroundColor = app.CardColor;
            app.NutritionPanel.BorderType = 'line';
            app.NutritionPanel.BorderColor = [0.92, 0.92, 0.94];
            
            nutritionGrid = uigridlayout(app.NutritionPanel, [5, 1]);
            nutritionGrid.RowHeight = {70, 28, 28, 28, 35};
            nutritionGrid.Padding = [15, 10, 15, 10];
            nutritionGrid.RowSpacing = 2;
            nutritionGrid.BackgroundColor = app.CardColor;
            
            calRowGrid = uigridlayout(nutritionGrid, [1, 2]);
            calRowGrid.ColumnWidth = {'1x', '1x'};
            calRowGrid.Padding = [0, 0, 0, 0];
            calRowGrid.BackgroundColor = app.CardColor;
            calRowGrid.Layout.Row = 1;
            
            app.CaloriesValueLabel = uilabel(calRowGrid);
            app.CaloriesValueLabel.Text = '---';
            app.CaloriesValueLabel.FontSize = 40;
            app.CaloriesValueLabel.FontWeight = 'bold';
            app.CaloriesValueLabel.FontColor = [1.0, 0.58, 0.0];
            app.CaloriesValueLabel.HorizontalAlignment = 'right';
            app.CaloriesValueLabel.Layout.Column = 1;
            
            app.CaloriesUnitLabel = uilabel(calRowGrid);
            app.CaloriesUnitLabel.Text = ' kcal';
            app.CaloriesUnitLabel.FontSize = 16;
            app.CaloriesUnitLabel.FontWeight = 'bold';
            app.CaloriesUnitLabel.FontColor = [0.5, 0.5, 0.55];
            app.CaloriesUnitLabel.VerticalAlignment = 'bottom';
            app.CaloriesUnitLabel.Layout.Column = 2;
            
            % Nutrient Details
            app.ProteinLabel = uilabel(nutritionGrid);
            app.ProteinLabel.Text = 'Protein: --';
            app.ProteinLabel.FontSize = 13;
            app.ProteinLabel.Layout.Row = 2;
            
            app.CarbsLabel = uilabel(nutritionGrid);
            app.CarbsLabel.Text = 'Carbohydrates: --';
            app.CarbsLabel.FontSize = 13;
            app.CarbsLabel.Layout.Row = 3;
            
            app.FatLabel = uilabel(nutritionGrid);
            app.FatLabel.Text = 'Total Fat: --';
            app.FatLabel.FontSize = 13;
            app.FatLabel.Layout.Row = 4;
            
            app.DailyValueLabel = uilabel(nutritionGrid);
            app.DailyValueLabel.Text = '-- Daily Value';
            app.DailyValueLabel.FontSize = 14;
            app.DailyValueLabel.FontWeight = 'bold';
            app.DailyValueLabel.FontColor = app.TextSecondary;
            app.DailyValueLabel.Layout.Row = 5;
            
            % Processing time
            app.ProcessingTimeLabel = uilabel(app.ControlGrid);
            app.ProcessingTimeLabel.Text = '';
            app.ProcessingTimeLabel.FontSize = 10;
            app.ProcessingTimeLabel.FontName = 'Arial';
            app.ProcessingTimeLabel.FontColor = app.TextSecondary;
            app.ProcessingTimeLabel.HorizontalAlignment = 'right';
            app.ProcessingTimeLabel.Layout.Row = 10;
            
            %% ================ STATUS BAR ================
            app.StatusPanel = uipanel(app.MainGrid);
            app.StatusPanel.Layout.Row = 3;
            app.StatusPanel.Layout.Column = [1, 2];
            app.StatusPanel.BackgroundColor = app.CardColor;
            app.StatusPanel.BorderType = 'none';
            
            statusGrid = uigridlayout(app.StatusPanel, [1, 2]);
            statusGrid.ColumnWidth = {'1x', 200};
            statusGrid.Padding = [15, 5, 15, 5];
            statusGrid.BackgroundColor = app.CardColor;
            
            app.StatusLabel = uilabel(statusGrid);
            app.StatusLabel.Text = 'Ready';
            app.StatusLabel.FontSize = 11;
            app.StatusLabel.FontColor = app.TextSecondary;
            app.StatusLabel.Layout.Row = 1;
            app.StatusLabel.Layout.Column = 1;
            
            teamLabel = uilabel(statusGrid);
            teamLabel.Text = 'CSC566 Team One | 2024';
            teamLabel.FontSize = 10;
            teamLabel.FontColor = [0.7, 0.7, 0.7];
            teamLabel.HorizontalAlignment = 'right';
            teamLabel.Layout.Row = 1;
            teamLabel.Layout.Column = 2;
        end
        
        function ClassifierChanged(app, ~, ~)
            selectedValue = app.ClassifierDropdown.Value;
            if contains(selectedValue, 'CNN')
                app.UseDeepLearning = true;
                app.AnalyzeButton.BackgroundColor = app.DLColor;
                app.AnalyzeButton.Text = '🧠  Analyze (CNN)';
                app.updateStatus('Mode: Deep Learning (CNN)', app.DLColor);
            else
                app.UseDeepLearning = false;
                app.AnalyzeButton.BackgroundColor = app.SecondaryColor;
                app.AnalyzeButton.Text = '🔍  Analyze Food';
                app.updateStatus('Mode: Classical (SVM)', app.SecondaryColor);
            end
        end
    end
end
