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
        
        % Colors
        PrimaryColor = [0.16, 0.50, 0.73]      % Blue
        SecondaryColor = [0.18, 0.80, 0.44]    % Green
        AccentColor = [0.90, 0.49, 0.13]       % Orange
        DangerColor = [0.91, 0.30, 0.24]       % Red
        DLColor = [0.58, 0.26, 0.70]           % Purple (for DL)
        BackgroundColor = [0.95, 0.96, 0.98]   % Light Gray
        CardColor = [1, 1, 1]                   % White
        TextPrimary = [0.15, 0.15, 0.15]       % Dark
        TextSecondary = [0.45, 0.45, 0.45]     % Gray
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
            app.ProteinLabel.Text = 'Protein: -- g';
            app.CarbsLabel.Text = 'Carbs: -- g';
            app.FatLabel.Text = 'Fat: -- g';
            app.DailyValueLabel.Text = '-- % Daily Value';
            
            app.ProcessingTimeLabel.Text = '';
            
            app.CurrentImage = [];
            app.CurrentResults = [];
            
            app.AnalyzeButton.Enable = 'off';
            app.updateStatus('Ready. Load an image to begin.', app.TextSecondary);
        end
        
        function showPlaceholder(app, ax, text)
            cla(ax);
            ax.Color = [0.97, 0.97, 0.97];
            text(ax, 0.5, 0.5, text, ...
                 'HorizontalAlignment', 'center', ...
                 'VerticalAlignment', 'middle', ...
                 'FontSize', 11, 'Color', [0.6, 0.6, 0.6]);
            ax.XLim = [0 1];
            ax.YLim = [0 1];
        end
    end
    
    methods (Access = private)
        
        function LoadButtonPushed(app, ~)
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
                app.OriginalAxes.Color = 'none';
                
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
        
        function AnalyzeButtonPushed(app, ~)
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
                app.ProcessedAxes.Color = 'none';
                
                % Display segmented image
                cla(app.SegmentedAxes);
                imshow(results.segmentedImage, 'Parent', app.SegmentedAxes);
                app.SegmentedAxes.Color = 'none';
                
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
                app.CaloriesValueLabel.FontColor = app.AccentColor;
                
                nutrition = results.nutrition;
                app.ProteinLabel.Text = sprintf('Protein: %.1f g', nutrition.protein);
                app.CarbsLabel.Text = sprintf('Carbs: %.1f g', nutrition.carbs);
                app.FatLabel.Text = sprintf('Fat: %.1f g', nutrition.fat);
                app.DailyValueLabel.Text = sprintf('%d%% Daily Value', nutrition.caloriesDV);
                
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
        
        function ResetButtonPushed(app, ~)
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
            app.UIFigure.Position = [50, 50, 1280, 800];
            app.UIFigure.Color = app.BackgroundColor;
            app.UIFigure.Resize = 'on';
            
            %% Main Grid Layout
            app.MainGrid = uigridlayout(app.UIFigure, [3, 2]);
            app.MainGrid.RowHeight = {70, '1x', 40};
            app.MainGrid.ColumnWidth = {'2x', '1x'};
            app.MainGrid.Padding = [20, 20, 20, 20];
            app.MainGrid.RowSpacing = 15;
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
            headerGrid.Padding = [25, 10, 25, 10];
            headerGrid.BackgroundColor = app.PrimaryColor;
            
            app.TitleLabel = uilabel(headerGrid);
            app.TitleLabel.Text = '🍜 Malaysian Hawker Food Recognition';
            app.TitleLabel.FontSize = 22;
            app.TitleLabel.FontWeight = 'bold';
            app.TitleLabel.FontColor = 'white';
            app.TitleLabel.Layout.Row = 1;
            app.TitleLabel.Layout.Column = 1;
            
            app.SubtitleLabel = uilabel(headerGrid);
            app.SubtitleLabel.Text = 'Portion-Based Calorie Estimation System';
            app.SubtitleLabel.FontSize = 12;
            app.SubtitleLabel.FontColor = [0.85, 0.90, 0.95];
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
            app.ImageGrid.RowHeight = {25, '1x'};
            app.ImageGrid.ColumnWidth = {'1x', '1x', '1x'};
            app.ImageGrid.Padding = [15, 15, 15, 15];
            app.ImageGrid.RowSpacing = 8;
            app.ImageGrid.ColumnSpacing = 15;
            app.ImageGrid.BackgroundColor = app.CardColor;
            
            % Labels
            app.OriginalLabel = uilabel(app.ImageGrid);
            app.OriginalLabel.Text = '📷 Original';
            app.OriginalLabel.FontWeight = 'bold';
            app.OriginalLabel.FontSize = 12;
            app.OriginalLabel.FontColor = app.TextPrimary;
            app.OriginalLabel.HorizontalAlignment = 'center';
            app.OriginalLabel.Layout.Row = 1;
            app.OriginalLabel.Layout.Column = 1;
            
            app.ProcessedLabel = uilabel(app.ImageGrid);
            app.ProcessedLabel.Text = '🔧 Processed';
            app.ProcessedLabel.FontWeight = 'bold';
            app.ProcessedLabel.FontSize = 12;
            app.ProcessedLabel.FontColor = app.TextPrimary;
            app.ProcessedLabel.HorizontalAlignment = 'center';
            app.ProcessedLabel.Layout.Row = 1;
            app.ProcessedLabel.Layout.Column = 2;
            
            app.SegmentedLabel = uilabel(app.ImageGrid);
            app.SegmentedLabel.Text = '✂️ Segmented';
            app.SegmentedLabel.FontWeight = 'bold';
            app.SegmentedLabel.FontSize = 12;
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
            app.ControlGrid.RowHeight = {25, 35, 50, 50, 50, 30, 80, 30, '1x', 30};
            app.ControlGrid.Padding = [20, 20, 20, 20];
            app.ControlGrid.RowSpacing = 10;
            app.ControlGrid.BackgroundColor = app.CardColor;
            
            % Classifier Selection Label
            app.ClassifierLabel = uilabel(app.ControlGrid);
            app.ClassifierLabel.Text = '🧠 Classification Method';
            app.ClassifierLabel.FontSize = 11;
            app.ClassifierLabel.FontWeight = 'bold';
            app.ClassifierLabel.FontColor = app.TextPrimary;
            app.ClassifierLabel.Layout.Row = 1;
            
            % Classifier Dropdown
            app.ClassifierDropdown = uidropdown(app.ControlGrid);
            app.ClassifierDropdown.Items = {'SVM (Classical)', 'CNN (Deep Learning)'};
            app.ClassifierDropdown.Value = 'SVM (Classical)';
            app.ClassifierDropdown.FontSize = 12;
            app.ClassifierDropdown.BackgroundColor = [0.98, 0.98, 0.98];
            app.ClassifierDropdown.Layout.Row = 2;
            app.ClassifierDropdown.ValueChangedFcn = @app.ClassifierChanged;
            
            % Load Button
            app.LoadButton = uibutton(app.ControlGrid, 'push');
            app.LoadButton.Text = '📂  Load Image';
            app.LoadButton.FontSize = 14;
            app.LoadButton.FontWeight = 'bold';
            app.LoadButton.BackgroundColor = app.PrimaryColor;
            app.LoadButton.FontColor = 'white';
            app.LoadButton.Layout.Row = 3;
            app.LoadButton.ButtonPushedFcn = @app.LoadButtonPushed;
            
            % Analyze Button
            app.AnalyzeButton = uibutton(app.ControlGrid, 'push');
            app.AnalyzeButton.Text = '🔍  Analyze Food';
            app.AnalyzeButton.FontSize = 14;
            app.AnalyzeButton.FontWeight = 'bold';
            app.AnalyzeButton.BackgroundColor = app.SecondaryColor;
            app.AnalyzeButton.FontColor = 'white';
            app.AnalyzeButton.Layout.Row = 4;
            app.AnalyzeButton.Enable = 'off';
            app.AnalyzeButton.ButtonPushedFcn = @app.AnalyzeButtonPushed;
            
            % Reset Button
            app.ResetButton = uibutton(app.ControlGrid, 'push');
            app.ResetButton.Text = '🔄  Reset';
            app.ResetButton.FontSize = 12;
            app.ResetButton.BackgroundColor = [0.9, 0.9, 0.9];
            app.ResetButton.FontColor = app.TextSecondary;
            app.ResetButton.Layout.Row = 5;
            app.ResetButton.ButtonPushedFcn = @app.ResetButtonPushed;
            
            % Divider
            divider1 = uilabel(app.ControlGrid);
            divider1.Text = '─────── Results ───────';
            divider1.FontColor = [0.7, 0.7, 0.7];
            divider1.HorizontalAlignment = 'center';
            divider1.Layout.Row = 6;
            
            % Results Panel
            app.ResultsPanel = uipanel(app.ControlGrid);
            app.ResultsPanel.Layout.Row = 7;
            app.ResultsPanel.BackgroundColor = [0.98, 0.98, 0.98];
            app.ResultsPanel.BorderType = 'none';
            
            resultsGrid = uigridlayout(app.ResultsPanel, [3, 2]);
            resultsGrid.RowHeight = {'1x', '1x', '1x'};
            resultsGrid.ColumnWidth = {'1x', 80};
            resultsGrid.Padding = [12, 8, 12, 8];
            resultsGrid.BackgroundColor = [0.98, 0.98, 0.98];
            
            app.FoodNameLabel = uilabel(resultsGrid);
            app.FoodNameLabel.Text = 'No food detected';
            app.FoodNameLabel.FontSize = 16;
            app.FoodNameLabel.FontWeight = 'bold';
            app.FoodNameLabel.FontColor = app.TextSecondary;
            app.FoodNameLabel.Layout.Row = 1;
            app.FoodNameLabel.Layout.Column = 1;
            
            app.ConfidenceMeter = uilabel(resultsGrid);
            app.ConfidenceMeter.Text = '-- %';
            app.ConfidenceMeter.FontSize = 18;
            app.ConfidenceMeter.FontWeight = 'bold';
            app.ConfidenceMeter.FontColor = app.TextSecondary;
            app.ConfidenceMeter.HorizontalAlignment = 'right';
            app.ConfidenceMeter.Layout.Row = 1;
            app.ConfidenceMeter.Layout.Column = 2;
            
            app.ConfidenceBar = uilabel(resultsGrid);
            app.ConfidenceBar.Text = '';
            app.ConfidenceBar.BackgroundColor = [0.9, 0.9, 0.9];
            app.ConfidenceBar.Layout.Row = 2;
            app.ConfidenceBar.Layout.Column = [1, 2];
            
            app.PortionLabel = uilabel(resultsGrid);
            app.PortionLabel.Text = 'Portion: --';
            app.PortionLabel.FontSize = 12;
            app.PortionLabel.FontColor = app.TextSecondary;
            app.PortionLabel.Layout.Row = 3;
            app.PortionLabel.Layout.Column = [1, 2];
            
            % Nutrition Divider
            divider2 = uilabel(app.ControlGrid);
            divider2.Text = '────── Nutrition ──────';
            divider2.FontColor = [0.7, 0.7, 0.7];
            divider2.HorizontalAlignment = 'center';
            divider2.Layout.Row = 8;
            
            % Nutrition Panel
            app.NutritionPanel = uipanel(app.ControlGrid);
            app.NutritionPanel.Layout.Row = 9;
            app.NutritionPanel.BackgroundColor = app.CardColor;
            app.NutritionPanel.BorderType = 'line';
            app.NutritionPanel.BorderColor = [0.9, 0.9, 0.9];
            
            nutritionGrid = uigridlayout(app.NutritionPanel, [5, 2]);
            nutritionGrid.RowHeight = {50, 25, 25, 25, 25};
            nutritionGrid.ColumnWidth = {'1x', '1x'};
            nutritionGrid.Padding = [15, 15, 15, 15];
            nutritionGrid.RowSpacing = 5;
            nutritionGrid.BackgroundColor = app.CardColor;
            
            % Calories (big display)
            app.CaloriesValueLabel = uilabel(nutritionGrid);
            app.CaloriesValueLabel.Text = '---';
            app.CaloriesValueLabel.FontSize = 36;
            app.CaloriesValueLabel.FontWeight = 'bold';
            app.CaloriesValueLabel.FontColor = app.TextSecondary;
            app.CaloriesValueLabel.HorizontalAlignment = 'center';
            app.CaloriesValueLabel.Layout.Row = 1;
            app.CaloriesValueLabel.Layout.Column = 1;
            
            app.CaloriesUnitLabel = uilabel(nutritionGrid);
            app.CaloriesUnitLabel.Text = 'kcal';
            app.CaloriesUnitLabel.FontSize = 14;
            app.CaloriesUnitLabel.FontColor = app.TextSecondary;
            app.CaloriesUnitLabel.VerticalAlignment = 'bottom';
            app.CaloriesUnitLabel.Layout.Row = 1;
            app.CaloriesUnitLabel.Layout.Column = 2;
            
            % Macros
            app.ProteinLabel = uilabel(nutritionGrid);
            app.ProteinLabel.Text = 'Protein: -- g';
            app.ProteinLabel.FontSize = 12;
            app.ProteinLabel.FontColor = app.TextPrimary;
            app.ProteinLabel.Layout.Row = 2;
            app.ProteinLabel.Layout.Column = [1, 2];
            
            app.CarbsLabel = uilabel(nutritionGrid);
            app.CarbsLabel.Text = 'Carbs: -- g';
            app.CarbsLabel.FontSize = 12;
            app.CarbsLabel.FontColor = app.TextPrimary;
            app.CarbsLabel.Layout.Row = 3;
            app.CarbsLabel.Layout.Column = [1, 2];
            
            app.FatLabel = uilabel(nutritionGrid);
            app.FatLabel.Text = 'Fat: -- g';
            app.FatLabel.FontSize = 12;
            app.FatLabel.FontColor = app.TextPrimary;
            app.FatLabel.Layout.Row = 4;
            app.FatLabel.Layout.Column = [1, 2];
            
            app.DailyValueLabel = uilabel(nutritionGrid);
            app.DailyValueLabel.Text = '-- % Daily Value';
            app.DailyValueLabel.FontSize = 11;
            app.DailyValueLabel.FontColor = app.TextSecondary;
            app.DailyValueLabel.FontAngle = 'italic';
            app.DailyValueLabel.Layout.Row = 5;
            app.DailyValueLabel.Layout.Column = [1, 2];
            
            % Processing time
            app.ProcessingTimeLabel = uilabel(app.ControlGrid);
            app.ProcessingTimeLabel.Text = '';
            app.ProcessingTimeLabel.FontSize = 10;
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
        
        function ClassifierChanged(app, ~)
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
