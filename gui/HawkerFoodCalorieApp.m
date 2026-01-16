classdef HawkerFoodCalorieApp < matlab.apps.AppBase

    % Properties that correspond to app components
    properties (Access = public)
        UIFigure                   matlab.ui.Figure
        MainGrid                   matlab.ui.container.GridLayout
        LeftPanel                  matlab.ui.container.Panel
        RightPanel                 matlab.ui.container.Panel
        
        % Left Panel Components
        TitleLabel                 matlab.ui.control.Label
        LoadImageButton            matlab.ui.control.Button
        AnalyzeButton              matlab.ui.control.Button
        ResetButton                matlab.ui.control.Button
        OriginalImageAxes          matlab.ui.control.UIAxes
        ProcessedImageAxes         matlab.ui.control.UIAxes
        SegmentedImageAxes         matlab.ui.control.UIAxes
        StatusLabel                matlab.ui.control.Label
        
        % Right Panel Components
        ResultsTitleLabel          matlab.ui.control.Label
        FoodClassLabel             matlab.ui.control.Label
        ConfidenceLabel            matlab.ui.control.Label
        PortionLabel               matlab.ui.control.Label
        CaloriesLabel              matlab.ui.control.Label
        ProteinLabel               matlab.ui.control.Label
        CarbsLabel                 matlab.ui.control.Label
        FatLabel                   matlab.ui.control.Label
        DVLabel                    matlab.ui.control.Label
        DescriptionLabel           matlab.ui.control.Label
        ProcessingTimeLabel        matlab.ui.control.Label
    end
    
    % Properties for data storage
    properties (Access = private)
        CurrentImage               % Loaded image
        CurrentResults             % Analysis results
        ProjectPath                % Project root path
    end
    
    methods (Access = private)
        
        function updateStatus(app, message)
            app.StatusLabel.Text = message;
            drawnow;
        end
        
        function clearResults(app)
            % Clear image axes
            cla(app.OriginalImageAxes);
            cla(app.ProcessedImageAxes);
            cla(app.SegmentedImageAxes);
            
            % Reset labels
            app.FoodClassLabel.Text = 'Food: -';
            app.ConfidenceLabel.Text = 'Confidence: -';
            app.PortionLabel.Text = 'Portion: -';
            app.CaloriesLabel.Text = 'Calories: -';
            app.ProteinLabel.Text = 'Protein: -';
            app.CarbsLabel.Text = 'Carbohydrates: -';
            app.FatLabel.Text = 'Fat: -';
            app.DVLabel.Text = 'Daily Value: -';
            app.DescriptionLabel.Text = '';
            app.ProcessingTimeLabel.Text = 'Processing Time: -';
            
            % Clear data
            app.CurrentImage = [];
            app.CurrentResults = [];
            
            app.updateStatus('Ready. Load an image to begin.');
        end
        
        function displayImage(app, img, ax, titleText)
            cla(ax);
            imshow(img, 'Parent', ax);
            title(ax, titleText, 'FontWeight', 'bold');
        end
    end
    
    % Callbacks
    methods (Access = private)
        
        function LoadImageButtonPushed(app, ~)
            % Open file dialog
            [filename, pathname] = uigetfile(...
                {'*.jpg;*.jpeg;*.png;*.bmp', 'Image Files (*.jpg, *.png, *.bmp)'; ...
                 '*.*', 'All Files (*.*)'}, ...
                'Select a food image');
            
            if isequal(filename, 0)
                return;  % User cancelled
            end
            
            % Load image
            imagePath = fullfile(pathname, filename);
            try
                app.CurrentImage = imread(imagePath);
                app.displayImage(app.CurrentImage, app.OriginalImageAxes, 'Original Image');
                
                % Clear previous results
                cla(app.ProcessedImageAxes);
                cla(app.SegmentedImageAxes);
                title(app.ProcessedImageAxes, 'Pre-processed', 'FontWeight', 'bold');
                title(app.SegmentedImageAxes, 'Segmentation', 'FontWeight', 'bold');
                
                app.updateStatus(['Loaded: ', filename, '. Click "Analyze" to process.']);
                app.AnalyzeButton.Enable = 'on';
                
            catch ME
                uialert(app.UIFigure, ['Error loading image: ', ME.message], 'Load Error');
            end
        end
        
        function AnalyzeButtonPushed(app, ~)
            if isempty(app.CurrentImage)
                uialert(app.UIFigure, 'Please load an image first.', 'No Image');
                return;
            end
            
            app.updateStatus('Analyzing image... Please wait.');
            app.AnalyzeButton.Enable = 'off';
            drawnow;
            
            try
                % Run analysis
                results = analyzeHawkerFood(app.CurrentImage);
                app.CurrentResults = results;
                
                % Display processed images
                app.displayImage(results.processedImage, app.ProcessedImageAxes, 'Pre-processed');
                app.displayImage(results.segmentedImage, app.SegmentedImageAxes, 'Segmentation');
                
                % Update results labels
                displayName = upper(strrep(results.foodClass, '_', ' '));
                app.FoodClassLabel.Text = ['Food: ', displayName];
                app.ConfidenceLabel.Text = sprintf('Confidence: %.1f%%', results.confidence * 100);
                app.PortionLabel.Text = sprintf('Portion: %s (%.2fx)', results.portionLabel, results.portionRatio);
                app.CaloriesLabel.Text = sprintf('Calories: %d kcal', results.calories);
                
                nutrition = results.nutrition;
                app.ProteinLabel.Text = sprintf('Protein: %.1f g', nutrition.protein);
                app.CarbsLabel.Text = sprintf('Carbohydrates: %.1f g', nutrition.carbs);
                app.FatLabel.Text = sprintf('Fat: %.1f g', nutrition.fat);
                app.DVLabel.Text = sprintf('Daily Value: %d%% of 2000 kcal', nutrition.caloriesDV);
                app.DescriptionLabel.Text = nutrition.description;
                app.ProcessingTimeLabel.Text = sprintf('Processing Time: %.2f s', results.processingTime);
                
                app.updateStatus('Analysis complete!');
                
            catch ME
                uialert(app.UIFigure, ['Analysis error: ', ME.message], 'Error');
                app.updateStatus('Analysis failed. Check if classifier is trained.');
            end
            
            app.AnalyzeButton.Enable = 'on';
        end
        
        function ResetButtonPushed(app, ~)
            app.clearResults();
        end
    end
    
    % App creation and deletion
    methods (Access = public)
        
        function app = HawkerFoodCalorieApp()
            % Create and configure components
            createComponents(app);
            
            % Get project path
            app.ProjectPath = fileparts(mfilename('fullpath'));
            
            % Add paths
            addpath(genpath(app.ProjectPath));
            
            % Initialize
            app.clearResults();
            
            % Show the figure
            app.UIFigure.Visible = 'on';
        end
        
        function delete(app)
            delete(app.UIFigure);
        end
    end
    
    methods (Access = private)
        
        function createComponents(app)
            % Create main figure
            app.UIFigure = uifigure('Name', 'Malaysian Hawker Food Calorie Estimator');
            app.UIFigure.Position = [100, 100, 1200, 700];
            app.UIFigure.Color = [0.94, 0.94, 0.94];
            
            % Create main grid
            app.MainGrid = uigridlayout(app.UIFigure, [1, 2]);
            app.MainGrid.ColumnWidth = {'2x', '1x'};
            
            % Create left panel
            app.LeftPanel = uipanel(app.MainGrid);
            app.LeftPanel.Title = 'Image Analysis';
            app.LeftPanel.FontWeight = 'bold';
            
            leftGrid = uigridlayout(app.LeftPanel, [5, 3]);
            leftGrid.RowHeight = {30, 30, '1x', '1x', 25};
            leftGrid.ColumnWidth = {'1x', '1x', '1x'};
            
            % Title
            app.TitleLabel = uilabel(leftGrid);
            app.TitleLabel.Text = 'Malaysian Hawker Food Recognition System';
            app.TitleLabel.FontSize = 16;
            app.TitleLabel.FontWeight = 'bold';
            app.TitleLabel.HorizontalAlignment = 'center';
            app.TitleLabel.Layout.Row = 1;
            app.TitleLabel.Layout.Column = [1, 3];
            
            % Buttons
            app.LoadImageButton = uibutton(leftGrid, 'push');
            app.LoadImageButton.Text = 'Load Image';
            app.LoadImageButton.Layout.Row = 2;
            app.LoadImageButton.Layout.Column = 1;
            app.LoadImageButton.ButtonPushedFcn = @app.LoadImageButtonPushed;
            
            app.AnalyzeButton = uibutton(leftGrid, 'push');
            app.AnalyzeButton.Text = 'Analyze';
            app.AnalyzeButton.Layout.Row = 2;
            app.AnalyzeButton.Layout.Column = 2;
            app.AnalyzeButton.Enable = 'off';
            app.AnalyzeButton.ButtonPushedFcn = @app.AnalyzeButtonPushed;
            
            app.ResetButton = uibutton(leftGrid, 'push');
            app.ResetButton.Text = 'Reset';
            app.ResetButton.Layout.Row = 2;
            app.ResetButton.Layout.Column = 3;
            app.ResetButton.ButtonPushedFcn = @app.ResetButtonPushed;
            
            % Image axes
            app.OriginalImageAxes = uiaxes(leftGrid);
            app.OriginalImageAxes.Layout.Row = 3;
            app.OriginalImageAxes.Layout.Column = 1;
            title(app.OriginalImageAxes, 'Original', 'FontWeight', 'bold');
            
            app.ProcessedImageAxes = uiaxes(leftGrid);
            app.ProcessedImageAxes.Layout.Row = 3;
            app.ProcessedImageAxes.Layout.Column = 2;
            title(app.ProcessedImageAxes, 'Pre-processed', 'FontWeight', 'bold');
            
            app.SegmentedImageAxes = uiaxes(leftGrid);
            app.SegmentedImageAxes.Layout.Row = 3;
            app.SegmentedImageAxes.Layout.Column = 3;
            title(app.SegmentedImageAxes, 'Segmentation', 'FontWeight', 'bold');
            
            % Status label
            app.StatusLabel = uilabel(leftGrid);
            app.StatusLabel.Text = 'Ready';
            app.StatusLabel.Layout.Row = 5;
            app.StatusLabel.Layout.Column = [1, 3];
            
            % Create right panel
            app.RightPanel = uipanel(app.MainGrid);
            app.RightPanel.Title = 'Analysis Results';
            app.RightPanel.FontWeight = 'bold';
            
            rightGrid = uigridlayout(app.RightPanel, [12, 1]);
            rightGrid.RowHeight = repmat({25}, 1, 12);
            
            % Results title
            app.ResultsTitleLabel = uilabel(rightGrid);
            app.ResultsTitleLabel.Text = 'NUTRITIONAL INFORMATION';
            app.ResultsTitleLabel.FontWeight = 'bold';
            app.ResultsTitleLabel.FontSize = 14;
            app.ResultsTitleLabel.Layout.Row = 1;
            
            % Food class
            app.FoodClassLabel = uilabel(rightGrid);
            app.FoodClassLabel.Text = 'Food: -';
            app.FoodClassLabel.FontSize = 14;
            app.FoodClassLabel.FontWeight = 'bold';
            app.FoodClassLabel.Layout.Row = 2;
            
            % Confidence
            app.ConfidenceLabel = uilabel(rightGrid);
            app.ConfidenceLabel.Text = 'Confidence: -';
            app.ConfidenceLabel.Layout.Row = 3;
            
            % Portion
            app.PortionLabel = uilabel(rightGrid);
            app.PortionLabel.Text = 'Portion: -';
            app.PortionLabel.Layout.Row = 4;
            
            % Calories
            app.CaloriesLabel = uilabel(rightGrid);
            app.CaloriesLabel.Text = 'Calories: -';
            app.CaloriesLabel.FontSize = 16;
            app.CaloriesLabel.FontWeight = 'bold';
            app.CaloriesLabel.FontColor = [0.8, 0.2, 0.2];
            app.CaloriesLabel.Layout.Row = 5;
            
            % Macronutrients
            app.ProteinLabel = uilabel(rightGrid);
            app.ProteinLabel.Text = 'Protein: -';
            app.ProteinLabel.Layout.Row = 6;
            
            app.CarbsLabel = uilabel(rightGrid);
            app.CarbsLabel.Text = 'Carbohydrates: -';
            app.CarbsLabel.Layout.Row = 7;
            
            app.FatLabel = uilabel(rightGrid);
            app.FatLabel.Text = 'Fat: -';
            app.FatLabel.Layout.Row = 8;
            
            % Daily value
            app.DVLabel = uilabel(rightGrid);
            app.DVLabel.Text = 'Daily Value: -';
            app.DVLabel.Layout.Row = 9;
            
            % Description
            app.DescriptionLabel = uilabel(rightGrid);
            app.DescriptionLabel.Text = '';
            app.DescriptionLabel.WordWrap = 'on';
            app.DescriptionLabel.FontAngle = 'italic';
            app.DescriptionLabel.Layout.Row = 10;
            
            % Processing time
            app.ProcessingTimeLabel = uilabel(rightGrid);
            app.ProcessingTimeLabel.Text = 'Processing Time: -';
            app.ProcessingTimeLabel.FontColor = [0.5, 0.5, 0.5];
            app.ProcessingTimeLabel.Layout.Row = 12;
        end
    end
end
