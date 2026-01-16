%% CALCULATE CALORIES - Portion-Adjusted Calorie Calculation
% Calculates estimated calories based on food class and portion size
%
% Syntax:
%   calories = calculateCalories(foodClass, portionRatio)
%   [calories, nutrition] = calculateCalories(foodClass, portionRatio)
%
% Inputs:
%   foodClass    - Food class name (string)
%   portionRatio - Portion ratio relative to standard (1.0 = standard)
%
% Outputs:
%   calories  - Estimated calorie count
%   nutrition - Struct with adjusted nutritional values:
%               .calories, .protein, .carbs, .fat, .portionRatio

function [calories, nutrition] = calculateCalories(foodClass, portionRatio)
    %% Input validation
    if nargin < 2
        portionRatio = 1.0;
    end
    
    % Ensure valid portion ratio
    portionRatio = max(0.1, min(2.5, portionRatio));
    
    %% Get base nutritional values
    foodInfo = foodDatabase(foodClass);
    
    %% Calculate portion-adjusted values
    calories = round(foodInfo.baseCalories * portionRatio);
    
    if nargout > 1
        nutrition = struct();
        nutrition.foodClass = foodClass;
        nutrition.displayName = foodInfo.displayName;
        nutrition.description = foodInfo.description;
        nutrition.referenceServing = foodInfo.referenceServing;
        nutrition.portionRatio = portionRatio;
        nutrition.calories = calories;
        nutrition.protein = round(foodInfo.protein * portionRatio, 1);
        nutrition.carbs = round(foodInfo.carbs * portionRatio, 1);
        nutrition.fat = round(foodInfo.fat * portionRatio, 1);
        
        % Calculate percentage of daily values (based on 2000 kcal diet)
        nutrition.caloriesDV = round(calories / 2000 * 100);
        nutrition.proteinDV = round(nutrition.protein / 50 * 100);  % 50g RDA
        nutrition.carbsDV = round(nutrition.carbs / 300 * 100);     % 300g RDA
        nutrition.fatDV = round(nutrition.fat / 65 * 100);           % 65g RDA
    end
end
