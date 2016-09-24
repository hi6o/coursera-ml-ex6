function [C, sigma] = dataset3Params(X, y, Xval, yval)
%EX6PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = EX6PARAMS(X, y, Xval, yval) returns your choice of C and
%   sigma. You should complete this function to return the optimal C and
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example,
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using
%        mean(double(predictions ~= yval))
%

% モデル作成に時間がかかるため、実装途中ではfor文を回す回数を少なくした方がいいかも。

values = [0.01;0.03;0.1;0.3;1;3;10;30];
m = size(values,1);
error = [];
error_list = [];
for i = 1:m
  C = values(i);
  for j = 1:m
    sigma = values(j);

    % 上記で設定したCとsigmaでモデルを作成
    model= svmTrain(X, y, C, @(x1, x2) gaussianKernel(x1, x2, sigma));

    %　作成したモデルで予測
    predictions = svmPredict(model,Xval);

    % 予測が外れていた割合をerrorとして取得
    error = mean(double(predictions ~= yval));

    % error値とその時のCとsigmaをlistに保存
    error_list = [error_list ; C sigma error];
  endfor
endfor

% errorlistの3列目にerror値が入ってるので、それが最小値の行数をixとして取得
[x,ix] = min(error_list(:,3))

% ix行にあるCとsigmaを取得する
C = error_list(ix,1)
sigma = error_list(ix,2)

% =========================================================================

end
