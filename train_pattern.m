function MLP=train_pattern(pattern)
% Column vector input pattern
% Input patterns are indexed [1 N], outputs should map to associated value
%   NN takes serialized values and should output signal (or image)

  input_pattern=pattern(:,1); %Serialized values
  output_pattern=pattern(:,2); %Associated values

  global MLP eta alpha N L M;

  % % Input to hidden layer;
  % hidden_sum=MLP.BIAS_IH + MLP.WEIGHTS_IH*input_pattern;%add bias weight
  % MLP.HIDDEN=1.0./(1.0+exp(-hidden_sum)); %Sigmoid transfer function
  % % Hidden to output layer
  % output_sum=MLP.BIAS_HO + MLP.WEIGHTS_HO*MLP.HIDDEN;
  % MLP.OUTPUT=1.0./(1.0+exp(-output_sum));
  MLP.OUTPUT=output_mlp(input_pattern);
  MLP.DELTA_O=(MLP.PATTERN(:,2)-MLP.OUTPUT).*MLP.OUTPUT.*(1.0- MLP.OUTPUT);

  % Back-propagate errors to hidden layer
  sum_DOW=MLP.WEIGHTS_HO'*MLP.DELTA_O;
  % display(MLP.HIDDEN);
  % display(sum_DOW.*MLP.HIDDEN.*(1.0-MLP.HIDDEN));
  MLP.DELTA_H=sum_DOW.*MLP.HIDDEN.*(1.0-MLP.HIDDEN);

  % Update input-hidden weights
  % Bias
  MLP.DELTA_BIAS_IH=eta.*MLP.DELTA_H + alpha.*MLP.DELTA_BIAS_IH;
  MLP.BIAS_IH=MLP.BIAS_IH+MLP.DELTA_BIAS_IH;
  % Weights
  MLP.DELTA_WEIGHTS_IH=eta.*MLP.DELTA_H*input_pattern' + alpha.*MLP.DELTA_WEIGHTS_IH;
  % display(MLP.WEIGHTS_IH(1:10));
  MLP.WEIGHTS_IH=MLP.WEIGHTS_IH+MLP.DELTA_WEIGHTS_IH;
  % display(MLP.WEIGHTS_IH(1:10));

  % Update hidden-output weights
  MLP.DELTA_BIAS_HO=eta.*MLP.DELTA_O + alpha.*MLP.DELTA_BIAS_HO;
  MLP.BIAS_HO=MLP.BIAS_HO+MLP.DELTA_BIAS_HO;
  MLP.DELTA_WEIGHTS_HO=eta.*MLP.DELTA_O*MLP.HIDDEN' + alpha.*MLP.DELTA_WEIGHTS_HO;
  % display(MLP.WEIGHTS_HO(1:10));
  MLP.WEIGHTS_HO=MLP.WEIGHTS_HO+MLP.DELTA_WEIGHTS_HO;
  % display(MLP.WEIGHTS_HO(1:10));
end
