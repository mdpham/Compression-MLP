function output=output_mlp(input)

  global MLP;

  % Input to hidden layer;
  hidden_sum=MLP.BIAS_IH + MLP.WEIGHTS_IH*input;%add bias weight
  MLP.HIDDEN=1.0./(1.0+exp(-hidden_sum)); %Sigmoid transfer function
  % Hidden to output layer
  output_sum=MLP.BIAS_HO + MLP.WEIGHTS_HO*MLP.HIDDEN;
  output=1.0./(1.0+exp(-output_sum));
end
