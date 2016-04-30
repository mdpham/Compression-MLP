% Call test.m after setup.m
training=MLP.PATTERN;

% Max number of epochs
EPOCH_MAX=100;
for epoch=0:EPOCH_MAX
  train_pattern(training);
  display(['epoch: ' num2str(epoch) 'sse: ' num2str(MLP.SSE())]);
end;
