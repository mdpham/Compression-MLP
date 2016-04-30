% Globals
sigmoid=@(x)1.0./(1.0+exp(-x));


% N inputs, L hidden nodes, M outputs
global N L M;
N=64;L=6;M=64;
global eta alpha;
eta=0.2;alpha=0.3;
% Initiate multilayer perceptron object
global MLP;
MLP=mlp();

MLP.BIAS_IH=2*(rand(L,1)-0.5)*0.01;
MLP.WEIGHTS_IH=2*(rand(L,N)-0.5)*0.01;
MLP.HIDDEN=zeros(L,1);
MLP.BIAS_HO=2*(rand(M,1)-0.5)*0.01;
MLP.WEIGHTS_HO=2*(rand(M,L)-0.5)*0.01;
MLP.OUTPUT=zeros(M,1);

MLP.DELTA_O=zeros(M,1);
MLP.DELTA_BIAS_HO=zeros(M,1);
MLP.DELTA_WEIGHTS_HO=zeros(M,L);

MLP.DELTA_H=zeros(L,1);
MLP.DELTA_BIAS_IH=zeros(L,1);
MLP.DELTA_WEIGHTS_IH=zeros(L,N);

% SERIALIZED DATA WE WANT TO COMPRESS
% [avena avenaFS]=audioread('avena.mp3');
% avena=1.0./(1.0+exp(-mean(avena')))';
MLP.PATTERN=[linspace(1,N,N)' rand(N,1)];
